"""Resolve human-authored Hydra dataset cards into the canonical stage spec.

The public YAML format describes the dataset: source, columns, timeline, and
outputs. The runtime stages still consume the stricter ``DatasetSpec`` shape.
Keeping this translation in one module prevents Hydra authoring ergonomics from
leaking into planning, resume, and execution code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_authoring_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical DatasetSpec payload.

    Canonical payloads are passed through unchanged. Authoring payloads are
    identified by their dataset-card sections such as ``source`` / ``columns`` /
    ``outputs`` / ``recipe`` and expanded into ``convert`` / ``tokenize`` /
    ``materialize`` sections.
    """

    data = dict(payload)
    if _is_canonical_payload(data) or not _is_authoring_payload(data):
        return data

    name = _require_str(data, "name")
    recipe = _mapping(data.get("recipe"), "recipe")
    source = _mapping(data.get("source"), "source")
    columns = _mapping(data.get("columns"), "columns")
    timeline = _mapping(data.get("timeline"), "timeline")
    outputs = _mapping(data.get("outputs"), "outputs")
    tokenizer = _mapping(data.get("tokenizer"), "tokenizer")
    conversion = _mapping(data.get("conversion"), "conversion")
    tokenization = _mapping(data.get("tokenization"), "tokenization")
    materialization = _mapping(data.get("materialization"), "materialization")
    if "sft" in materialization:
        raise ValueError(
            "materialization.sft is not supported. SFT config was removed "
            "until the SFT assembler lands with its schema and tests."
        )

    source_type = _str_value(source.get("type") or recipe.get("source_type"), "source.type")
    pipeline_mode = _str_value(recipe.get("mode"), "recipe.mode")

    canonical: dict[str, Any] = {"name": name}

    if outputs.get("shar_dir") is not None:
        canonical["convert"] = _build_convert(
            source_type=source_type,
            source=source,
            columns=columns,
            timeline=timeline,
            outputs=outputs,
            tokenizer=tokenizer,
            conversion=conversion,
            language=data.get("language"),
        )

    if outputs.get("tokenized_dir") is not None:
        canonical["tokenize"] = _build_tokenize(
            name=name,
            pipeline_mode=pipeline_mode,
            outputs=outputs,
            tokenizer=tokenizer,
            tokenization=tokenization,
            recipe=recipe,
        )

    canonical["materialize"] = _build_materialize(
        name=name,
        pipeline_mode=pipeline_mode,
        outputs=outputs,
        tokenizer=tokenizer,
        materialization=materialization,
        recipe=recipe,
    )
    return canonical


def _is_canonical_payload(payload: Mapping[str, Any]) -> bool:
    convert = payload.get("convert")
    return isinstance(convert, Mapping) and (
        "family" in convert or "input" in convert or "output" in convert
    )


def _is_authoring_payload(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in ("recipe", "source", "columns", "timeline", "outputs"))


def _build_convert(
    *,
    source_type: str,
    source: Mapping[str, Any],
    columns: Mapping[str, Any],
    timeline: Mapping[str, Any],
    outputs: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    conversion: Mapping[str, Any],
    language: Any,
) -> dict[str, Any]:
    family = _source_family(source_type)
    return {
        "enabled": bool(conversion.get("enabled", True)),
        "family": family,
        "input": _source_input(family, source),
        "output": {
            "shar_dir": _require_str(outputs, "shar_dir"),
            "shard_size": _get(conversion, "shard_size"),
            "shar_format": _get(conversion, "shar_format"),
            "target_sr": _get(conversion, "target_sr"),
            "text_tokenizer": _conversion_text_tokenizer(conversion, tokenizer),
            "num_workers": _get(conversion, "num_workers"),
            "resampling_backend": _get(conversion, "resampling_backend"),
            "mp_start_method": _get(conversion, "mp_start_method"),
            "read_batch_size": _get(conversion, "read_batch_size"),
        },
        "metadata": _metadata(columns, timeline, language=language),
    }


def _build_tokenize(
    *,
    name: str,
    pipeline_mode: str,
    outputs: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    tokenization: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    mode, audio_text_format = _tokenize_mode(pipeline_mode, recipe)
    return {
        "tokenizer": {
            "path": _require_str(tokenizer, "path"),
            "sampling_rate": _get(tokenizer, "sampling_rate"),
            "torch_compile": _get(tokenizer, "torch_compile"),
            "trim_last_tokens": _get(tokenizer, "trim_last_tokens"),
        },
        "output": {
            "output_dir": _require_str(outputs, "tokenized_dir"),
            "output_name": outputs.get("name", name),
            "shar_index_filename": _get(tokenization, "shar_index_filename"),
        },
        "mode": mode,
        "audio_text_format": audio_text_format,
        "audio_text_task": tokenization.get("audio_text_task", recipe.get("audio_text_task")),
        "input_shar_dir": tokenization.get("input_shar_dir"),
        "partitioning": tokenization.get("partitioning"),
        "filter": {
            "min_duration": _get(tokenization, "min_duration"),
            "max_duration": _get(tokenization, "max_duration"),
            "min_sample_rate": _get(tokenization, "min_sample_rate"),
            "min_rms_db": _get(tokenization, "min_rms_db"),
            "normalize_peak_db": _get(tokenization, "normalize_peak_db"),
        },
        "dataloader": {
            "num_workers": _get(tokenization, "num_workers"),
            "prefetch_factor": _get(tokenization, "prefetch_factor"),
            "max_batch_duration": _get(tokenization, "max_batch_duration"),
            "checkpoint_interval_batches": _get(tokenization, "checkpoint_interval_batches"),
            "max_batch_cuts": _get(tokenization, "max_batch_cuts"),
            "num_buckets": _get(tokenization, "num_buckets"),
            "bucket_buffer_size": _get(tokenization, "bucket_buffer_size"),
            "sampler_shuffle": _get(tokenization, "sampler_shuffle"),
            "sampler_seed": _get(tokenization, "sampler_seed"),
            "quadratic_duration": _get(tokenization, "quadratic_duration"),
        },
        "wandb": tokenization.get("wandb", {}),
    }


def _build_materialize(
    *,
    name: str,
    pipeline_mode: str,
    outputs: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    materialization: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    interleave_defaults = _mapping(materialization.get("interleave"), "materialization.interleave")
    interleave_enabled = bool(_materialize_value(
        materialization,
        interleave_defaults,
        "enabled",
        recipe.get("materialize_interleave", pipeline_mode == "audio_text_interleaved"),
    ))
    tokenized_dir = outputs.get("tokenized_dir")
    interleaved_dir = outputs.get("interleaved_dir")
    if interleaved_dir is None and tokenized_dir is not None:
        interleaved_dir = f"{tokenized_dir}/{outputs.get('name', name)}_interleaved"
    return {
        "interleave": {
            "enabled": interleave_enabled,
            "strategy": _materialize_value(materialization, interleave_defaults, "strategy"),
            "cache_dir": _materialize_value(materialization, interleave_defaults, "cache_dir"),
            "output_dir": _materialize_value(
                materialization,
                interleave_defaults,
                "output_dir",
                interleaved_dir,
            ),
            "tokenizer_path": _materialize_value(
                materialization,
                interleave_defaults,
                "tokenizer_path",
                tokenizer.get("path"),
            ),
            "max_seq_len": _materialize_value(materialization, interleave_defaults, "max_seq_len"),
            "max_gap_sec": _materialize_value(materialization, interleave_defaults, "max_gap_sec"),
            "seq_threshold": _materialize_value(materialization, interleave_defaults, "seq_threshold"),
            "transcribe_ratio": _materialize_value(
                materialization,
                interleave_defaults,
                "transcribe_ratio",
            ),
            "num_workers": _materialize_value(materialization, interleave_defaults, "num_workers"),
            "tmp_dir": _materialize_value(materialization, interleave_defaults, "tmp_dir"),
        },
    }


def _source_family(source_type: str) -> str:
    if source_type in {"parquet", "hf", "wds", "audio_dir", "lhotse_recipe"}:
        return source_type
    raise ValueError(
        "source.type must be one of parquet, hf, wds, audio_dir, lhotse_recipe; "
        f"got {source_type!r}"
    )


def _source_input(family: str, source: Mapping[str, Any]) -> dict[str, Any]:
    if family == "parquet":
        return {
            "parquet_dir": _require_str(source, "path"),
            "parquet_glob": _get(source, "files"),
        }
    if family == "hf":
        return {
            "arrow_dir": source.get("path"),
            "arrow_glob": _get(source, "files"),
            "arrow_files": source.get("arrow_files"),
        }
    if family == "wds":
        vad = _mapping(source.get("vad"), "source.vad")
        return {
            "wds_shards": _str_list(source.get("shards", source.get("path"))),
            "min_sr": _get(source, "min_sr"),
            "no_mono_downmix": _get(source, "no_mono_downmix"),
            "vad_segmentation": bool(_get(vad, "enabled")),
            "vad_per_shard_dir": _get(vad, "per_shard_dir"),
            "vad_max_chunk_sec": _get(vad, "max_chunk_sec"),
            "vad_min_chunk_sec": _get(vad, "min_chunk_sec"),
            "vad_sample_rate": _get(vad, "sample_rate"),
            "vad_max_merge_gap_sec": _get(vad, "max_merge_gap_sec"),
            "vad_max_duration_sec": _get(vad, "max_duration_sec"),
        }
    if family == "audio_dir":
        vad = _mapping(source.get("vad"), "source.vad")
        return {
            "audio_root": _require_str(source, "path"),
            "jsonl_files": _str_list(source.get("manifests", source.get("files"))),
            "audio_ext": _get(source, "audio_ext"),
            "min_sr": _get(source, "min_sr"),
            "no_mono_downmix": _get(source, "no_mono_downmix"),
            "vad_max_chunk_sec": _get(vad, "max_chunk_sec"),
            "vad_min_chunk_sec": _get(vad, "min_chunk_sec"),
            "vad_sample_rate": _get(vad, "sample_rate"),
            "vad_max_merge_gap_sec": _get(vad, "max_merge_gap_sec"),
            "vad_max_duration_sec": _get(vad, "max_duration_sec"),
        }
    if family == "lhotse_recipe":
        return {
            "recipe": _require_str(source, "recipe"),
            "corpus_dir": _require_str(source, "path"),
            "split": _require_str(source, "split"),
            "recipe_kwargs": _get(source, "recipe_kwargs"),
            "min_sample_rate": _get(source, "min_sample_rate"),
            "trim_to_supervisions": _get(source, "trim_to_supervisions"),
            "shar_index_filename": _get(source, "shar_index_filename"),
        }
    raise AssertionError(f"unreachable source family {family!r}")


def _metadata(
    columns: Mapping[str, Any],
    timeline: Mapping[str, Any],
    *,
    language: Any,
) -> dict[str, Any]:
    out = {
        "audio_column": _get(columns, "audio"),
        "text_column": _get(columns, "text"),
        "duration_column": _get(columns, "duration"),
        "id_column": _get(columns, "id"),
        "id_prefix": _get(columns, "id_prefix"),
        "language_column": _get(columns, "language_column"),
        "language": language if language is not None else columns.get("language"),
        "custom_columns": _str_list(_get(columns, "keep")),
        "constant_custom": _mapping(columns.get("constant"), "columns.constant"),
        "derived_custom": _mapping(columns.get("derived"), "columns.derived"),
        "text_tokenize_custom_columns": _str_list(_get(columns, "text_tokenize")),
        "input_clip_id_parser": _get(timeline, "parser"),
        "external_metadata": _get(columns, "external_metadata"),
        "custom_fields": _str_list(_get(columns, "custom_fields")),
        "id_field": _get(columns, "id_field"),
        "text_field": _get(columns, "text_field"),
    }
    for src, dst in (
        ("source_id", "source_id_column"),
        ("clip_num", "clip_num_column"),
        ("clip_start", "clip_start_column"),
        ("clip_end", "clip_end_column"),
        ("clip_duration", "clip_duration_column"),
        ("chunks", "chunks_column"),
    ):
        out[dst] = timeline.get(src)
    return out


def _tokenize_mode(pipeline_mode: str, recipe: Mapping[str, Any]) -> tuple[str, str]:
    mode = recipe.get("tokenize_mode")
    fmt = recipe.get("audio_text_format")
    if mode and fmt:
        return str(mode), str(fmt)
    if pipeline_mode == "audio_only":
        return "audio_only", "direct"
    if pipeline_mode == "audio_text_direct":
        return "audio_text", "direct"
    if pipeline_mode == "audio_text_interleaved":
        return "audio_text", "interleaved"
    raise ValueError(
        "recipe.mode must be one of audio_only, audio_text_direct, "
        f"audio_text_interleaved; got {pipeline_mode!r}"
    )


def _conversion_text_tokenizer(
    conversion: Mapping[str, Any], tokenizer: Mapping[str, Any]
) -> Any:
    if "text_tokenizer" in conversion:
        return conversion["text_tokenizer"]
    return tokenizer.get("text_tokenizer")


def _materialize_value(
    materialization: Mapping[str, Any],
    interleave_defaults: Mapping[str, Any],
    key: str,
    fallback: Any = None,
) -> Any:
    """Resolve materialize authoring values without letting null defaults win.

    Dataset cards may set concise top-level overrides such as
    ``materialization.max_gap_sec`` while recipe defaults provide the full
    ``materialization.interleave`` shape. A nested ``null`` default must not
    suppress those top-level values or derived output paths.
    """
    value = materialization.get(key)
    if value is not None:
        return value
    value = interleave_defaults.get(key)
    if value is not None:
        return value
    return fallback


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    return _str_value(value, key)


def _str_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise TypeError("expected a string or list of strings")


def _get(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(
            f"Missing Hydra default for {key!r}. Put policy defaults in the recipe YAML, "
            "not in the authoring resolver."
        )
    return mapping[key]
