"""Config-driven prepare stage adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_tokenization.config.schema import DatasetSpec
from audio_tokenization.utils.prepare_data import prepare_parquet_to_shar


def _id_column_to_cli_list(value: str | list[str] | None) -> list[str] | None:
    # argparse stores --id-column as a list (nargs="*"); preserve that shape
    # so list-vs-str distinctions reach the fingerprint untouched.
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return list(value) or None
    raise TypeError(f"Unsupported id_column shape: {type(value).__name__}")


def build_prepare_namespace(spec: DatasetSpec) -> argparse.Namespace:
    prepare = spec.prepare
    if prepare.family != "parquet":
        raise NotImplementedError(
            f"Config-driven prepare currently supports family='parquet' only; got {prepare.family!r}"
        )

    return argparse.Namespace(
        parquet_dir=Path(prepare.input.parquet_dir),
        parquet_glob=prepare.input.parquet_glob,
        shar_dir=Path(prepare.output.shar_dir),
        shard_size=prepare.output.shard_size,
        shar_format=prepare.output.shar_format,
        target_sr=prepare.output.target_sr,
        text_tokenizer=prepare.output.text_tokenizer,
        num_workers=prepare.output.num_workers,
        resampling_backend=prepare.output.resampling_backend,
        mp_start_method=prepare.output.mp_start_method,
        read_batch_size=prepare.output.read_batch_size,
        id_column=_id_column_to_cli_list(prepare.metadata.id_column),
        audio_column=prepare.metadata.audio_column,
        text_column=prepare.metadata.text_column,
        duration_column=prepare.metadata.duration_column,
        language_column=prepare.metadata.language_column,
        language=prepare.metadata.language,
        custom_columns=list(prepare.metadata.custom_columns) or None,
        text_tokenize_custom_columns=list(prepare.metadata.text_tokenize_custom_columns) or None,
        external_metadata=prepare.metadata.external_metadata,
        custom_fields=list(prepare.metadata.custom_fields) or None,
        id_field=prepare.metadata.id_field,
        text_field=prepare.metadata.text_field,
        input_clip_id_parser=prepare.metadata.input_clip_id_parser,
    )


def run_prepare(spec: DatasetSpec):
    if not spec.prepare.enabled:
        return {"skipped": True, "reason": "prepare.disabled"}

    args = build_prepare_namespace(spec)

    if spec.prepare.family == "parquet":
        return prepare_parquet_to_shar.run(args)

    raise NotImplementedError(
        f"Unsupported config-driven prepare family {spec.prepare.family!r}"
    )
