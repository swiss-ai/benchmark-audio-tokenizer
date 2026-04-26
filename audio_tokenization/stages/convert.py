"""Resolved-plan convert stage adapter."""

from __future__ import annotations

import importlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from audio_tokenization.config.schema import DatasetSpec, PrepareSpec
from audio_tokenization.contracts.artifacts import SHAR_INDEX_FILENAME
from audio_tokenization.prepare.cli import expand_path_patterns
from audio_tokenization.prepare.constants import PREPARE_STATE_FILE, SUCCESS_MARKER_FILE
from audio_tokenization.prepare.runtime import resolve_num_workers, validate_prepare_runtime
from audio_tokenization.stages._plans import ResolvedStagePlan, disabled_stage_plan
from audio_tokenization.stages._resume import try_skip_if_complete


_DISPATCH: dict[str, str] = {
    "parquet": "audio_tokenization.prepare.prepare_parquet_to_shar",
    "hf": "audio_tokenization.prepare.prepare_hf_to_shar",
    "wds": "audio_tokenization.prepare.prepare_wds_to_shar",
    "audio_dir": "audio_tokenization.prepare.prepare_audio_dir_to_shar",
    "lhotse_recipe": "audio_tokenization.prepare.prepare_lhotse_recipe_to_shar",
}

logger = logging.getLogger(__name__)


def resolve_convert_plan(spec: DatasetSpec) -> ResolvedStagePlan:
    if spec.convert is None or not spec.convert.enabled:
        return disabled_stage_plan(stage="convert", reason="convert.disabled")

    prepare_spec = _resolve_prepare_spec(spec.convert)
    resolved_inputs, input_summary = _resolve_convert_inputs(prepare_spec)
    output_dir = Path(prepare_spec.output.shar_dir)
    state_path = output_dir / PREPARE_STATE_FILE
    success_marker = output_dir / SUCCESS_MARKER_FILE

    return ResolvedStagePlan(
        stage="convert",
        enabled=True,
        reason=None,
        inputs=input_summary,
        outputs={
            "shar_dir": str(output_dir),
            "state_file": str(state_path),
            "success_marker": str(success_marker),
        },
        effective=_effective_convert_values(prepare_spec, resolved_inputs),
        fingerprint=prepare_spec.fingerprint_payload(),
        output_dir=output_dir,
        state_path=state_path,
        success_marker=success_marker,
        preflight=lambda: _preflight_convert_plan(prepare_spec, resolved_inputs),
        execute=lambda resume: _execute_convert_plan(prepare_spec, resume=resume),
    )


def run_convert(spec: DatasetSpec, *, resume: bool = True) -> dict[str, Any]:
    if spec.convert is None or not spec.convert.enabled:
        return {"skipped": True, "reason": "convert.disabled"}

    # Skip BEFORE plan resolution so a completed convert can be reused on
    # nodes that no longer have the raw inputs mounted or the prepare-time
    # runtime deps (ffmpeg, polars, ...) installed. Plan resolution eagerly
    # globs raw inputs and preflight loads the text tokenizer; both fail
    # loudly in those environments and would block legitimate resumes.
    skipped = try_skip_if_complete(
        output_dir=Path(spec.convert.output.shar_dir),
        state_filename=PREPARE_STATE_FILE,
        fingerprint=spec.convert.fingerprint_payload(),
        stage_label="convert",
        resume=resume,
        logger=logger,
    )
    if skipped is not None:
        _ensure_convert_shar_manifest(spec.convert)
        return skipped

    plan = resolve_convert_plan(spec)
    plan.preflight()
    return plan.execute(resume)


def _resolve_prepare_spec(spec: PrepareSpec) -> PrepareSpec:
    # The lhotse_recipe runner does ``range(num_workers)`` directly, so it
    # needs a concrete int when None comes through. Hardcoded to 64 (not the
    # SLURM-aware resolver) because lhotse_recipe is the only family that
    # fingerprints num_workers; a SLURM-derived value would invalidate resume
    # across nodes with different ``SLURM_CPUS_PER_TASK``. Other families
    # call ensure_worker_assignment which resolves None at runtime without
    # touching the fingerprint.
    if spec.family != "lhotse_recipe" or spec.output.num_workers is not None:
        return spec
    return spec.model_copy(
        update={"output": spec.output.model_copy(update={"num_workers": 64})},
    )


def _resolve_convert_inputs(spec: PrepareSpec) -> tuple[list[str], dict[str, Any]]:
    i = spec.input
    if spec.family == "parquet":
        parquet_dir = Path(i.parquet_dir)
        resolved = sorted(str(p) for p in parquet_dir.glob(i.parquet_glob))
        if not resolved:
            raise FileNotFoundError(f"No files match {parquet_dir / i.parquet_glob}")
        return resolved, {
            "family": spec.family,
            "parquet_dir": str(parquet_dir),
            "parquet_glob": i.parquet_glob,
            "resolved_inputs": resolved,
        }
    if spec.family == "hf":
        if i.arrow_files:
            resolved = expand_path_patterns(i.arrow_files)
            source = {"arrow_files": list(i.arrow_files)}
        elif i.arrow_dir:
            arrow_dir = Path(i.arrow_dir)
            resolved = sorted(str(p) for p in arrow_dir.glob(i.arrow_glob))
            source = {"arrow_dir": str(arrow_dir), "arrow_glob": i.arrow_glob}
        else:
            raise ValueError("convert.input requires arrow_dir or arrow_files")
        if not resolved:
            raise FileNotFoundError("No arrow files resolved for HF input")
        return resolved, {
            "family": spec.family,
            **source,
            "resolved_inputs": resolved,
        }
    if spec.family == "wds":
        resolved = expand_path_patterns(i.wds_shards)
        if not resolved:
            raise FileNotFoundError(f"No files match patterns: {i.wds_shards}")
        return resolved, {
            "family": spec.family,
            "wds_shards": list(i.wds_shards),
            "resolved_inputs": resolved,
        }
    if spec.family == "audio_dir":
        resolved = expand_path_patterns(i.jsonl_files)
        if not resolved:
            raise FileNotFoundError(f"No files match patterns: {i.jsonl_files}")
        return resolved, {
            "family": spec.family,
            "audio_root": i.audio_root,
            "jsonl_files": resolved,
            "audio_ext": i.audio_ext,
        }
    if spec.family == "lhotse_recipe":
        return [], {
            "family": spec.family,
            "recipe": i.recipe,
            "corpus_dir": i.corpus_dir,
            "split": i.split,
            "recipe_kwargs": i.recipe_kwargs,
        }
    raise ValueError(f"Unsupported convert family {spec.family!r}")


def _effective_convert_values(spec: PrepareSpec, resolved_inputs: list[str]) -> dict[str, Any]:
    # Empty input list (lhotse_recipe) means no fan-out cap; only SLURM/CPU applies.
    num_inputs = len(resolved_inputs) or None
    return {
        "family": spec.family,
        "effective_num_workers": resolve_num_workers(spec.output.num_workers, num_inputs=num_inputs),
        "shard_size": spec.output.shard_size,
        "target_sr": spec.output.target_sr,
        "shar_format": spec.output.shar_format,
        "text_tokenizer": spec.output.text_tokenizer,
        "resampling_backend": spec.output.resampling_backend,
        "resolved_input_count": len(resolved_inputs),
    }


def _preflight_convert_plan(spec: PrepareSpec, resolved_inputs: list[str]) -> None:
    i, o = spec.input, spec.output
    if spec.family == "parquet":
        parquet_dir = Path(i.parquet_dir)
        if not parquet_dir.is_dir():
            raise NotADirectoryError(f"Parquet input dir not found: {parquet_dir}")
        validate_prepare_runtime(
            resampling_backend=o.resampling_backend,
            require_ffmpeg=False,
            text_tokenizer_path=o.text_tokenizer,
        )
        return

    if spec.family == "hf":
        if i.arrow_dir is not None and not Path(i.arrow_dir).is_dir():
            raise NotADirectoryError(f"Arrow dir not found: {i.arrow_dir}")
        validate_prepare_runtime(
            resampling_backend=o.resampling_backend,
            require_ffmpeg=False,
            text_tokenizer_path=o.text_tokenizer,
        )
        return

    if spec.family == "wds":
        validate_prepare_runtime(
            resampling_backend=o.resampling_backend,
            require_ffmpeg=True,
            text_tokenizer_path=o.text_tokenizer,
        )
        if i.vad_segmentation:
            if i.vad_per_shard_dir is None:
                raise ValueError("vad_per_shard_dir is required with vad_segmentation")
            if not Path(i.vad_per_shard_dir).is_dir():
                raise NotADirectoryError(f"VAD per-shard directory not found: {i.vad_per_shard_dir}")
            _validate_vad_thresholds(i)
        return

    if spec.family == "audio_dir":
        audio_root = Path(i.audio_root)
        if not audio_root.is_dir():
            raise NotADirectoryError(f"Audio root not found: {audio_root}")
        validate_prepare_runtime(
            resampling_backend=o.resampling_backend,
            require_ffmpeg=False,
            text_tokenizer_path=None,
        )
        _validate_vad_thresholds(i)
        return

    if spec.family == "lhotse_recipe":
        validate_prepare_runtime(
            resampling_backend=None,
            require_ffmpeg=False,
            text_tokenizer_path=o.text_tokenizer,
        )
        if not Path(i.corpus_dir).exists():
            raise FileNotFoundError(f"Corpus dir not found: {i.corpus_dir}")
        json.loads(i.recipe_kwargs)
        return

    raise ValueError(f"Unsupported convert family {spec.family!r}")


def _validate_vad_thresholds(input_spec: Any) -> None:
    if input_spec.vad_max_chunk_sec <= 0:
        raise ValueError("vad_max_chunk_sec must be > 0")
    if input_spec.vad_min_chunk_sec < 0:
        raise ValueError("vad_min_chunk_sec must be >= 0")
    if input_spec.vad_min_chunk_sec > input_spec.vad_max_chunk_sec:
        raise ValueError("vad_min_chunk_sec must be <= vad_max_chunk_sec")
    if input_spec.vad_sample_rate <= 0:
        raise ValueError("vad_sample_rate must be > 0")
    if input_spec.vad_max_merge_gap_sec < 0:
        raise ValueError("vad_max_merge_gap_sec must be >= 0")


def _execute_convert_plan(spec: PrepareSpec, *, resume: bool) -> dict[str, Any]:
    shar_dir = Path(spec.output.shar_dir)
    # Mirrors run_with_resume's skip path for tokenize/materialize: avoids
    # the slow validate_shar_directory pass on every restart of a completed
    # convert. On drift we still fall through to the runner, which raises
    # via write_prepare_state_for_spec.
    skipped = try_skip_if_complete(
        output_dir=shar_dir,
        state_filename=PREPARE_STATE_FILE,
        fingerprint=spec.fingerprint_payload(),
        stage_label="convert",
        resume=resume,
        logger=logger,
    )
    if skipped is not None:
        _ensure_convert_shar_manifest(spec)
        return skipped

    if not resume and shar_dir.is_dir():
        logger.warning("Removing %s for resume=false re-run.", shar_dir)
        shutil.rmtree(shar_dir)
    runner = importlib.import_module(_DISPATCH[spec.family]).run
    result = runner(spec) or {}
    _ensure_convert_shar_manifest(spec)
    return result


def _ensure_convert_shar_manifest(spec: PrepareSpec) -> None:
    """Backfill the durable SHAR manifest after convert success or resume skip.

    Prepare runners are still family-specific, so the stage adapter owns the
    cross-family handoff contract: every completed SHAR root should expose
    ``_shar_work_manifest.json`` for tokenization planning, regardless of
    whether the run was fresh or skipped by resume.
    """

    from audio_tokenization.pipelines.lhotse.planning import (
        SHAR_WORK_MANIFEST_FILE,
        write_shar_work_manifest,
    )

    shar_dir = Path(spec.output.shar_dir)
    manifest_path = shar_dir / SHAR_WORK_MANIFEST_FILE
    if manifest_path.is_file():
        return
    manifest = write_shar_work_manifest(
        shar_dir,
        index_name=_convert_shar_index_filename(spec),
    )
    logger.info(
        "Wrote SHAR work manifest at %s: work_units=%s hours=%.2f",
        manifest_path,
        len(manifest.work_units),
        sum(unit.duration_sec for unit in manifest.work_units) / 3600.0,
    )


def _convert_shar_index_filename(spec: PrepareSpec) -> str:
    if spec.family == "lhotse_recipe":
        return spec.input.shar_index_filename
    return SHAR_INDEX_FILENAME
