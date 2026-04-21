"""Resolved-plan tokenize stage adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audio_tokenization.config.schema import DatasetSpec
from audio_tokenization.prepare.constants import SUCCESS_MARKER_FILE
from audio_tokenization.stages._plans import ResolvedStagePlan, disabled_stage_plan
from audio_tokenization.stages._provenance import (
    build_tokenize_resume_fingerprint,
    read_prepare_provenance,
)
from audio_tokenization.stages._resume import run_with_resume


logger = logging.getLogger(__name__)


TOKENIZE_STATE_FILE = "tokenize_state.json"


def resolve_tokenize_plan(spec: DatasetSpec) -> ResolvedStagePlan:
    if spec.tokenize is None:
        return disabled_stage_plan(stage="tokenize", reason="tokenize.disabled")

    input_shar_dirs = _resolve_input_shar_dirs(spec)
    pipeline_cfg = _build_pipeline_cfg(spec, input_shar_dirs)
    final_output_dir = _resolve_final_output_dir(pipeline_cfg)
    fingerprint = build_tokenize_resume_fingerprint(
        spec,
        input_shar_dirs=input_shar_dirs,
        prepare_provenance=read_prepare_provenance(input_shar_dirs),
    )
    input_was_explicit = spec.tokenize.input_shar_dir is not None

    return ResolvedStagePlan(
        stage="tokenize",
        enabled=True,
        reason=None,
        inputs={
            "resolved_input_shar_dirs": input_shar_dirs,
            "input_was_explicit": input_was_explicit,
        },
        outputs={
            "output_dir": str(final_output_dir),
            "state_file": str(final_output_dir / TOKENIZE_STATE_FILE),
            "success_marker": str(final_output_dir / SUCCESS_MARKER_FILE),
        },
        effective={
            "mode": spec.tokenize.mode,
            "audio_text_format": spec.tokenize.audio_text_format,
            "audio_text_task": spec.tokenize.audio_text_task,
            "effective_tokenizer_path": spec.tokenize.tokenizer.path,
            "effective_num_workers": spec.tokenize.dataloader.num_workers,
            "resolved_output_dir": str(final_output_dir),
        },
        fingerprint=fingerprint,
        output_dir=final_output_dir,
        state_path=final_output_dir / TOKENIZE_STATE_FILE,
        success_marker=final_output_dir / SUCCESS_MARKER_FILE,
        preflight=lambda: _preflight_tokenize_plan(input_shar_dirs, input_was_explicit),
        execute=lambda resume: _execute_tokenize_plan(
            pipeline_cfg=pipeline_cfg,
            final_output_dir=final_output_dir,
            fingerprint=fingerprint,
            resume=resume,
        ),
    )


def run_tokenize(spec: DatasetSpec, *, resume: bool = True) -> dict[str, Any]:
    plan = resolve_tokenize_plan(spec)
    plan.preflight()
    return plan.execute(resume)


def _resolve_input_shar_dirs(spec: DatasetSpec) -> list[str]:
    assert spec.tokenize is not None
    if spec.tokenize.input_shar_dir is not None:
        return list(spec.tokenize.input_shar_dir)
    return [spec.convert.output.shar_dir]


def _preflight_tokenize_plan(shar_dirs: list[str], input_was_explicit: bool) -> None:
    _require_input_shar_dirs_exist(shar_dirs)
    if not input_was_explicit:
        _require_prepare_success(shar_dirs)


def _require_input_shar_dirs_exist(shar_dirs: list[str]) -> None:
    for d in shar_dirs:
        if not Path(d).is_dir():
            raise FileNotFoundError(f"Tokenize input SHAR dir not found: {d!r}")


def _require_prepare_success(shar_dirs: list[str]) -> None:
    for d in shar_dirs:
        marker = Path(d) / SUCCESS_MARKER_FILE
        if not marker.is_file():
            raise RuntimeError(
                f"Tokenize input SHAR at {d!r} is missing {SUCCESS_MARKER_FILE}. "
                f"Run `stage=convert` first, or set tokenize.input_shar_dir "
                f"explicitly in YAML to consume an externally built SHAR."
            )


def _build_pipeline_cfg(spec: DatasetSpec, input_shar_dirs: list[str]) -> dict[str, Any]:
    t = spec.tokenize
    assert t is not None
    f = t.filter
    d = t.dataloader
    tok = t.tokenizer

    output_name = t.output.output_name or spec.name

    return {
        "tokenizer_path": tok.path,
        "target_sample_rate": tok.sampling_rate,
        "torch_compile": tok.torch_compile,
        "trim_last_tokens": tok.trim_last_tokens,
        "output_dir": t.output.output_dir,
        "output_name": output_name,
        "dataset_name": spec.name,
        "mode": t.mode,
        "audio_text_format": t.audio_text_format,
        "audio_text_task": t.audio_text_task,
        "resume": False,
        "shar_dir": list(input_shar_dirs),
        "shar_index_filename": t.output.shar_index_filename,
        "min_duration": f.min_duration,
        "max_duration": f.max_duration,
        "min_sample_rate": f.min_sample_rate,
        "min_rms_db": f.min_rms_db,
        "normalize_rms_db": f.normalize_rms_db,
        "max_batch_duration": d.max_batch_duration,
        "max_batch_cuts": d.max_batch_cuts,
        "num_buckets": d.num_buckets,
        "bucket_buffer_size": d.bucket_buffer_size,
        "sampler_shuffle": d.sampler_shuffle,
        "sampler_seed": d.sampler_seed,
        "quadratic_duration": d.quadratic_duration,
        "num_workers": d.num_workers,
        "prefetch_factor": d.prefetch_factor,
        "checkpoint_interval_batches": d.checkpoint_interval_batches,
        "wandb": dict(t.wandb),
    }


def _resolve_final_output_dir(pipeline_cfg: dict[str, Any]) -> Path:
    from audio_tokenization.pipelines.lhotse.core import _build_output_subdir

    subdir = _build_output_subdir(pipeline_cfg)
    return Path(pipeline_cfg["output_dir"]) / subdir


def _execute_tokenize_plan(
    *,
    pipeline_cfg: dict[str, Any],
    final_output_dir: Path,
    fingerprint: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    return run_with_resume(
        output_dir=final_output_dir,
        state_filename=TOKENIZE_STATE_FILE,
        fingerprint=fingerprint,
        guidance=(
            "Tokenize output at this path was produced under different spec "
            "values. Either re-issue the matching config to resume, or remove "
            f"{final_output_dir} and restart from scratch."
        ),
        stage_label="tokenize",
        resume=resume,
        work=lambda: _invoke_pipeline(pipeline_cfg),
        logger=logger,
    )


def _invoke_pipeline(pipeline_cfg: dict[str, Any]) -> dict[str, Any]:
    from audio_tokenization.pipelines.lhotse import run_lhotse_pipeline

    return run_lhotse_pipeline(pipeline_cfg)
