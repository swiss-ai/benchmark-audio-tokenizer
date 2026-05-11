"""Resolved-plan materialize stage adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audio_tokenization.config.schema import (
    DatasetSpec,
    InterleaveProductSpec,
)
from audio_tokenization.prepare.constants import SUCCESS_MARKER_FILE
from audio_tokenization.prepare.runtime import resolve_num_workers
from audio_tokenization.stages._plans import (
    ResolvedStagePlan,
    disabled_stage_plan,
)
from audio_tokenization.stages._provenance import (
    build_interleave_resume_fingerprint,
    read_stage_provenance,
)
from audio_tokenization.stages._resume import run_with_resume
from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE


logger = logging.getLogger(__name__)


INTERLEAVE_STATE_FILE = "products_interleave_state.json"


def resolve_materialize_plan(spec: DatasetSpec) -> ResolvedStagePlan:
    interleave = spec.materialize.interleave
    if not interleave.enabled:
        return disabled_stage_plan(stage="materialize", reason="interleave.disabled")

    parquet_dir = _resolve_parquet_dir(spec, interleave)
    tokenizer_path = _resolve_tokenizer_path(spec, interleave)
    output_dir = Path(interleave.output_dir) if interleave.output_dir is not None else None
    if output_dir is None:
        raise ValueError("materialize.interleave.enabled=true requires an explicit output_dir.")

    fingerprint = build_interleave_resume_fingerprint(
        interleave,
        cache_dir=parquet_dir,
        tokenizer_path=tokenizer_path,
        tokenize_provenance=read_stage_provenance(
            [parquet_dir], state_filename=TOKENIZE_STATE_FILE
        ),
    )

    return ResolvedStagePlan(
        stage="materialize",
        enabled=True,
        reason=None,
        inputs={
            "resolved_cache_dir": str(parquet_dir),
            "resolved_tokenizer_path": tokenizer_path,
            "cache_was_explicit": interleave.cache_dir is not None,
        },
        outputs={
            "output_dir": str(output_dir),
            "state_file": str(output_dir / INTERLEAVE_STATE_FILE),
            "success_marker": str(output_dir / SUCCESS_MARKER_FILE),
        },
        effective={
            "strategy": interleave.strategy,
            "max_seq_len": interleave.max_seq_len,
            "max_gap_sec": interleave.max_gap_sec,
            "num_workers": interleave.num_workers,
        },
        fingerprint=fingerprint,
        output_dir=output_dir,
        state_path=output_dir / INTERLEAVE_STATE_FILE,
        success_marker=output_dir / SUCCESS_MARKER_FILE,
        preflight=lambda: _preflight_materialize_plan(spec, interleave, parquet_dir),
        execute=lambda resume: _execute_materialize_plan(
            spec=spec,
            interleave=interleave,
            parquet_dir=parquet_dir,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            fingerprint=fingerprint,
            resume=resume,
        ),
    )


def run_materialize(spec: DatasetSpec, *, resume: bool = True) -> dict[str, Any]:
    if not spec.materialize.interleave.enabled:
        raise ValueError(
            "stage=materialize requested but DatasetSpec has no materialize section. "
            "Add materialization.interleave.enabled=true with output_dir to enable this stage."
        )
    interleave_plan = resolve_materialize_plan(spec)
    return interleave_plan.execute(resume)


def _preflight_materialize_plan(
    spec: DatasetSpec, interleave: InterleaveProductSpec, parquet_dir: Path
) -> None:
    if interleave.strategy != "shift_by_one":
        raise NotImplementedError(
            f"materialize.interleave.strategy={interleave.strategy!r} is not wired "
            "through the unified stage graph yet. Use the direct CLI: "
            "`python -m audio_tokenization.interleave.pattern ...` or "
            "`python -m audio_tokenization.interleave.greedy ...`."
        )
    if interleave.cache_dir is None:
        _require_tokenize_success(parquet_dir)
    elif not parquet_dir.is_dir():
        raise FileNotFoundError(
            f"Explicit interleave cache_dir not found: {str(parquet_dir)!r}."
        )
    if interleave.output_dir is None:
        raise ValueError("materialize.interleave.enabled=true requires an explicit output_dir.")


def _execute_materialize_plan(
    *,
    spec: DatasetSpec,
    interleave: InterleaveProductSpec,
    parquet_dir: Path,
    tokenizer_path: str,
    output_dir: Path,
    fingerprint: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    _preflight_materialize_plan(spec, interleave, parquet_dir)
    argv = _build_interleave_argv(interleave, parquet_dir, tokenizer_path)
    return {
        "interleave": run_with_resume(
            output_dir=output_dir,
            state_filename=INTERLEAVE_STATE_FILE,
            fingerprint=fingerprint,
            guidance=(
                "Interleave product at this path was produced under different spec "
                "values. Either re-issue the matching config to resume, or remove "
                f"{output_dir} and restart from scratch."
            ),
            stage_label="interleave",
            resume=resume,
            work=lambda: _invoke_shift_by_one(argv),
            logger=logger,
        )
    }


def _resolve_parquet_dir(spec: DatasetSpec, interleave: InterleaveProductSpec) -> Path:
    if interleave.cache_dir is not None:
        return Path(interleave.cache_dir)

    assert spec.tokenize is not None  # gated by DatasetSpec cross-section validator
    from audio_tokenization.output_layout import resolve_tokenize_output_dir

    return resolve_tokenize_output_dir(spec.tokenize, dataset_name=spec.name)


def _resolve_tokenizer_path(spec: DatasetSpec, interleave: InterleaveProductSpec) -> str:
    if interleave.tokenizer_path is not None:
        return interleave.tokenizer_path
    if spec.tokenize is not None:
        return spec.tokenize.tokenizer.path
    raise ValueError(
        "materialize.interleave requires a tokenizer_path — set it explicitly on "
        "the interleave product, or add a tokenize section whose "
        "tokenizer.path can be inherited."
    )


def _require_tokenize_success(cache_dir: Path) -> None:
    marker = cache_dir / SUCCESS_MARKER_FILE
    if not marker.is_file():
        raise RuntimeError(
            f"Derived interleave cache at {str(cache_dir)!r} is missing {SUCCESS_MARKER_FILE}. "
            "Run `stage=tokenize` first, or set materialize.interleave.cache_dir "
            "explicitly to consume an externally built cache."
        )
    state_path = cache_dir / TOKENIZE_STATE_FILE
    if not state_path.is_file():
        raise RuntimeError(
            f"Derived interleave cache at {str(cache_dir)!r} is missing {TOKENIZE_STATE_FILE}. "
            "Tokenize must complete successfully before materialize can derive its "
            "input cache from this pipeline."
        )


def _build_interleave_argv(
    interleave: InterleaveProductSpec,
    parquet_dir: Path,
    tokenizer_path: str,
) -> list[str]:
    # Resolve before argv-building so shift_by_one receives a SLURM-aware
    # concrete int, not a passthrough that re-falls-back to its own
    # cpu_count - 2 default.
    num_workers = resolve_num_workers(interleave.num_workers)
    argv = [
        "--parquet-dir", str(parquet_dir),
        "--output-dir", str(interleave.output_dir),
        "--tokenizer-path", tokenizer_path,
        "--max-seq-len", str(interleave.max_seq_len),
        "--num-workers", str(num_workers),
    ]
    if interleave.max_gap_sec is not None:
        argv += ["--max-gap-sec", str(interleave.max_gap_sec)]
    if interleave.seq_threshold is not None:
        argv += ["--seq-threshold", str(interleave.seq_threshold)]
    if interleave.transcribe_ratio is not None:
        argv += ["--transcribe-ratio", str(interleave.transcribe_ratio)]
    if interleave.tmp_dir is not None:
        argv += ["--tmp-dir", interleave.tmp_dir]
    return argv


def _invoke_shift_by_one(argv: list[str]) -> None:
    from audio_tokenization.interleave.shift_by_one import main as shift_by_one_main

    shift_by_one_main(argv)
