"""Resolved-plan materialize stage adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audio_tokenization.config.schema import (
    DatasetSpec,
    InterleaveProductSpec,
    SftProductSpec,
)
from audio_tokenization.contracts.artifacts import SUCCESS_MARKER_FILE
from audio_tokenization.prepare.runtime import resolve_num_workers
from audio_tokenization.stages._plans import (
    ResolvedStagePlan,
    disabled_stage_plan,
)
from audio_tokenization.stages._stage_runner import MANIFEST_FILE, run_stage


logger = logging.getLogger(__name__)


def resolve_materialize_plan(spec: DatasetSpec) -> ResolvedStagePlan:
    interleave = spec.materialize.interleave
    sft = spec.materialize.sft
    if interleave.enabled:
        return _resolve_interleave_materialize_plan(spec)
    if sft.enabled:
        return _resolve_sft_materialize_plan(spec)
    return disabled_stage_plan(stage="materialize")


def run_materialize(spec: DatasetSpec, *, overwrite: bool = False) -> dict[str, Any]:
    interleave = spec.materialize.interleave
    sft = spec.materialize.sft
    if interleave.enabled:
        return _resolve_interleave_materialize_plan(spec).execute(overwrite)
    if sft.enabled:
        return _resolve_sft_materialize_plan(spec).execute(overwrite)
    raise ValueError(
        "stage=materialize requested but DatasetSpec has no materialize section. "
        "Add a materialization.interleave or materialization.sft product section "
        "with output_dir to enable this stage."
    )


def _resolve_interleave_materialize_plan(spec: DatasetSpec) -> ResolvedStagePlan:
    interleave = spec.materialize.interleave
    parquet_dir = _resolve_product_cache_dir(spec, interleave)
    tokenizer_path = _resolve_tokenizer_path(spec, interleave)
    output_dir = Path(interleave.output_dir) if interleave.output_dir is not None else None
    if output_dir is None:
        raise ValueError("materialize.interleave.enabled=true requires an explicit output_dir.")

    fingerprint = {
        **interleave.fingerprint_payload(),
        "resolved_cache_dir": str(parquet_dir),
        "resolved_tokenizer_path": tokenizer_path,
    }

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
            "manifest": str(output_dir / MANIFEST_FILE),
            "success_marker": str(output_dir / SUCCESS_MARKER_FILE),
        },
        effective={
            "product": "interleave",
            "strategy": interleave.strategy,
            "max_seq_len": interleave.max_seq_len,
            "max_gap_sec": interleave.max_gap_sec,
            "num_workers": interleave.num_workers,
        },
        fingerprint=fingerprint,
        output_dir=output_dir,
        success_marker=output_dir / SUCCESS_MARKER_FILE,
        preflight=lambda: _preflight_materialize_plan(interleave, parquet_dir),
        execute=lambda overwrite: _execute_materialize_plan(
            interleave=interleave,
            parquet_dir=parquet_dir,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            fingerprint=fingerprint,
            overwrite=overwrite,
        ),
    )


def _preflight_materialize_plan(
    interleave: InterleaveProductSpec, parquet_dir: Path
) -> None:
    if interleave.strategy != "shift_by_one":
        raise NotImplementedError(
            f"materialize.interleave.strategy={interleave.strategy!r} is not supported. "
            "Only 'shift_by_one' is wired through the stage graph. Legacy strategies "
            "(greedy, pattern) were removed; recover the modules from git history "
            "(`git log --diff-filter=D -- audio_tokenization/interleave/`) for ablation use."
        )
    if interleave.cache_dir is None:
        _require_tokenize_success(parquet_dir, product="interleave")
    elif not parquet_dir.is_dir():
        raise FileNotFoundError(
            f"Explicit interleave cache_dir not found: {str(parquet_dir)!r}."
        )
    if interleave.output_dir is None:
        raise ValueError("materialize.interleave.enabled=true requires an explicit output_dir.")


def _execute_materialize_plan(
    *,
    interleave: InterleaveProductSpec,
    parquet_dir: Path,
    tokenizer_path: str,
    output_dir: Path,
    fingerprint: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    argv = _build_interleave_argv(interleave, parquet_dir, tokenizer_path)
    return {
        "interleave": run_stage(
            stage="materialize.interleave",
            output_dir=output_dir,
            fingerprint=fingerprint,
            work=lambda: _invoke_shift_by_one(argv),
            overwrite=overwrite,
            logger=logger,
            preflight=lambda: _preflight_materialize_plan(interleave, parquet_dir),
        )
    }


def _resolve_sft_materialize_plan(spec: DatasetSpec) -> ResolvedStagePlan:
    sft = spec.materialize.sft
    conversations_dir = _resolve_sft_conversations_dir(sft)
    cache_dir = _resolve_product_cache_dir(spec, sft)
    tokenizer_path = _resolve_sft_tokenizer_path(sft)
    output_dir = Path(sft.output_dir) if sft.output_dir is not None else None
    if output_dir is None:
        raise ValueError("materialize.sft.enabled=true requires output_dir.")

    fingerprint = {
        **sft.fingerprint_payload(),
        "resolved_conversations_dir": str(conversations_dir),
        "resolved_cache_dir": str(cache_dir),
        "resolved_tokenizer_path": tokenizer_path,
    }

    return ResolvedStagePlan(
        stage="materialize",
        enabled=True,
        reason=None,
        inputs={
            "resolved_conversations_dir": str(conversations_dir),
            "resolved_cache_dir": str(cache_dir),
            "resolved_tokenizer_path": tokenizer_path,
            "cache_was_explicit": sft.cache_dir is not None,
        },
        outputs={
            "output_dir": str(output_dir),
            "manifest": str(output_dir / MANIFEST_FILE),
            "success_marker": str(output_dir / SUCCESS_MARKER_FILE),
        },
        effective={
            "product": "sft",
            "max_seq_len": sft.max_seq_len,
            "seq_threshold": sft.seq_threshold,
            "audio_placeholder": sft.audio_placeholder,
            "conversations_glob": sft.conversations_glob,
            "messages_column": sft.messages_column,
            "audio_ids_column": sft.audio_ids_column,
            "num_workers": sft.num_workers,
        },
        fingerprint=fingerprint,
        output_dir=output_dir,
        success_marker=output_dir / SUCCESS_MARKER_FILE,
        preflight=lambda: _preflight_sft_materialize_plan(
            sft, conversations_dir, cache_dir
        ),
        execute=lambda overwrite: _execute_sft_materialize_plan(
            sft=sft,
            conversations_dir=conversations_dir,
            cache_dir=cache_dir,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            fingerprint=fingerprint,
            overwrite=overwrite,
        ),
    )


def _preflight_sft_materialize_plan(
    sft: SftProductSpec,
    conversations_dir: Path,
    cache_dir: Path,
) -> None:
    """Validate SFT materialize inputs before output is created."""
    if not conversations_dir.is_dir():
        raise FileNotFoundError(f"SFT conversations_dir not found: {str(conversations_dir)!r}.")
    if sft.cache_dir is None:
        _require_tokenize_success(cache_dir, product="sft")
    elif not cache_dir.is_dir():
        raise FileNotFoundError(f"SFT audio cache_dir not found: {str(cache_dir)!r}.")
    if sft.output_dir is None:
        raise ValueError("materialize.sft requires output_dir.")
    if sft.tokenizer_path is None:
        raise ValueError("materialize.sft requires tokenizer_path.")
    from audio_tokenization.sft.preflight import validate_sft_package

    validate_sft_package(
        conversations_dir=conversations_dir,
        conversations_glob=sft.conversations_glob,
        messages_column=sft.messages_column,
        audio_ids_column=sft.audio_ids_column,
        audio_placeholder=sft.audio_placeholder,
    )


def _execute_sft_materialize_plan(
    *,
    sft: SftProductSpec,
    conversations_dir: Path,
    cache_dir: Path,
    tokenizer_path: str,
    output_dir: Path,
    fingerprint: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    def _work() -> dict[str, Any]:
        from audio_tokenization.sft.materialize import (
            SftMaterializeConfig,
            materialize_sft,
        )

        return materialize_sft(
            SftMaterializeConfig(
                conversations_dir=conversations_dir,
                cache_dir=cache_dir,
                output_dir=output_dir,
                tokenizer_path=tokenizer_path,
                max_seq_len=sft.max_seq_len,
                seq_threshold=sft.seq_threshold,
                audio_placeholder=sft.audio_placeholder,
                conversations_glob=sft.conversations_glob,
                messages_column=sft.messages_column,
                audio_ids_column=sft.audio_ids_column,
                num_workers=resolve_num_workers(sft.num_workers),
            )
        )

    return {
        "sft": run_stage(
            stage="materialize.sft",
            output_dir=output_dir,
            fingerprint=fingerprint,
            work=_work,
            overwrite=overwrite,
            logger=logger,
            preflight=lambda: _preflight_sft_materialize_plan(
                sft, conversations_dir, cache_dir
            ),
        )
    }


def _resolve_product_cache_dir(
    spec: DatasetSpec, product: InterleaveProductSpec | SftProductSpec
) -> Path:
    """Resolve a materialize product's cache dir: explicit, or derived from tokenize."""
    if product.cache_dir is not None:
        return Path(product.cache_dir)

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


def _resolve_sft_conversations_dir(sft: SftProductSpec) -> Path:
    if sft.conversations_dir is None:
        raise ValueError("materialize.sft requires conversations_dir.")
    return Path(sft.conversations_dir)


def _resolve_sft_tokenizer_path(sft: SftProductSpec) -> str:
    if sft.tokenizer_path is None:
        raise ValueError("materialize.sft requires tokenizer_path.")
    return sft.tokenizer_path


def _require_tokenize_success(cache_dir: Path, *, product: str) -> None:
    marker = cache_dir / SUCCESS_MARKER_FILE
    if not marker.is_file():
        raise RuntimeError(
            f"Derived {product} cache at {str(cache_dir)!r} is missing {SUCCESS_MARKER_FILE}. "
            f"Run `stage=tokenize` first, or set materialize.{product}.cache_dir "
            "explicitly to consume an externally built cache."
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
