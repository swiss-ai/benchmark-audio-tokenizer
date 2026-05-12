"""Resolved-plan tokenize stage adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from audio_tokenization.config.schema import DatasetSpec, TokenizeSpec
from audio_tokenization.prepare.constants import SUCCESS_MARKER_FILE
from audio_tokenization.output_layout import resolve_tokenize_output_dir
from audio_tokenization.utils.io import atomic_write_json
from audio_tokenization.stages._plans import (
    ResolvedStagePlan,
    disabled_stage_plan,
)
from audio_tokenization.stages._provenance import (
    build_tokenize_resume_fingerprint,
    read_prepare_provenance,
)
from audio_tokenization.stages._resume import (
    prepare_output_for_work,
    try_skip_if_complete,
)


logger = logging.getLogger(__name__)


TOKENIZE_STATE_FILE = "tokenize_state.json"
TOKENIZE_START_FILE = ".tokenize_start.json"


def resolve_tokenize_plan(spec: DatasetSpec) -> ResolvedStagePlan:
    if spec.tokenize is None:
        return disabled_stage_plan(stage="tokenize", reason="tokenize.disabled")

    tokenize_spec = spec.tokenize
    input_shar_dirs = _resolve_input_shar_dirs(spec)
    final_output_dir = resolve_tokenize_output_dir(tokenize_spec, dataset_name=spec.name)
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
        effective=_effective_tokenize_values(spec.tokenize, final_output_dir),
        fingerprint=fingerprint,
        output_dir=final_output_dir,
        state_path=final_output_dir / TOKENIZE_STATE_FILE,
        success_marker=final_output_dir / SUCCESS_MARKER_FILE,
        preflight=lambda: _preflight_tokenize_plan(
            input_shar_dirs,
            input_was_explicit,
        ),
        execute=lambda resume: _execute_tokenize_plan(
            spec=tokenize_spec,
            dataset_name=spec.name,
            input_shar_dirs=input_shar_dirs,
            input_was_explicit=input_was_explicit,
            final_output_dir=final_output_dir,
            fingerprint=fingerprint,
            resume=resume,
        ),
    )


def _effective_tokenize_values(tokenize: TokenizeSpec, output_dir: Path) -> dict[str, Any]:
    values = {
        "mode": tokenize.mode,
        "resampling_backend": tokenize.resampling_backend,
        "effective_tokenizer_path": tokenize.tokenizer.path,
        "effective_num_workers": tokenize.dataloader.num_workers,
        "resolved_output_dir": str(output_dir),
    }
    if tokenize.mode == "audio_text":
        values["audio_text_format"] = tokenize.audio_text_format
        values["audio_text_task"] = tokenize.audio_text_task
    return values


def run_tokenize(spec: DatasetSpec, *, resume: bool = True) -> dict[str, Any]:
    if spec.tokenize is None:
        raise ValueError(
            "stage=tokenize requested but DatasetSpec has no tokenize section. "
            "Add outputs.tokenized_dir to the dataset YAML to enable this stage."
        )
    plan = resolve_tokenize_plan(spec)
    return plan.execute(resume)


def _resolve_input_shar_dirs(spec: DatasetSpec) -> list[str]:
    assert spec.tokenize is not None
    from audio_tokenization.pipelines.lhotse.data import resolve_shar_dirs

    if spec.tokenize.input_shar_dir is not None:
        raw_dirs = list(spec.tokenize.input_shar_dir)
    else:
        if spec.convert is None:
            raise ValueError(
                "tokenize.input_shar_dir is required when the dataset spec has no convert section"
            )
        raw_dirs = [spec.convert.output.shar_dir]
    return resolve_shar_dirs(
        raw_dirs,
        index_name=spec.tokenize.output.shar_index_filename,
    )


def _preflight_tokenize_plan(
    shar_dirs: list[str],
    input_was_explicit: bool,
) -> None:
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


def _execute_tokenize_plan(
    *,
    spec: TokenizeSpec,
    dataset_name: str,
    input_shar_dirs: list[str],
    input_was_explicit: bool,
    final_output_dir: Path,
    fingerprint: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    _preflight_tokenize_plan(input_shar_dirs, input_was_explicit)
    rank, world_size, local_rank = _distributed_rank_info()
    if world_size > 1:
        return _execute_tokenize_plan_distributed(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            spec=spec,
            dataset_name=dataset_name,
            input_shar_dirs=input_shar_dirs,
            final_output_dir=final_output_dir,
            fingerprint=fingerprint,
            resume=resume,
        )

    guidance = (
        "Tokenize output at this path was produced under different spec "
        "values. Either re-issue the matching config to resume, or remove "
        f"{final_output_dir} and restart from scratch."
    )
    skipped = prepare_output_for_work(
        output_dir=final_output_dir,
        state_filename=TOKENIZE_STATE_FILE,
        fingerprint=fingerprint,
        guidance=guidance,
        stage_label="tokenize",
        resume=resume,
        logger=logger,
    )
    if skipped is not None:
        return skipped
    assignment = _write_tokenize_assignment(
        spec,
        input_shar_dirs=input_shar_dirs,
        final_output_dir=final_output_dir,
        world_size=1,
    )
    result = _invoke_pipeline_for_assignment(
        spec,
        dataset_name=dataset_name,
        input_shar_dirs=input_shar_dirs,
        final_output_dir=final_output_dir,
        rank_assignment=assignment.assignment_for_rank(0),
        world_size=1,
        local_rank=local_rank,
    )
    return {**result, "skipped": False, "output_dir": str(final_output_dir)}


def _execute_tokenize_plan_distributed(
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    spec: TokenizeSpec,
    dataset_name: str,
    input_shar_dirs: list[str],
    final_output_dir: Path,
    fingerprint: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    guidance = (
        "Tokenize output at this path was produced under different spec "
        "values. Either re-issue the matching config to resume, or remove "
        f"{final_output_dir} and restart from scratch."
    )
    start_marker = final_output_dir / TOKENIZE_START_FILE

    if rank == 0:
        try:
            skipped = prepare_output_for_work(
                output_dir=final_output_dir,
                state_filename=TOKENIZE_STATE_FILE,
                fingerprint=fingerprint,
                guidance=guidance,
                stage_label="tokenize",
                resume=resume,
                logger=logger,
            )
            if skipped is not None:
                return skipped
            assignment = _write_tokenize_assignment(
                spec,
                input_shar_dirs=input_shar_dirs,
                final_output_dir=final_output_dir,
                world_size=world_size,
            )
            _write_start_marker(
                start_marker,
                payload=_build_start_marker_payload(fingerprint, world_size=world_size),
            )
        except Exception as exc:
            try:
                final_output_dir.mkdir(parents=True, exist_ok=True)
                _write_start_marker(
                    start_marker,
                    payload=_build_start_marker_payload(
                        fingerprint,
                        world_size=world_size,
                        status="aborted",
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
            except Exception:
                logger.warning("Failed to publish tokenize abort marker", exc_info=True)
            raise
    else:
        skipped = _wait_for_rank0_tokenize_start(
            final_output_dir=final_output_dir,
            state_filename=TOKENIZE_STATE_FILE,
            fingerprint=fingerprint,
            start_marker=start_marker,
            world_size=world_size,
            resume=resume,
            rank=rank,
        )
        if skipped is not None:
            return skipped
        assignment = _read_tokenize_assignment(final_output_dir)

    rank_assignment = assignment.assignment_for_rank(rank)
    result = _invoke_pipeline_for_assignment(
        spec,
        dataset_name=dataset_name,
        input_shar_dirs=input_shar_dirs,
        final_output_dir=final_output_dir,
        rank_assignment=rank_assignment,
        world_size=world_size,
        local_rank=local_rank,
    )
    # _SUCCESS publication is convoy-leader-driven: whichever rank's stats
    # write completes the rank set checks all-rank success and publishes.
    return {**result, "skipped": False, "output_dir": str(final_output_dir)}


def _write_tokenize_assignment(
    spec: TokenizeSpec,
    *,
    input_shar_dirs: list[str],
    final_output_dir: Path,
    world_size: int,
):
    """Create the launch-specific rank assignment for this tokenize run.

    The input SHAR manifest is durable and rank-independent. This assignment is
    not: it depends on the current ``world_size`` and tokenization filters, so
    it belongs under the tokenized output directory and participates in resume
    debugging rather than conversion provenance.
    """

    from audio_tokenization.pipelines.lhotse.planning import (
        TokenizeFilter,
        build_tokenize_assignment,
        load_or_build_shar_work_manifest,
        write_tokenize_plan_artifacts,
    )

    manifest, manifest_source = load_or_build_shar_work_manifest(
        input_shar_dirs,
        index_name=spec.output.shar_index_filename,
        tokenize_filter=TokenizeFilter.from_spec(spec),
        require_interleave_ids=(
            spec.mode == "audio_text"
            and spec.audio_text_format == "interleaved"
        ),
    )
    assignment = build_tokenize_assignment(manifest, world_size=world_size)
    write_tokenize_plan_artifacts(
        final_output_dir,
        manifest=manifest,
        assignment=assignment,
    )
    logger.info(
        "Tokenize assignment planned: world_size=%s active_ranks=%s "
        "work_units=%s planned_hours=%.2f manifest_source=%s",
        world_size,
        assignment.active_ranks,
        len(manifest.work_units),
        sum(u.duration_sec for u in manifest.work_units) / 3600.0,
        manifest_source,
    )
    return assignment


def _invoke_pipeline_for_assignment(
    spec: TokenizeSpec,
    *,
    dataset_name: str,
    input_shar_dirs: list[str],
    final_output_dir: Path,
    rank_assignment,
    world_size: int,
    local_rank: int,
) -> dict[str, Any]:
    if not rank_assignment.active:
        return _write_inactive_rank_stats(
            final_output_dir,
            rank=rank_assignment.rank,
            world_size=world_size,
        )

    return _invoke_pipeline(
        spec,
        dataset_name=dataset_name,
        input_shar_dirs=input_shar_dirs,
        planned_shar_fields=rank_assignment.fields,
        rank=rank_assignment.rank,
        world_size=world_size,
        local_rank=local_rank,
        final_output_dir=final_output_dir,
        assigned_cut_count=rank_assignment.cut_count,
    ) or {}


def _read_tokenize_assignment(final_output_dir: Path):
    from audio_tokenization.pipelines.lhotse.planning import (
        TOKENIZE_ASSIGNMENT_FILE,
        read_tokenize_assignment,
    )

    return read_tokenize_assignment(final_output_dir / TOKENIZE_ASSIGNMENT_FILE)


def _write_inactive_rank_stats(
    output_dir: Path,
    *,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Write a normal stats file for ranks that receive no work units.

    This keeps multi-rank orchestration simple: rank 0 can wait for one stats
    file per launched rank, while surplus ranks exit cleanly instead of
    entering Lhotse with an empty CutSet.
    """

    from audio_tokenization.pipelines.lhotse.stats_reducer import (
        maybe_publish_terminal_artifacts,
        write_rank_stats,
    )

    result: dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "inactive": True,
        "success": True,
        "samples_processed": 0,
        "tokens_generated": 0,
        "text_tokens_generated": 0,
        "errors": 0,
        "samples_skipped": 0,
        "rms_skipped": 0,
        "no_text_skipped": 0,
        "chunks_written": 0,
        "elapsed_time": 0.0,
        "output_dir": str(output_dir),
    }
    write_rank_stats(output_dir, result)
    maybe_publish_terminal_artifacts(output_dir, expected_ranks=world_size)
    return result


def _distributed_rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    if "LOCAL_RANK" in os.environ or "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    else:
        try:
            import torch

            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 0
        local_rank = rank % gpu_count if gpu_count > 0 else 0
    return rank, world_size, local_rank


def _build_start_marker_payload(
    fingerprint: dict[str, Any],
    *,
    world_size: int,
    status: str = "ready",
    error: str | None = None,
) -> dict[str, Any]:
    fingerprint_hash = _fingerprint_hash(fingerprint)
    payload = {
        "run_id": _distributed_launch_id(fingerprint_hash, world_size=world_size),
        "world_size": world_size,
        "fingerprint_hash": fingerprint_hash,
        "status": status,
    }
    if error is not None:
        payload["error"] = error
    return payload


def _write_start_marker(path: Path, *, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _distributed_launch_id(fingerprint_hash: str, *, world_size: int) -> str:
    explicit = (
        os.environ.get("AUDIO_TOKENIZATION_RUN_ID")
        or os.environ.get("TOKENIZE_RUN_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
    )
    if explicit:
        return explicit

    slurm_parts = [
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_STEP_ID"),
        os.environ.get("SLURM_RESTART_COUNT"),
    ]
    if any(part is not None for part in slurm_parts):
        return "slurm:" + ":".join(part or "none" for part in slurm_parts)

    # No common launch identifier is available outside a launcher. This still
    # keeps mismatched config/world-size stale markers from releasing ranks.
    return f"manual:{world_size}:{fingerprint_hash}"


def _read_start_marker(path: Path) -> dict[str, Any] | None:
    """Return the parsed marker, or None when it does not exist yet.

    A corrupt marker file (unparseable JSON, or valid JSON of the wrong
    shape like a list or scalar) raises so waiting ranks fail loud instead
    of polling forever. Rank 0's own writer is atomic-rename, so any
    invalid payload means an earlier process truly went sideways.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Tokenize start marker at {path} is corrupt ({exc}); "
            "delete the output directory and rerun."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Tokenize start marker at {path} has wrong shape "
            f"(got {type(payload).__name__}, expected object); "
            "delete the output directory and rerun."
        )
    return payload


def _start_marker_matches(
    payload: dict[str, Any] | None,
    *,
    fingerprint_hash: str,
    run_id: str,
    world_size: int,
    status: Literal["ready", "aborted"],
) -> bool:
    if payload is None:
        return False
    return (
        payload.get("world_size") == world_size
        and payload.get("fingerprint_hash") == fingerprint_hash
        and payload.get("run_id") == run_id
        and payload.get("status") == status
    )


def _wait_for_rank0_tokenize_start(
    *,
    final_output_dir: Path,
    state_filename: str,
    fingerprint: dict[str, Any],
    start_marker: Path,
    world_size: int,
    resume: bool,
    rank: int,
) -> dict[str, Any] | None:
    expected_hash = _fingerprint_hash(fingerprint)
    expected_run_id = _distributed_launch_id(expected_hash, world_size=world_size)
    assignment_path = final_output_dir / "_tokenize_assignment.json"
    while True:
        skipped = try_skip_if_complete(
            output_dir=final_output_dir,
            state_filename=state_filename,
            fingerprint=fingerprint,
            stage_label="tokenize",
            resume=resume,
            logger=logger,
        )
        if skipped is not None:
            return skipped
        payload = _read_start_marker(start_marker)
        if _start_marker_matches(
            payload,
            fingerprint_hash=expected_hash,
            run_id=expected_run_id,
            world_size=world_size,
            status="aborted",
        ):
            raise RuntimeError(
                "Rank 0 aborted tokenize startup before publishing a fresh "
                f"assignment marker: {payload.get('error', 'unknown error')}"
            )
        if _start_marker_matches(
            payload,
            fingerprint_hash=expected_hash,
            run_id=expected_run_id,
            world_size=world_size,
            status="ready",
        ):
            if assignment_path.is_file():
                return None
            logger.debug(
                "[rank %s] fresh start marker is visible but assignment is not "
                "published yet; continuing to wait",
                rank,
            )
        logger.debug("[rank %s] waiting for rank 0 tokenize start marker", rank)
        time.sleep(0.5)


def _invoke_pipeline(
    spec: TokenizeSpec,
    *,
    dataset_name: str,
    input_shar_dirs: list[str],
    planned_shar_fields: dict[str, list[str]],
    rank: int,
    world_size: int,
    local_rank: int,
    final_output_dir: Path,
    assigned_cut_count: int | None = None,
) -> dict[str, Any]:
    from audio_tokenization.pipelines.lhotse import run_lhotse_pipeline

    return run_lhotse_pipeline(
        spec,
        dataset_name=dataset_name,
        input_shar_dirs=input_shar_dirs,
        planned_shar_fields=planned_shar_fields,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        final_output_dir=final_output_dir,
        assigned_cut_count=assigned_cut_count,
    )
