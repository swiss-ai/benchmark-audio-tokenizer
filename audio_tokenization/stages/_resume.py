"""Shared dual-marker resume protocol for stage adapters.

The ``tokenize`` and ``materialize`` adapters write ``_SUCCESS`` plus a
versioned state-fingerprint JSON via ``run_with_resume`` here. Convert
routes through ``audio_tokenization/prepare/runtime.py`` instead — its
state lives next to per-family worker_XX dirs and predates this helper.
Re-runs short-circuit iff marker AND state fingerprint match; drift
raises with field-level diffs.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Literal, Mapping


# Free-form labels caused two cycles of "is this a stage or a product?" already
# (materialize calls itself "interleave" when invoking run_with_resume). Make
# the allowed set explicit so adding a new stage forces a deliberate edit.
StageLabel = Literal["convert", "tokenize", "interleave"]

from audio_tokenization.prepare.constants import SUCCESS_MARKER_FILE
from audio_tokenization.prepare.runtime import (
    diff_fingerprint,
    mark_partition_success,
    read_prepare_state,
    validate_or_write_prepare_state,
)


def try_skip_if_complete(
    *,
    output_dir: Path,
    state_filename: str,
    fingerprint: Mapping[str, Any],
    stage_label: StageLabel,
    resume: bool,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    """Return a ``{"skipped": True, ...}`` payload iff the stage can be
    safely skipped, else ``None``.

    Skip conditions: ``resume`` is true, ``_SUCCESS`` exists, state file
    exists, and the on-disk fingerprint matches *fingerprint*. Any
    divergence returns None so callers can decide how to rerun — drift is
    the caller's responsibility to surface (via the full run_with_resume
    protocol below, or by invoking the stage runner whose internal
    state-writer will raise).
    """
    if not resume:
        return None
    success_marker = output_dir / SUCCESS_MARKER_FILE
    state_path = output_dir / state_filename
    if not (success_marker.is_file() and state_path.is_file()):
        return None
    on_disk = read_prepare_state(state_path)
    if diff_fingerprint(fingerprint, on_disk):
        return None
    logger.info(
        "Stage %r already complete at %s; state fingerprint matches, skipping.",
        stage_label, output_dir,
    )
    return {
        "skipped": True,
        "reason": f"{stage_label}._SUCCESS and state fingerprint match",
        "output_dir": str(output_dir),
    }


def run_with_resume(
    *,
    output_dir: Path,
    state_filename: str,
    fingerprint: dict[str, Any],
    guidance: str,
    stage_label: StageLabel,
    resume: bool,
    work: Callable[[], dict[str, Any] | None],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Execute *work* under the dual-marker protocol.

    Returns ``{"skipped": True, ...}`` when both marker and state exist
    and the fingerprint matches; otherwise validates/writes state, runs
    *work*, marks success, and returns ``{"skipped": False, **work_result, ...}``.
    Raises ``AssertionError`` (via ``validate_or_write_prepare_state``) on
    config drift.
    """
    state_path = output_dir / state_filename

    skipped = try_skip_if_complete(
        output_dir=output_dir,
        state_filename=state_filename,
        fingerprint=fingerprint,
        stage_label=stage_label,
        resume=resume,
        logger=logger,
    )
    if skipped is not None:
        return skipped
    # Fall through: any pre-existing state must still pass drift check
    # (validate_or_write_prepare_state raises) before we wipe and rerun.
    if state_path.is_file():
        validate_or_write_prepare_state(
            state_path,
            expected=fingerprint,
            invariant_keys=tuple(fingerprint.keys()),
            guidance=guidance,
        )

    if output_dir.is_dir():
        logger.warning(
            "Stage %r restarting at %s; removing pre-existing partial output.",
            stage_label,
            output_dir,
        )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    validate_or_write_prepare_state(
        state_path,
        expected=fingerprint,
        invariant_keys=tuple(fingerprint.keys()),
        guidance=guidance,
    )

    result = work() or {}
    mark_partition_success(output_dir)
    return {**result, "skipped": False, "output_dir": str(output_dir)}
