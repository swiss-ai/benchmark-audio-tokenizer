"""Stage runner primitive: skip-if-success, fail-on-partial, atomic-publish-audit.

Contract:

- ``<output_dir>/_SUCCESS`` is the only completion signal.
- ``<output_dir>/_STAGE_MANIFEST.json`` is audit-only — written at success,
  never compared. Product-specific artifacts may still own ``_MANIFEST.json``.
- A fresh run sees one of: directory absent (run), directory + ``_SUCCESS`` (skip),
  directory without ``_SUCCESS`` (partial; refuse without ``overwrite=True``).
"""

from __future__ import annotations

import datetime as _dt
import functools
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from audio_tokenization.contracts.artifacts import SUCCESS_MARKER_FILE
from audio_tokenization.utils.io import atomic_write_json, write_success_marker


MANIFEST_FILE = "_STAGE_MANIFEST.json"


def check_stage_output(
    *,
    stage: str,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any] | None:
    """Return a skip result or raise for partial output.

    This is intentionally side-effect free: callers may use it before expensive
    plan resolution, and ``run_stage`` uses the same check before running work.
    Destructive cleanup is a separate step that happens only after preflight
    succeeds.
    """

    success_marker = output_dir / SUCCESS_MARKER_FILE
    if success_marker.is_file():
        if not overwrite:
            return {
                "stage": stage,
                "skipped": True,
                "reason": f"{stage}._SUCCESS present",
                "output_dir": str(output_dir),
            }
        return None
    if output_dir.is_dir() and not overwrite:
        raise RuntimeError(
            f"Stage {stage!r} output at {output_dir} exists but is missing "
            f"{SUCCESS_MARKER_FILE} (partial or failed prior run). "
            f"Pass overwrite=True or remove the directory to rebuild."
        )
    return None


def run_stage(
    *,
    stage: str,
    output_dir: Path,
    fingerprint: Mapping[str, Any],
    work: Callable[[], dict[str, Any] | None],
    overwrite: bool,
    logger: logging.Logger,
    preflight: Callable[[], None] | None = None,
    finalize: Callable[[dict[str, Any]], None] | None = None,
    on_failure: Callable[[Exception], None] | None = None,
) -> dict[str, Any]:
    """Run *work* under the skip/partial/overwrite contract.

    Execution order on a fresh run:

    1. ``check_stage_output`` — skip on ``_SUCCESS``, raise on partial.
    2. ``preflight()`` — runs *before* any destructive cleanup, so a failure
       leaves the existing artifact intact.
    3. ``shutil.rmtree(output_dir)`` if it exists (only reached with overwrite).
    4. ``work()`` — produces the primary result dict.
    5. ``finalize(result)`` — writes terminal artifacts (e.g., aggregated
       summaries). Its failures abort the stage *before* ``_SUCCESS`` is
       written, so an incomplete output is never advertised as complete.
    6. ``_STAGE_MANIFEST.json`` (audit) and ``_SUCCESS`` are written last.

    ``on_failure(exc)`` fires on any exception from steps 1-5, before
    re-raising. Its own exceptions are logged and suppressed so the original
    error is preserved.
    """
    try:
        skipped = check_stage_output(stage=stage, output_dir=output_dir, overwrite=overwrite)
        if skipped is not None:
            logger.info("Stage %r already complete at %s; skipping.", stage, output_dir)
            return skipped

        if preflight is not None:
            preflight()

        if output_dir.is_dir():
            logger.warning(
                "Stage %r overwrite=True at %s; removing existing output.", stage, output_dir,
            )
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = _dt.datetime.now(_dt.timezone.utc)
        result = work() or {}
        if finalize is not None:
            finalize(result)
        completed_at = _dt.datetime.now(_dt.timezone.utc)

        write_stage_manifest(
            output_dir=output_dir,
            stage=stage,
            fingerprint=fingerprint,
            started_at=started_at,
            completed_at=completed_at,
        )
        write_success_marker(output_dir)
        return {**result, "stage": stage, "skipped": False, "output_dir": str(output_dir)}
    except Exception as exc:
        if on_failure is not None:
            try:
                on_failure(exc)
            except Exception:
                logger.warning("Stage %r failure hook raised.", stage, exc_info=True)
        raise


def write_stage_manifest(
    *,
    output_dir: Path,
    stage: str,
    fingerprint: Mapping[str, Any],
    started_at: _dt.datetime | None = None,
    completed_at: _dt.datetime | None = None,
) -> None:
    completed = completed_at or _dt.datetime.now(_dt.timezone.utc)
    started = started_at or completed
    atomic_write_json(
        output_dir / MANIFEST_FILE,
        {
            "version": 1,
            "stage": stage,
            "spec_fingerprint": dict(fingerprint),
            "started_at": started.isoformat(timespec="seconds"),
            "completed_at": completed.isoformat(timespec="seconds"),
            "wallclock_sec": round((completed - started).total_seconds(), 3),
            "git_sha": _current_git_sha(),
        },
    )


@functools.cache
def _current_git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None
