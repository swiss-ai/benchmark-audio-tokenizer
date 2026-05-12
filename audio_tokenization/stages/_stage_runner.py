"""Stage runner primitive: skip-if-success, fail-on-partial, atomic-publish-audit.

Replaces the previous fingerprint-diff resume protocol. The contract is:

- ``<output_dir>/_SUCCESS`` is the only completion signal.
- ``<output_dir>/_MANIFEST.json`` is audit-only — written at success, never compared.
- A fresh run sees one of: directory absent (run), directory + ``_SUCCESS`` (skip),
  directory without ``_SUCCESS`` (partial; refuse without ``overwrite=True``).
"""

from __future__ import annotations

import datetime as _dt
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from audio_tokenization.prepare.constants import SUCCESS_MARKER_FILE
from audio_tokenization.utils.io import atomic_streaming_write, atomic_write_json


MANIFEST_FILE = "_MANIFEST.json"


def run_stage(
    *,
    stage: str,
    output_dir: Path,
    fingerprint: Mapping[str, Any],
    work: Callable[[], dict[str, Any] | None],
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run *work* under the skip/partial/overwrite contract.

    - If ``output_dir/_SUCCESS`` is present and ``overwrite=False``: skip and
      return ``{"skipped": True, ...}``.
    - If ``output_dir`` exists without ``_SUCCESS`` and ``overwrite=False``: raise
      ``RuntimeError`` with an explicit overwrite instruction.
    - Otherwise: wipe any pre-existing directory, run *work*, write ``_SUCCESS``
      plus ``_MANIFEST.json`` (containing *fingerprint* + audit metadata), and
      return ``{"skipped": False, **work_result, ...}``.
    """
    success_marker = output_dir / SUCCESS_MARKER_FILE
    if success_marker.is_file():
        if not overwrite:
            logger.info("Stage %r already complete at %s; skipping.", stage, output_dir)
            return {
                "stage": stage,
                "skipped": True,
                "reason": f"{stage}._SUCCESS present",
                "output_dir": str(output_dir),
            }
        logger.warning(
            "Stage %r overwrite=True at %s; removing existing output.", stage, output_dir,
        )
        shutil.rmtree(output_dir)
    elif output_dir.is_dir():
        if not overwrite:
            raise RuntimeError(
                f"Stage {stage!r} output at {output_dir} exists but is missing "
                f"{SUCCESS_MARKER_FILE} (partial or failed prior run). "
                f"Pass overwrite=True or remove the directory to rebuild."
            )
        logger.warning(
            "Stage %r overwrite=True at %s; removing partial output.", stage, output_dir,
        )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _dt.datetime.now(_dt.timezone.utc)
    result = work() or {}
    completed_at = _dt.datetime.now(_dt.timezone.utc)

    atomic_write_json(
        output_dir / MANIFEST_FILE,
        {
            "version": 1,
            "stage": stage,
            "spec_fingerprint": dict(fingerprint),
            "started_at": started_at.isoformat(timespec="seconds"),
            "completed_at": completed_at.isoformat(timespec="seconds"),
            "wallclock_sec": round((completed_at - started_at).total_seconds(), 3),
            "git_sha": _current_git_sha(),
        },
    )
    with atomic_streaming_write(success_marker, mode="w") as f:
        f.write("ok\n")
    return {**result, "stage": stage, "skipped": False, "output_dir": str(output_dir)}


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
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
