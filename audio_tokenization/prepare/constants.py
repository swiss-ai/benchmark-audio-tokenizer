"""Shared constants for dataset preparation helpers."""

from __future__ import annotations

from typing import Optional


SUCCESS_MARKER_FILE = "_SUCCESS"

# Minimum RMS threshold (dB) for keeping audio during SHAR conversion.
# -50dB keeps quiet but audible speech; only drops near-silence.
MIN_RMS_DB = -50.0
PREPARE_STATE_FILE = "_PREPARE_STATE.json"

# Generic stage-state schema version used by tokenize/materialize state files.
CURRENT_STAGE_STATE_VERSION = 1

# On-disk schema version for _PREPARE_STATE.json. Stale or unversioned prepare
# files are rejected; rebuild from raw inputs instead of migrating in place.
# Bumped to 2 when vad_min_rms_db was removed: V1 prepare states fingerprint
# that field, so a partial-resume from a V1 prepare directory would produce
# mixed SHAR (RMS-filtered chunks from completed workers, unfiltered from new
# ones). This version is intentionally prepare-specific; it must not invalidate
# tokenize/materialize outputs.
CURRENT_PREPARE_STATE_VERSION = 2


def state_version_for_filename(filename: str) -> int:
    """Schema version expected for a given state file name."""
    return (
        CURRENT_PREPARE_STATE_VERSION
        if filename == PREPARE_STATE_FILE
        else CURRENT_STAGE_STATE_VERSION
    )


MetadataEntry = tuple[Optional[str], dict]
_MISSING = object()

WORKER_ASSIGNMENT_FILE = "_worker_assignment.json"
WORKER_STATS_FILE = "worker_stats.json"
PREPARE_SUMMARY_FILE = "prepare_summary.json"
