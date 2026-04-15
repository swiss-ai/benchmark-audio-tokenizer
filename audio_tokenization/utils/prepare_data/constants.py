"""Shared constants for dataset preparation helpers."""

from __future__ import annotations

from typing import Optional


SUCCESS_MARKER_FILE = "_SUCCESS"

# Minimum RMS threshold (dB) for keeping audio during SHAR conversion.
# -50dB keeps quiet but audible speech; only drops near-silence.
MIN_RMS_DB = -50.0
PREPARE_STATE_FILE = "_PREPARE_STATE.json"
MetadataEntry = tuple[Optional[str], dict]
_MISSING = object()

WORKER_ASSIGNMENT_FILE = "_worker_assignment.json"
WORKER_STATS_FILE = "worker_stats.json"
PREPARE_SUMMARY_FILE = "prepare_summary.json"
