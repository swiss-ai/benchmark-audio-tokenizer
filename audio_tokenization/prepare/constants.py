"""Shared constants for dataset preparation helpers."""

from __future__ import annotations

from typing import Optional


# Minimum RMS threshold (dB) for keeping audio during SHAR conversion.
# -50dB keeps quiet but audible speech; only drops near-silence.
MIN_RMS_DB = -50.0


MetadataEntry = tuple[Optional[str], dict]
_MISSING = object()

WORKER_STATS_FILE = "worker_stats.json"
PREPARE_SUMMARY_FILE = "prepare_summary.json"
PREPARE_SHAR_COMMIT_MODE = "atomic"
