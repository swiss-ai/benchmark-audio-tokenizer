#!/usr/bin/env python3
"""VAD-aware chunking helpers for WDS -> Shar preparation.

Split long recordings into speech-aware chunks with hard caps:
- no chunk longer than ``max_chunk_sec``
- drop chunks shorter than ``min_chunk_sec``

VAD timestamps are read from per-shard JSONL files produced by
``filter_langid_vad.py``. Each worker loads only the per-shard files
that correspond to its assigned tar shards — no pre-build step needed.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class VADChunkingConfig:
    """Config for speech-aware chunking.

    max_chunk_sec: Target maximum for packing multiple segments. Single
        continuous speech segments CAN exceed this and will be kept intact.
    min_chunk_sec: Drop chunks shorter than this (too fragmented).
    sample_rate: Sample rate VAD was run at (for timestamp conversion).
    max_merge_gap_sec: Merge segments if gap is <= this threshold.
    max_duration_sec: Drop raw segments longer than this. None means
        same as max_chunk_sec.
    """

    max_chunk_sec: float = 200.0
    min_chunk_sec: float = 10.0
    sample_rate: int = 16000
    max_merge_gap_sec: float = 0.5
    max_duration_sec: float | None = None
    min_rms_db: float | None = None


def canonical_sample_key(key: str) -> str:
    """Normalize sample keys for stable matching across WDS/metadata/VAD."""
    return key.strip().lower()


def shard_name_from_tar_path(tar_path: str | Path) -> str:
    """Match the shard naming convention used by the WDS pipeline."""
    p = Path(tar_path)
    return f"{p.parent.name}/{p.stem}".lower()


def vad_per_shard_file(vad_per_shard_dir: Path, shard_name: str) -> Path:
    """Return per-shard VAD JSONL path (audio/000001 -> audio_000001.jsonl)."""
    return vad_per_shard_dir / f"{shard_name.replace('/', '_')}.jsonl"


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _normalize_raw_timestamps(raw_timestamps: Any) -> List[Tuple[int, int]]:
    """Normalize timestamps into sorted [(start, end), ...] sample-index pairs."""
    normalized: List[Tuple[int, int]] = []
    if not isinstance(raw_timestamps, list):
        return normalized

    for item in raw_timestamps:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue

        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError):
            continue

        if end_i <= start_i:
            continue
        normalized.append((start_i, end_i))

    normalized.sort(key=lambda x: x[0])
    return normalized


def _parse_vad_jsonl_line(
    line: str,
    *,
    with_duration: bool = False,
    with_sample_rate: bool = False,
    with_lang: bool = False,
) -> Optional[Tuple]:
    """Parse one line from a per-shard VAD JSONL file.

    Returns ``(normalized_key, timestamps)`` by default.  When
    *with_duration* is True, returns ``(key, timestamps, duration_sec)``
    instead and skips lines that lack a positive ``duration_sec``.
    When *with_sample_rate* is also True, appends ``sample_rate``
    (int or None) to the tuple.  When *with_lang* is True, appends
    ``lang`` (str, defaults to ``"unknown"``) as the last element.
    """
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or len(payload) != 1:
        return None

    ((raw_key, raw_value),) = payload.items()
    if not isinstance(raw_value, dict):
        return None

    key = canonical_sample_key(str(raw_key))
    timestamps = _normalize_raw_timestamps(raw_value.get("timestamps", []))

    result = (key, timestamps)

    if with_duration:
        duration_sec = raw_value.get("duration_sec", 0.0)
        try:
            duration_sec = float(duration_sec)
        except (TypeError, ValueError):
            return None
        if duration_sec <= 0:
            return None
        result = result + (duration_sec,)
        if with_sample_rate:
            sr_raw = raw_value.get("sample_rate")
            try:
                sr = int(sr_raw) if sr_raw is not None else None
            except (TypeError, ValueError):
                sr = None
            result = result + (sr,)

    if with_lang:
        result = result + (raw_value.get("lang", "unknown"),)

    return result


# ---------------------------------------------------------------------------
# Per-shard VAD loading
# ---------------------------------------------------------------------------

def load_vad_from_per_shard_dir(
    vad_per_shard_dir: Path,
    tar_paths: Sequence[str],
    *,
    with_lang: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Tuple[int, int]]] | Tuple[Dict[str, List[Tuple[int, int]]], ...]:
    """Load VAD entries from per-shard JSONL files for the given tar shards.

    Each worker calls this with its own tar_paths and gets only the
    relevant VAD entries. No pre-build step or cache needed.

    Always returns ``(vad_lookup, sr_lookup, ...)`` where ``sr_lookup``
    maps normalized sample keys to per-recording sample rates (int).
    When *with_lang* is True, also returns ``lang_lookup``.

    Returns:
        ``(vad_lookup, sr_lookup)`` or ``(vad_lookup, sr_lookup, lang_lookup)``
    """
    lookup: Dict[str, List[Tuple[int, int]]] = {}
    sr_lookup: Dict[str, int] = {}
    lang_lookup: Dict[str, str] = {}
    files_read = 0
    for tar_path in tar_paths:
        shard_name = shard_name_from_tar_path(tar_path)
        vad_file = vad_per_shard_file(vad_per_shard_dir, shard_name)
        if not vad_file.is_file():
            continue
        files_read += 1
        with open(vad_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict) or len(payload) != 1:
                    continue
                ((raw_key, raw_value),) = payload.items()
                if not isinstance(raw_value, dict):
                    continue
                key = canonical_sample_key(str(raw_key))
                timestamps = _normalize_raw_timestamps(raw_value.get("timestamps", []))
                lookup[key] = timestamps
                # Per-recording sample rate
                sr_raw = raw_value.get("sample_rate")
                if sr_raw is not None:
                    try:
                        sr_lookup[key] = int(sr_raw)
                    except (TypeError, ValueError):
                        pass
                if with_lang:
                    lang_lookup[key] = raw_value.get("lang", "unknown")
    if logger is not None:
        logger.info(
            f"Loaded {len(lookup)} VAD entries from "
            f"{files_read}/{len(tar_paths)} shard files "
            f"({len(sr_lookup)} with per-recording sample rate)"
        )
    if with_lang:
        return lookup, sr_lookup, lang_lookup
    return lookup, sr_lookup


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def _timestamps_to_seconds(
    timestamps: Sequence[Tuple[int, int]],
    *,
    sample_rate: int,
    audio_duration_sec: float,
) -> List[Tuple[float, float]]:
    spans_sec: List[Tuple[float, float]] = []
    sr = float(sample_rate)
    for start_i, end_i in timestamps:
        start = max(0.0, float(start_i) / sr)
        end = min(audio_duration_sec, float(end_i) / sr)
        if end <= start:
            continue
        spans_sec.append((start, end))
    spans_sec.sort(key=lambda x: x[0])
    return spans_sec


def _merge_spans(
    spans: Sequence[Tuple[float, float]],
    *,
    max_gap_sec: float,
) -> List[Tuple[float, float]]:
    if not spans:
        return []
    merged: List[List[float]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap_sec:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def merge_and_pack_vad(
    timestamps: Sequence[Tuple[int, int]],
    audio_duration_sec: float,
    sample_rate: int,
    *,
    max_merge_gap_sec: float,
    max_chunk_sec: float,
    min_chunk_sec: float,
    max_duration_sec: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Convert raw VAD timestamps into kept chunks as ``(offset_sec, duration_sec)``.

    Steps:
      1. Convert sample-based timestamps to seconds, clamp to [0, duration],
         and drop spans invalidated by clamping.
      2. Drop individual raw segments > *max_duration_sec* (single continuous
         speech blob too long).
      3. Merge adjacent remaining segments when gap <= *max_merge_gap_sec*.
      4. Pack into chunks up to *max_chunk_sec*.  Merged segments that exceed
         *max_chunk_sec* are emitted as standalone chunks (valid because no
         individual piece exceeds *max_duration_sec*).
      5. Drop chunks < *min_chunk_sec*.
    """
    if not timestamps:
        return []

    if max_duration_sec is None:
        max_duration_sec = max_chunk_sec

    sr = float(sample_rate)

    # Step 1 — convert to seconds, clamp, and drop invalid post-clamp spans.
    spans: List[Tuple[float, float]] = []
    for s, e in timestamps:
        if e <= s:
            continue
        start = max(0.0, s / sr)
        end = min(audio_duration_sec, e / sr)
        if end > start:
            spans.append((start, end))
    if not spans:
        return []

    # Step 2 — drop individual raw segments > max_duration_sec.
    spans = [(s, e) for s, e in spans if e - s <= max_duration_sec]
    if not spans:
        return []

    # Step 3 — merge adjacent segments when gap <= max_merge_gap_sec,
    #   but only if the result doesn't exceed max_chunk_sec.
    merged: List[Tuple[float, float]] = [spans[0]]
    for seg_start, seg_end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if (seg_start - prev_end <= max_merge_gap_sec
                and seg_end - prev_start <= max_chunk_sec):
            merged[-1] = (prev_start, seg_end)
        else:
            merged.append((seg_start, seg_end))

    # Step 4 — pack into chunks up to max_chunk_sec.
    #   Merged segments exceeding max_chunk_sec are emitted as standalone
    #   chunks (valid — no individual piece exceeds max_duration_sec).
    chunks: List[Tuple[float, float]] = []
    chunk_start, chunk_end = merged[0]
    for seg_start, seg_end in merged[1:]:
        gap = seg_start - chunk_end
        new_duration = seg_end - chunk_start
        if gap <= max_merge_gap_sec and new_duration <= max_chunk_sec:
            chunk_end = seg_end
        else:
            chunks.append((chunk_start, chunk_end - chunk_start))
            chunk_start, chunk_end = seg_start, seg_end
    chunks.append((chunk_start, chunk_end - chunk_start))

    # Step 5 — drop chunks < min_chunk_sec.
    return [(offset, dur) for offset, dur in chunks if dur >= min_chunk_sec]


def filter_vad_timestamps_by_rms(
    *,
    cut: Any,
    timestamps: Sequence[Tuple[int, int]],
    sample_rate: int,
    min_rms_db: float | None,
    runtime_counts: Counter | None = None,
) -> List[Tuple[int, int]]:
    """Drop VAD spans whose own audio RMS is below *min_rms_db*.

    This intentionally decodes span subcuts rather than slicing the full
    recording. VAD timestamps are in the VAD sample-rate domain, while cuts
    may already be resampled for SHAR output; Lhotse's truncate keeps that
    time-to-sample mapping correct without loading an entire long recording.
    """
    if min_rms_db is None:
        return list(timestamps)

    from audio_tokenization.prepare.audio_ops import (
        below_rms_threshold,
        rms_db_from_audio,
    )

    duration = float(getattr(cut, "duration", 0.0) or 0.0)
    if duration <= 0:
        return []

    threshold = float(min_rms_db)
    sr = float(sample_rate)
    kept: List[Tuple[int, int]] = []

    for start_i, end_i in timestamps:
        if end_i <= start_i:
            if runtime_counts is not None:
                runtime_counts["vad_spans_rms_invalid"] += 1
            continue

        offset = max(0.0, float(start_i) / sr)
        end_sec = min(duration, float(end_i) / sr)
        span_duration = end_sec - offset
        if span_duration <= 0:
            if runtime_counts is not None:
                runtime_counts["vad_spans_rms_invalid"] += 1
            continue

        try:
            try:
                span_cut = cut.truncate(
                    offset=offset,
                    duration=span_duration,
                    preserve_id=True,
                )
            except TypeError:
                span_cut = cut.truncate(offset=offset, duration=span_duration)
            rms_db = rms_db_from_audio(span_cut.load_audio())
        except Exception:
            if runtime_counts is not None:
                runtime_counts["vad_spans_rms_errors"] += 1
            raise

        if runtime_counts is not None:
            runtime_counts["vad_spans_rms_checked"] += 1

        if below_rms_threshold(rms_db, threshold):
            if runtime_counts is not None:
                runtime_counts["vad_spans_rms_dropped"] += 1
            continue

        if runtime_counts is not None:
            runtime_counts["vad_spans_rms_kept"] += 1
        kept.append((start_i, end_i))

    if runtime_counts is not None and timestamps and not kept:
        runtime_counts["vad_cuts_all_spans_rms_dropped"] += 1
    return kept


def split_cut_by_vad(
    *,
    cut: Any,
    sample_key: str,
    vad_lookup: Dict[str, List[Tuple[int, int]]],
    cfg: VADChunkingConfig,
    sr_lookup: Optional[Dict[str, int]] = None,
    runtime_counts: Counter | None = None,
) -> Tuple[List[Any], str]:
    """Split one recording cut with VAD-aware chunking.

    All recordings are VAD-processed uniformly (no special treatment by
    duration).  Adjacent speech segments are merged when the gap is <=
    ``max_merge_gap_sec``, then packed into chunks up to ``max_chunk_sec``.

    If *sr_lookup* is provided, the per-recording sample rate is used
    to interpret VAD timestamps. Otherwise falls back to ``cfg.sample_rate``.

    Policy:
    - duration < min_chunk_sec: drop
    - missing VAD entry: drop
    - empty VAD timestamps: drop (non-speech)
    - all timestamps invalid after clamping to cut duration: drop
    - otherwise: produce speech chunks <= max_chunk_sec and >= min_chunk_sec
    """
    duration = float(getattr(cut, "duration", 0.0) or 0.0)
    if duration < cfg.min_chunk_sec:
        return [], "too_short"

    key = canonical_sample_key(sample_key)
    timestamps = vad_lookup.get(key)

    if timestamps is None:
        return [], "missing_vad"
    if not timestamps:
        return [], "empty_vad"
    # VAD timestamps are always in the 16 kHz domain (Silero resamples
    # internally). The per-recording sample_rate stored in the JSONL is the
    # *original* recording SR, NOT the VAD domain — so always use cfg.sample_rate.
    sr = cfg.sample_rate
    has_valid_span = any(
        min(duration, e / sr) > max(0.0, s / sr)
        for s, e in timestamps
        if e > s
    )
    if not has_valid_span:
        return [], "invalid_vad_after_clamp"

    if cfg.min_rms_db is not None:
        max_duration_sec = (
            cfg.max_duration_sec
            if cfg.max_duration_sec is not None
            else cfg.max_chunk_sec
        )
        before_count = len(timestamps)
        timestamps = [
            (s, e)
            for s, e in timestamps
            if min(duration, e / sr) - max(0.0, s / sr) <= max_duration_sec
        ]
        if runtime_counts is not None:
            runtime_counts["vad_spans_rms_skipped_max_duration"] += (
                before_count - len(timestamps)
            )
        if not timestamps:
            return [], "chunks_below_min_duration"

        timestamps = filter_vad_timestamps_by_rms(
            cut=cut,
            timestamps=timestamps,
            sample_rate=sr,
            min_rms_db=cfg.min_rms_db,
            runtime_counts=runtime_counts,
        )
        if not timestamps:
            return [], "spans_below_min_rms"

    ranges = merge_and_pack_vad(
        timestamps=timestamps,
        audio_duration_sec=duration,
        sample_rate=sr,
        max_merge_gap_sec=cfg.max_merge_gap_sec,
        max_chunk_sec=cfg.max_chunk_sec,
        min_chunk_sec=cfg.min_chunk_sec,
        max_duration_sec=cfg.max_duration_sec,
    )
    if not ranges:
        return [], "chunks_below_min_duration"

    out = []
    for offset, chunk_duration in ranges:
        try:
            # preserve_id=False avoids duplicate IDs when one source cut is split.
            subcut = cut.truncate(
                offset=offset, duration=chunk_duration, preserve_id=False
            )
        except TypeError:
            # Older Lhotse versions may not expose preserve_id.
            subcut = cut.truncate(offset=offset, duration=chunk_duration)

        # Store provenance so we can trace back / re-merge later.
        subcut.custom = subcut.custom or {}
        subcut.custom["source_recording_id"] = cut.id
        subcut.custom["global_offset_sec"] = offset
        out.append(subcut)
    return out, "chunked"
