"""Shared helpers for prepare_data stats scripts."""

from typing import List, Optional, Sequence, Tuple

from audio_tokenization.prepare.preprocess.chunking import (
    _parse_vad_jsonl_line,
    merge_and_pack_vad,
)


def speech_sec_in_chunks(
    timestamps: Sequence[Tuple[int, int]],
    audio_duration_sec: float,
    sample_rate: int,
    chunks: Sequence[Tuple[float, float]],
) -> float:
    """Sum speech-only seconds (excluding gaps) from VAD spans within kept chunks."""
    if not chunks or not timestamps:
        return 0.0
    sr = float(sample_rate)
    spans = [(max(0.0, s / sr), min(audio_duration_sec, e / sr))
             for s, e in timestamps if e > s]
    speech = 0.0
    si = 0
    for chunk_start, chunk_dur in chunks:
        chunk_end = chunk_start + chunk_dur
        # Advance past spans fully before this chunk
        while si < len(spans) and spans[si][1] <= chunk_start:
            si += 1
        # Sum overlapping spans (j doesn't advance si — spans may straddle chunks)
        j = si
        while j < len(spans) and spans[j][0] < chunk_end:
            speech += min(chunk_end, spans[j][1]) - max(chunk_start, spans[j][0])
            j += 1
    return speech


def read_jsonl_recordings(
    jsonl_paths: List,
    *,
    min_sr: Optional[int] = None,
    with_lang: bool = False,
) -> Tuple:
    """Read per-shard VAD JSONL files and collect recording metadata.

    Returns (recordings, skipped_min_sr) where each recording is:
      - (timestamps, duration_sec)            when with_lang=False
      - (timestamps, duration_sec, lang)      when with_lang=True
    """
    recordings = []
    skipped_min_sr = 0
    for jsonl_path in jsonl_paths:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_vad_jsonl_line(
                    line, with_duration=True, with_sample_rate=True,
                    with_lang=with_lang,
                )
                if parsed is None:
                    continue
                if with_lang:
                    _key, timestamps, duration_sec, sr, lang = parsed
                else:
                    _key, timestamps, duration_sec, sr = parsed
                if min_sr and sr is not None and sr < min_sr:
                    skipped_min_sr += 1
                    continue
                if with_lang:
                    recordings.append((timestamps, duration_sec, lang))
                else:
                    recordings.append((timestamps, duration_sec))
    return recordings, skipped_min_sr
