#!/usr/bin/env python3
"""Per-language breakdown of samples and post-VAD speech hours from VAD JSONL files.

Reads a directory of VAD JSONL files (one entry per recording, each with a
``lang`` field) and prints a table of sample counts, raw hours, and post-VAD
speech hours per language.  Each JSONL entry looks like::

    {"stem": {"timestamps": [...], "duration_sec": 20.8, "sample_rate": 16000, "lang": "en"}}

The ``--vad-dir`` can contain any flat set of ``*.jsonl`` files — splitting
by language, by shard, or by language+year all work.  More files means more
parallelism (set ``--num-workers`` accordingly).

Processing logic (applied uniformly to every recording):
  1. No valid VAD timestamps  -> skip (no speech detected)
  2. Merge adjacent segments when gap < max_merge_gap_sec
  3. Drop atomic segments exceeding max_duration_sec
  4. Pack remaining segments into chunks up to max_chunk_sec
  5. Drop chunks shorter than min_chunk_sec

Usage:
    python -m audio_tokenization.prepare.stats.lang_breakdown \
        --vad-dir /path/to/vad_results \
        --num-workers 64

    python -m audio_tokenization.prepare.stats.lang_breakdown \
        --vad-dir /path/to/vad_per_lang_year \
        --vad-min-chunk-sec 5 --vad-max-merge-gap-sec 1.0 \
        --token-rate 40 --num-workers 272
"""

import argparse
import io
import logging
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from audio_tokenization.prepare.stats._common import (
    merge_and_pack_vad,
    read_jsonl_recordings,
    speech_sec_in_chunks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Whisper language ID -> full name (99 languages)
WHISPER_LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}


def _lang_worker(args_tuple):
    """Compute per-language raw + post-VAD stats for a subset of JSONL files.

    Every recording is processed identically:
      - no valid VAD timestamps -> skipped
      - merge adjacent segments (gap < max_merge_gap_sec)
      - drop atomic segments > max_duration_sec
      - pack into chunks up to max_chunk_sec
      - drop chunks < min_chunk_sec
    """
    jsonl_paths, min_sr, min_chunk_sec, max_chunk_sec, max_merge_gap_sec, sample_rate, max_duration_sec = args_tuple

    recordings, skipped_min_sr = read_jsonl_recordings(
        jsonl_paths, min_sr=min_sr, with_lang=True,
    )

    lang_counts = Counter()          # total recordings per lang (after SR filter)
    lang_raw_sec = Counter()         # raw duration per lang
    lang_kept_count = Counter()      # recordings that produced >= 1 chunk
    lang_kept_sec = Counter()        # kept chunk duration (tokenized)
    lang_speech_sec = Counter()      # post-VAD speech seconds per lang
    lang_no_vad = Counter()          # recordings with no valid timestamps

    for timestamps, duration_sec, lang in recordings:
        lang_counts[lang] += 1
        lang_raw_sec[lang] += duration_sec

        if not timestamps:
            lang_no_vad[lang] += 1
            continue

        chunks = merge_and_pack_vad(
            timestamps, duration_sec, sample_rate,
            max_merge_gap_sec=max_merge_gap_sec,
            max_chunk_sec=max_chunk_sec,
            min_chunk_sec=min_chunk_sec,
            max_duration_sec=max_duration_sec,
        )
        if chunks:
            lang_kept_count[lang] += 1
            lang_kept_sec[lang] += sum(d for _, d in chunks)
            lang_speech_sec[lang] += speech_sec_in_chunks(
                timestamps, duration_sec, sample_rate, chunks,
            )

    return {
        "num_recordings": len(recordings),
        "skipped_min_sr": skipped_min_sr,
        "lang_counts": dict(lang_counts),
        "lang_raw_sec": dict(lang_raw_sec),
        "lang_kept_count": dict(lang_kept_count),
        "lang_kept_sec": dict(lang_kept_sec),
        "lang_speech_sec": dict(lang_speech_sec),
        "lang_no_vad": dict(lang_no_vad),
    }


def _print_lang_table(
    agg_counts: Counter,
    agg_raw_sec: Counter,
    agg_kept_count: Counter,
    agg_kept_sec: Counter,
    agg_speech_sec: Counter,
    agg_no_vad: Counter,
    total_recordings: int,
    token_rate: Optional[float] = None,
) -> None:
    """Print a formatted per-language breakdown table.

    Args:
        agg_counts: Total recordings per language (after SR filter).
        agg_raw_sec: Raw duration in seconds per language.
        agg_kept_count: Recordings that produced >= 1 VAD chunk per language.
        agg_kept_sec: Total kept chunk duration (seconds) per language.
        agg_speech_sec: Post-VAD speech seconds per language.
        agg_no_vad: Recordings with no valid VAD timestamps per language.
        total_recordings: Total recordings across all languages.
        token_rate: If provided, add an ``est_tokens`` column (tokens/sec).
    """
    total_raw_hrs = sum(agg_raw_sec.values()) / 3600.0
    total_kept_hrs = sum(agg_kept_sec.values()) / 3600.0
    total_speech_hrs = sum(agg_speech_sec.values()) / 3600.0
    total_kept = sum(agg_kept_count.values())

    # Build header with optional est_tokens column.
    header = (
        f" {'lang':<16s} | {'samples':>10s} | {'kept':>8s} | {'no_vad':>8s} | "
        f"{'raw_hrs':>10s} | {'kept_hrs':>10s} | {'speech_hrs':>10s} | "
        f"{'%samples':>8s} | {'%speech':>8s}"
    )
    if token_rate is not None:
        header += f" | {'est_tokens':>12s}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    total_est_tokens = 0
    for lang, count in agg_counts.most_common():
        raw_hrs = agg_raw_sec[lang] / 3600.0
        kept_hrs = agg_kept_sec[lang] / 3600.0
        speech_hrs = agg_speech_sec[lang] / 3600.0
        kept = agg_kept_count[lang]
        no_vad = agg_no_vad[lang]
        pct_samples = count / total_recordings * 100.0 if total_recordings else 0.0
        pct_speech = speech_hrs / total_speech_hrs * 100.0 if total_speech_hrs else 0.0
        lang_name = WHISPER_LANGUAGES.get(lang, lang)

        row = (
            f" {lang_name:<16s} | {count:>10,d} | {kept:>8,d} | {no_vad:>8,d} | "
            f"{raw_hrs:>10.1f} | {kept_hrs:>10.1f} | {speech_hrs:>10.1f} | "
            f"{pct_samples:>7.2f}% | {pct_speech:>7.2f}%"
        )
        if token_rate is not None:
            est = int(kept_hrs * 3600.0 * token_rate)
            total_est_tokens += est
            row += f" | {est:>12,d}"
        print(row)

    print(sep)

    total_row = (
        f" {'TOTAL':<16s} | {total_recordings:>10,d} | {total_kept:>8,d} | "
        f"{sum(agg_no_vad.values()):>8,d} | "
        f"{total_raw_hrs:>10.1f} | {total_kept_hrs:>10.1f} | {total_speech_hrs:>10.1f} | "
        f" 100.00% |  100.00%"
    )
    if token_rate is not None:
        total_row += f" | {total_est_tokens:>12,d}"
    print(total_row)
    print()


class _TeeWriter:
    """Write to two streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def _save_results(text: str, args: argparse.Namespace, prefix: str) -> None:
    """Save results to the parent of --vad-dir with config in the filename."""
    max_dur = args.vad_max_duration_sec or args.vad_max_chunk_sec
    name = f"{prefix}_maxdur{max_dur}.txt"
    out_path = args.vad_dir.parent / name
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"Results saved to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-language breakdown of samples and post-VAD speech hours",
    )
    parser.add_argument("--vad-dir", "--vad-per-shard-dir", type=Path, required=True,
                        dest="vad_dir",
                        help="Directory of VAD JSONL files")
    parser.add_argument("--min-sr", type=int, default=16000,
                        help="Drop audio below this sample rate (default: 16000)")
    parser.add_argument("--vad-min-chunk-sec", type=float, default=5.0,
                        help="Drop chunks shorter than this (default: 5.0)")
    parser.add_argument("--vad-max-chunk-sec", type=float, default=200.0,
                        help="Target max duration for VAD chunking (default: 200.0)")
    parser.add_argument("--vad-max-merge-gap-sec", type=float, default=1.0,
                        help="Merge adjacent VAD spans when gap < this (default: 1.0)")
    parser.add_argument("--vad-max-duration-sec", type=float, default=None,
                        help="Drop atomic speech segments longer than this "
                             "(default: same as --vad-max-chunk-sec)")
    parser.add_argument("--vad-sample-rate", type=int, default=16000,
                        help="Sample rate for VAD timestamp units (default: 16000)")
    parser.add_argument("--token-rate", type=float, default=None,
                        help="Tokens per second for estimation column (optional)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")

    args = parser.parse_args(argv)

    if not args.vad_dir.is_dir():
        raise NotADirectoryError(f"VAD directory not found: {args.vad_dir}")

    jsonl_files = sorted(args.vad_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in {args.vad_dir}")

    num_workers = min(args.num_workers, len(jsonl_files))
    worker_jsonls = [[] for _ in range(num_workers)]
    for i, jf in enumerate(jsonl_files):
        worker_jsonls[i % num_workers].append(jf)

    max_duration_sec = args.vad_max_duration_sec  # None means same as max_chunk_sec
    worker_args = [
        (jfiles, args.min_sr, args.vad_min_chunk_sec, args.vad_max_chunk_sec,
         args.vad_max_merge_gap_sec, args.vad_sample_rate, max_duration_sec)
        for jfiles in worker_jsonls
        if jfiles
    ]

    logger.info(f"Processing {len(jsonl_files)} JSONL files across {len(worker_args)} workers")
    logger.info(f"VAD config: min_chunk={args.vad_min_chunk_sec}s, "
                f"max_chunk={args.vad_max_chunk_sec}s, "
                f"merge_gap={args.vad_max_merge_gap_sec}s")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        partial_results = pool.map(_lang_worker, worker_args)
    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    # Aggregate partial results across workers.
    total_recordings = sum(p["num_recordings"] for p in partial_results)
    total_skipped = sum(p["skipped_min_sr"] for p in partial_results)
    agg_counts = Counter()
    agg_raw_sec = Counter()
    agg_kept_count = Counter()
    agg_kept_sec = Counter()
    agg_speech_sec = Counter()
    agg_no_vad = Counter()
    for p in partial_results:
        agg_counts.update(p["lang_counts"])
        agg_raw_sec.update(p["lang_raw_sec"])
        agg_kept_count.update(p["lang_kept_count"])
        agg_kept_sec.update(p["lang_kept_sec"])
        agg_speech_sec.update(p["lang_speech_sec"])
        agg_no_vad.update(p["lang_no_vad"])

    total_raw_hrs = sum(agg_raw_sec.values()) / 3600.0
    total_kept_hrs = sum(agg_kept_sec.values()) / 3600.0
    total_speech_hrs = sum(agg_speech_sec.values()) / 3600.0

    # Tee output to both stdout and a buffer for saving.
    buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = _TeeWriter(_orig_stdout, buf)

    # Summary header.
    print()
    print(f"Recordings after SR filter: {total_recordings:,}  (skipped {total_skipped:,} < {args.min_sr}Hz)")
    print(f"Total raw duration: {total_raw_hrs:,.1f} hours")
    if total_raw_hrs > 0:
        print(f"Kept (tokenized):   {total_kept_hrs:,.1f} hours  "
              f"({total_kept_hrs / total_raw_hrs * 100:.1f}% of raw)")
        print(f"Post-VAD speech:    {total_speech_hrs:,.1f} hours  "
              f"({total_speech_hrs / total_raw_hrs * 100:.1f}% of raw)")
    print(f"VAD config: min_chunk={args.vad_min_chunk_sec}s, "
          f"max_chunk={args.vad_max_chunk_sec}s, merge_gap={args.vad_max_merge_gap_sec}s")
    print()

    _print_lang_table(
        agg_counts, agg_raw_sec, agg_kept_count, agg_kept_sec,
        agg_speech_sec, agg_no_vad, total_recordings, args.token_rate,
    )

    sys.stdout = _orig_stdout
    _save_results(buf.getvalue(), args, "lang_breakdown")


if __name__ == "__main__":
    main()
