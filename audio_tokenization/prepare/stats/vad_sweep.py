#!/usr/bin/env python3
"""Dry-run VAD chunking stats — sweep merge-gap and min-chunk parameters.

Reads a directory of VAD JSONL files and computes chunk statistics for a
grid of (min_chunk_sec, merge_gap_sec) values without touching any audio
or writing Shar output.  After the sweep tables, prints a per-language
breakdown for the current (default) parameter combo.

The ``--vad-dir`` can contain any flat set of ``*.jsonl`` files — splitting
by language, by shard, or by language+year all work.  More files means more
parallelism (set ``--num-workers`` accordingly).

Processing logic (applied uniformly to every recording):
  1. No valid VAD timestamps  -> skip
  2. Merge adjacent segments when gap < merge_gap
  3. Drop atomic segments exceeding max_duration_sec
  4. Pack remaining segments into chunks up to max_chunk_sec
  5. Drop chunks shorter than min_chunk_sec

Usage:
    python -m audio_tokenization.prepare.stats.vad_sweep \
        --vad-dir /path/to/vad_results \
        --num-workers 32

    python -m audio_tokenization.prepare.stats.vad_sweep \
        --vad-dir /path/to/vad_per_lang_year \
        --min-chunk-sweep 1,5,10,20,30 \
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

from audio_tokenization.prepare.stats._common import (
    merge_and_pack_vad,
    read_jsonl_recordings,
    speech_sec_in_chunks,
)
from audio_tokenization.prepare.stats.lang_breakdown import (
    _TeeWriter,
    _lang_worker,
    _print_lang_table,
    _save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _dry_run_worker(args_tuple):
    """Compute VAD chunk stats for a subset of JSONL files across all (min_chunk, merge_gap) pairs.

    Every recording is processed identically — no tier-based shortcuts.
    The sweep is 2D: (min_chunk_sec, merge_gap).
    """
    jsonl_paths, merge_gaps, max_chunk_sec, min_chunk_secs, sample_rate, min_sr, max_duration_sec = args_tuple

    recordings, skipped_min_sr = read_jsonl_recordings(jsonl_paths, min_sr=min_sr)

    results_by_min_chunk = {}
    for min_chunk_sec in min_chunk_secs:
        gap_results = {}
        for gap in merge_gaps:
            chunked_sec = 0.0
            speech_sec = 0.0
            num_chunks = 0
            no_vad = 0
            no_chunks = 0
            chunks_over_max = 0
            kept_recordings = 0
            raw_sec_total = 0.0

            for timestamps, duration_sec in recordings:
                raw_sec_total += duration_sec

                if not timestamps:
                    no_vad += 1
                    continue

                chunks = merge_and_pack_vad(
                    timestamps, duration_sec, sample_rate,
                    max_merge_gap_sec=gap,
                    max_chunk_sec=max_chunk_sec,
                    min_chunk_sec=min_chunk_sec,
                    max_duration_sec=max_duration_sec,
                )
                if not chunks:
                    no_chunks += 1
                else:
                    kept_recordings += 1

                speech_sec += speech_sec_in_chunks(
                    timestamps, duration_sec, sample_rate, chunks,
                )
                for _offset, chunk_dur in chunks:
                    chunked_sec += chunk_dur
                    num_chunks += 1
                    if chunk_dur > max_chunk_sec:
                        chunks_over_max += 1

            gap_results[gap] = {
                "chunked_sec": chunked_sec,
                "speech_sec": speech_sec,
                "num_chunks": num_chunks,
                "no_vad": no_vad,
                "no_chunks": no_chunks,
                "chunks_over_max": chunks_over_max,
                "kept_recordings": kept_recordings,
                "raw_sec_total": raw_sec_total,
            }

        results_by_min_chunk[min_chunk_sec] = {"gap_results": gap_results}

    total_duration_sec = sum(dur for _, dur in recordings)
    return {
        "num_recordings": len(recordings),
        "skipped_min_sr": skipped_min_sr,
        "total_duration_sec": total_duration_sec,
        "results_by_min_chunk": results_by_min_chunk,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dry-run: sweep VAD chunking parameters and print stats",
    )
    parser.add_argument("--vad-dir", "--vad-per-shard-dir", type=Path, required=True,
                        dest="vad_dir",
                        help="Directory of VAD JSONL files")
    parser.add_argument("--vad-max-chunk-sec", type=float, default=200.0,
                        help="Target max duration while packing VAD segments")
    parser.add_argument("--vad-min-chunk-sec", type=float, default=10.0,
                        help="Drop chunks shorter than this duration")
    parser.add_argument("--vad-sample-rate", type=int, default=16000,
                        help="Sample rate used to decode VAD timestamp units")
    parser.add_argument("--vad-max-merge-gap-sec", type=float, default=0.5,
                        help="Merge adjacent VAD spans when silence gap < this threshold")
    parser.add_argument("--vad-max-duration-sec", type=float, default=None,
                        help="Drop atomic speech segments longer than this "
                             "(default: same as --vad-max-chunk-sec)")
    parser.add_argument("--min-sr", type=int, default=16000,
                        help="Drop audio below this sample rate (default: 16000)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--token-rate", type=float, default=None,
                        help="Tokens per second for estimation column (optional)")
    parser.add_argument("--min-chunk-sweep", type=str, default="1,5,10,20,30",
                        help="Comma-separated min_chunk_sec values to sweep "
                             "(default: '1,5,10,20,30')")
    parser.add_argument("--max-merge-gap-sweep", type=str, default="0,0.25,0.5,1,2,5",
                        help="Comma-separated max_merge_gap_sec values to sweep "
                             "(default: '0,0.25,0.5,1,2,5')")

    args = parser.parse_args(argv)

    if not args.vad_dir.is_dir():
        raise NotADirectoryError(f"VAD directory not found: {args.vad_dir}")

    sweep_gaps = sorted(set(float(v.strip()) for v in args.max_merge_gap_sweep.split(",")))
    sweep_min_chunks = sorted(set(float(v.strip()) for v in args.min_chunk_sweep.split(",")))
    logger.info(f"Dry-run: sweeping merge_gap_sec = {sweep_gaps}")
    logger.info(f"Dry-run: sweeping min_chunk_sec = {sweep_min_chunks}")

    # Glob JSONL files and distribute round-robin across workers.
    jsonl_files = sorted(args.vad_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files in {args.vad_dir}")

    num_workers = min(args.num_workers, len(jsonl_files))
    worker_jsonls = [[] for _ in range(num_workers)]
    for i, jf in enumerate(jsonl_files):
        worker_jsonls[i % num_workers].append(jf)

    max_duration_sec = args.vad_max_duration_sec  # None means same as max_chunk_sec
    dry_run_args = [
        (jfiles, sweep_gaps, args.vad_max_chunk_sec, sweep_min_chunks,
         args.vad_sample_rate, args.min_sr, max_duration_sec)
        for jfiles in worker_jsonls
        if jfiles
    ]

    logger.info(f"Dry-run: processing {len(jsonl_files)} JSONL files across {len(dry_run_args)} workers")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(dry_run_args)) as pool:
        partial_results = pool.map(_dry_run_worker, dry_run_args)
    elapsed = time.time() - t0
    logger.info(f"Dry-run: parallel computation done in {elapsed:.1f}s")

    # SR-filter stats (invariant across min_chunk).
    total_in_jsonl = sum(p["num_recordings"] + p["skipped_min_sr"] for p in partial_results)
    total_skipped_min_sr = sum(p["skipped_min_sr"] for p in partial_results)
    total_recordings = sum(p["num_recordings"] for p in partial_results)
    total_duration_hrs = sum(p["total_duration_sec"] for p in partial_results) / 3600.0

    # Tee output to both stdout and a buffer for saving.
    buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = _TeeWriter(_orig_stdout, buf)

    print()
    print(f"Recordings in JSONL:    {total_in_jsonl:>14,d}")
    print(f"  skipped_min_sr (<{args.min_sr}Hz): {total_skipped_min_sr:>10,d}   (dropped)")
    print(f"After SR filter:        {total_recordings:>14,d}   ({total_duration_hrs:,.1f} hours total)")

    # Print one sweep table per min_chunk_sec.
    for min_s in sweep_min_chunks:
        gap_stats = {}
        for gap in sweep_gaps:
            gap_stats[gap] = {
                k: sum(
                    p["results_by_min_chunk"][min_s]["gap_results"][gap][k]
                    for p in partial_results
                )
                for k in (
                    "chunked_sec", "speech_sec", "num_chunks",
                    "no_vad", "no_chunks", "chunks_over_max",
                    "kept_recordings", "raw_sec_total",
                )
            }

        label = f"min_chunk_sec = {min_s}"
        print()
        print(f"--- {label} ---")

        header = (
            f" {'merge_gap':>9s} | {'kept_hrs':>12s} | {'speech_hrs':>12s} | "
            f"{'%kept':>6s} | {'raw_hrs':>12s} | "
            f"{'chunks':>14s} | {'avg_sec':>8s} | {'no_vad':>8s} | "
            f"{'no_chk':>8s} | {'>max':>10s}"
        )
        if args.token_rate is not None:
            header += f" | {'est_tokens':>16s}"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for gap in sweep_gaps:
            gs = gap_stats[gap]
            kept_hrs = gs["chunked_sec"] / 3600.0
            speech_hrs = gs["speech_sec"] / 3600.0
            raw_hrs = gs["raw_sec_total"] / 3600.0
            pct_kept = (speech_hrs / total_duration_hrs * 100.0) if total_duration_hrs > 0 else 0.0
            n_chunks = gs["num_chunks"]
            avg = (gs["chunked_sec"] / n_chunks) if n_chunks else 0.0
            row = (
                f" {gap:>9.2f} | {kept_hrs:>12,.1f} | {speech_hrs:>12,.1f} | "
                f"{pct_kept:>5.1f}% | {raw_hrs:>12,.1f} | "
                f"{n_chunks:>14,d} | {avg:>8.1f} | {gs['no_vad']:>8,d} | "
                f"{gs['no_chunks']:>8,d} | {gs['chunks_over_max']:>10,d}"
            )
            if args.token_rate is not None:
                est_tokens = int(kept_hrs * 3600.0 * args.token_rate)
                row += f" | {est_tokens:>16,d}"
            print(row)
        print(sep)

    # Per-language breakdown for the current (default) parameters.
    logger.info("Computing per-language breakdown for current parameters...")
    lang_worker_args = [
        (jfiles, args.min_sr, args.vad_min_chunk_sec, args.vad_max_chunk_sec,
         args.vad_max_merge_gap_sec, args.vad_sample_rate, max_duration_sec)
        for jfiles in worker_jsonls
        if jfiles
    ]

    t0 = time.time()
    with ctx.Pool(processes=len(lang_worker_args)) as pool:
        lang_results = pool.map(_lang_worker, lang_worker_args)
    elapsed = time.time() - t0
    logger.info(f"Per-language breakdown done in {elapsed:.1f}s")

    # Aggregate per-language results.
    total_lang_recordings = sum(p["num_recordings"] for p in lang_results)
    agg_counts = Counter()
    agg_raw_sec = Counter()
    agg_kept_count = Counter()
    agg_kept_sec = Counter()
    agg_speech_sec = Counter()
    agg_no_vad = Counter()
    for p in lang_results:
        agg_counts.update(p["lang_counts"])
        agg_raw_sec.update(p["lang_raw_sec"])
        agg_kept_count.update(p["lang_kept_count"])
        agg_kept_sec.update(p["lang_kept_sec"])
        agg_speech_sec.update(p["lang_speech_sec"])
        agg_no_vad.update(p["lang_no_vad"])

    print()
    print(f"--- Per-language breakdown (min_chunk={args.vad_min_chunk_sec}s, "
          f"merge_gap={args.vad_max_merge_gap_sec}s) ---")
    print()

    _print_lang_table(
        agg_counts, agg_raw_sec, agg_kept_count, agg_kept_sec,
        agg_speech_sec, agg_no_vad, total_lang_recordings, args.token_rate,
    )

    sys.stdout = _orig_stdout
    _save_results(buf.getvalue(), args, "vad_sweep")

    logger.info("Dry-run complete.")


if __name__ == "__main__":
    main()
