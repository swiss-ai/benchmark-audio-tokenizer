#!/usr/bin/env python3
"""Convert individual audio files + VAD JSONL to Lhotse Shar format.

Reads audio files from a directory tree and VAD timestamps from JSONL files
(one per language+year or arbitrary grouping), applies VAD-aware chunking
via ``merge_and_pack_vad``, resamples to target SR, and writes to Shar.

Each worker processes a subset of JSONL files and writes to its own
``worker_XX/`` sub-directory.  After all workers finish, a merged
``shar_index.json`` is written so that the tokenization pipeline can load
the output directly.

Designed for datasets like VoxPopuli where audio is stored as individual
files (e.g. ``.ogg``) rather than WebDataset tar shards.

Usage:
    python -m audio_tokenization.prepare.prepare_audio_dir_to_shar \
        --audio-root /capstor/.../voxpopuli/raw_audios \
        --jsonl-files /capstor/.../per_lang_year/*.jsonl \
        --shar-dir /iopsstor/.../voxpopuli_shar \
        --target-sr 24000 \
        --num-workers 272 \
        --shard-size 2000 \
        --shar-format flac
"""

import argparse
from collections import Counter
import logging
import time
from pathlib import Path

from audio_tokenization.prepare.common import (
    build_audio_index,
    check_worker_reuse,
    distribute_round_robin,
    ensure_worker_assignment,
    init_worker_process,
    run_aggregate,
    run_pool_and_finalize,
    set_universal_cut_id,
    to_mono,
    write_worker_result,
)
from audio_tokenization.prepare.preprocess.chunking import (
    _parse_vad_jsonl_line,
    merge_and_pack_vad,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_ITEMS_KEY = "resolved_jsonls"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _convert_worker(args_tuple):
    """Convert a subset of VAD JSONL entries to Shar.

    Each worker writes to its own ``worker_XX/`` directory to avoid contention.
    Resume is considered complete only when ``worker_XX/_SUCCESS`` exists.
    """
    (
        worker_id,
        jsonl_paths,
        audio_index,
        shar_dir,
        target_sr,
        shard_size,
        shar_format,
        min_sr,
        mono_downmix,
        vad_max_chunk_sec,
        vad_min_chunk_sec,
        vad_sample_rate,
        vad_max_merge_gap_sec,
        vad_max_duration_sec,
        audio_ext,
        resampling_backend,
    ) = args_tuple

    reused = check_worker_reuse(worker_id, shar_dir)
    if reused is not None:
        return reused
    init_worker_process(resampling_backend)

    from lhotse import Recording
    from lhotse.shar import SharWriter

    reason_counts = Counter()
    runtime_counts = Counter()

    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    t0 = time.time()
    written = skipped = errors = 0
    total_duration_sec = 0.0

    with SharWriter(
        output_dir=str(worker_dir),
        fields={"recording": shar_format},
        shard_size=shard_size,
    ) as writer:
        for jsonl_path in jsonl_paths:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parsed = _parse_vad_jsonl_line(
                        line,
                        with_duration=True,
                        with_sample_rate=True,
                        with_lang=True,
                    )
                    if parsed is None:
                        runtime_counts["parse_failed"] += 1
                        continue

                    key, timestamps, duration_sec, sr, lang = parsed

                    # Resolve audio path
                    audio_path = audio_index.get(key)
                    if audio_path is None:
                        runtime_counts["missing_audio"] += 1
                        skipped += 1
                        continue

                    # Build Lhotse recording from file
                    try:
                        recording = Recording.from_file(audio_path)
                        cut = recording.to_cut()
                    except Exception as e:
                        errors += 1
                        runtime_counts["failed_build_cut"] += 1
                        if errors <= 5:
                            logger.warning(
                                f"Worker {worker_id} error loading {key}: {e}"
                            )
                        continue

                    # Min sample rate check
                    if min_sr and cut.sampling_rate < min_sr:
                        skipped += 1
                        runtime_counts["skipped_min_sr"] += 1
                        continue

                    # Resample if needed
                    if target_sr and cut.sampling_rate != target_sr:
                        cut = cut.resample(target_sr)
                        runtime_counts["resampled"] += 1

                    # VAD chunking
                    if not timestamps:
                        reason_counts["empty_vad"] += 1
                        skipped += 1
                        continue

                    ranges = merge_and_pack_vad(
                        timestamps=timestamps,
                        audio_duration_sec=float(cut.duration),
                        sample_rate=vad_sample_rate,
                        max_merge_gap_sec=vad_max_merge_gap_sec,
                        max_chunk_sec=vad_max_chunk_sec,
                        min_chunk_sec=vad_min_chunk_sec,
                        max_duration_sec=vad_max_duration_sec,
                    )

                    if not ranges:
                        reason_counts["chunks_below_min_duration"] += 1
                        skipped += 1
                        continue

                    reason_counts["chunked"] += 1

                    for chunk_idx, (offset, chunk_duration) in enumerate(ranges):
                        try:
                            try:
                                subcut = cut.truncate(
                                    offset=offset,
                                    duration=chunk_duration,
                                    preserve_id=False,
                                )
                            except TypeError:
                                subcut = cut.truncate(
                                    offset=offset, duration=chunk_duration
                                )

                            subcut = to_mono(
                                subcut,
                                mono_downmix=mono_downmix,
                                stats=runtime_counts,
                            )
                            subcut.custom = subcut.custom or {}
                            subcut.custom["global_offset_sec"] = offset
                            subcut.custom["lang"] = lang
                            set_universal_cut_id(
                                subcut,
                                key,
                                chunk_idx,
                                clip_start=offset,
                            )

                            writer.write(subcut)
                            written += 1
                            total_duration_sec += subcut.duration
                            runtime_counts["cuts_written"] += 1
                        except Exception as e:
                            errors += 1
                            runtime_counts["processing_errors"] += 1
                            if errors <= 5:
                                logger.warning(
                                    f"Worker {worker_id} error on chunk "
                                    f"{key}@{offset:.1f}: {e}"
                                )

                    if written % 1000 == 0 and written > 0:
                        elapsed = time.time() - t0
                        logger.info(
                            f"Worker {worker_id}: {written} written, "
                            f"{skipped} skipped, {errors} errors "
                            f"({written / elapsed:.1f} samples/s)"
                        )

    if reason_counts:
        logger.info(f"Worker {worker_id} VAD reasons: {dict(reason_counts)}")

    return write_worker_result(
        worker_id=worker_id, worker_dir=worker_dir,
        written=written, skipped=skipped, errors=errors,
        total_duration_sec=total_duration_sec,
        runtime_counts=runtime_counts, t0=t0,
        extra_stats={"reason_counts": dict(reason_counts)},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert audio dir + VAD JSONL -> Lhotse Shar (parallel)",
    )

    # Input
    parser.add_argument("--audio-root", type=Path, required=None,
                        help="Root directory of audio files (searched recursively)")
    parser.add_argument("--audio-ext", type=str, default=".ogg",
                        help="Audio file extension (default: .ogg)")
    parser.add_argument("--jsonl-files", nargs="+", required=None,
                        help="VAD JSONL file paths (shell glob expanded by caller)")

    # Shar output
    parser.add_argument("--shar-dir", type=Path, default=None,
                        help="Output directory for Shar format")
    parser.add_argument("--shard-size", type=int, default=2000,
                        help="Samples per Shar shard (default: 2000)")
    parser.add_argument("--shar-format", type=str, default="flac",
                        choices=["flac", "wav", "mp3", "opus"],
                        help="Audio format in Shar (default: flac)")

    # Audio processing
    parser.add_argument("--target-sr", type=int, default=24000,
                        help="Target sample rate (default: 24000)")
    parser.add_argument("--resampling-backend", type=str, default=None,
                        choices=["default", "sox"],
                        help="Lhotse resampling backend override (default: use "
                             "$LHOTSE_RESAMPLING_BACKEND or 'default')")
    parser.add_argument("--min-sr", type=int, default=16000,
                        help="Drop audio below this sample rate (default: 16000)")
    parser.add_argument("--no-mono-downmix", action="store_true",
                        help="Select channel 0 instead of averaging stereo channels")

    # VAD chunking
    parser.add_argument("--vad-max-chunk-sec", type=float, default=200.0,
                        help="Target max duration while packing VAD segments")
    parser.add_argument("--vad-min-chunk-sec", type=float, default=5.0,
                        help="Drop chunks shorter than this duration")
    parser.add_argument("--vad-sample-rate", type=int, default=16000,
                        help="Sample rate used to decode VAD timestamp units")
    parser.add_argument("--vad-max-merge-gap-sec", type=float, default=1.0,
                        help="Merge adjacent VAD spans when silence gap <= this threshold")
    parser.add_argument("--vad-max-duration-sec", type=float, default=None,
                        help="Drop atomic speech segments longer than this "
                             "(default: same as --vad-max-chunk-sec)")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: one per JSONL file)")

    parser.add_argument("--aggregate", type=Path, default=None, metavar="SHAR_ROOT",
                        help="Aggregate stats from completed multi-node runs and exit.")

    args = parser.parse_args(argv)

    # ---- Aggregate mode ----
    if args.aggregate is not None:
        run_aggregate(args.aggregate)
        return

    if args.audio_root is None:
        parser.error("--audio-root is required (unless using --aggregate)")
    if not args.jsonl_files:
        parser.error("--jsonl-files is required (unless using --aggregate)")
    if args.shar_dir is None:
        parser.error("--shar-dir is required (unless using --aggregate)")

    if not args.audio_root.is_dir():
        raise NotADirectoryError(f"Audio root not found: {args.audio_root}")

    resolved_jsonls = sorted(args.jsonl_files)
    if not resolved_jsonls:
        raise FileNotFoundError("No JSONL files provided via --jsonl-files")

    args.shar_dir.mkdir(parents=True, exist_ok=True)

    num_workers = ensure_worker_assignment(
        args.shar_dir, resolved_jsonls, args.num_workers, _ITEMS_KEY, "JSONL files",
    )

    logger.info(f"Found {len(resolved_jsonls)} JSONL files, using {num_workers} workers")
    logger.info(f"Output: {args.shar_dir}")

    # Build audio index (stem -> full path)
    logger.info(f"Building audio index from {args.audio_root} (*{args.audio_ext}) ...")
    t_idx = time.time()
    audio_index = build_audio_index(args.audio_root, f"**/*{args.audio_ext}")
    logger.info(
        f"Indexed {len(audio_index):,} audio files in {time.time() - t_idx:.1f}s"
    )
    if not audio_index:
        raise FileNotFoundError(
            f"No *{args.audio_ext} files found under {args.audio_root}"
        )

    # Distribute JSONL files across workers (round-robin)
    worker_jsonls = distribute_round_robin(resolved_jsonls, num_workers)

    # Validate VAD params
    if args.vad_max_chunk_sec <= 0:
        raise ValueError("--vad-max-chunk-sec must be > 0")
    if args.vad_min_chunk_sec < 0:
        raise ValueError("--vad-min-chunk-sec must be >= 0")
    if args.vad_min_chunk_sec > args.vad_max_chunk_sec:
        raise ValueError("--vad-min-chunk-sec must be <= --vad-max-chunk-sec")
    if args.vad_sample_rate <= 0:
        raise ValueError("--vad-sample-rate must be > 0")
    if args.vad_max_merge_gap_sec < 0:
        raise ValueError("--vad-max-merge-gap-sec must be >= 0")

    worker_args = [
        (
            wid,
            jsonls,
            audio_index,
            str(args.shar_dir),
            args.target_sr,
            args.shard_size,
            args.shar_format,
            args.min_sr,
            not args.no_mono_downmix,
            args.vad_max_chunk_sec,
            args.vad_min_chunk_sec,
            args.vad_sample_rate,
            args.vad_max_merge_gap_sec,
            args.vad_max_duration_sec,
            args.audio_ext,
            args.resampling_backend,
        )
        for wid, jsonls in enumerate(worker_jsonls)
        if jsonls
    ]

    run_pool_and_finalize(_convert_worker, worker_args, args.shar_dir, num_workers)


if __name__ == "__main__":
    main()
