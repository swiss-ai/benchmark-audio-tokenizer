#!/usr/bin/env python3
"""Reassign source_id/clip_num for YODAS SHAR using original_audio_id + temporal order.

Only modifies cut.custom fields (source_id, clip_num, clip_start).
NEVER modifies cut.id, recording.id, or supervision IDs — those must match
audio tar filenames and are immutable after SHAR creation.

When --max-gap-sec is set, a temporal gap larger than this between consecutive
segments starts a new run: source_id gets a ``_R{run_idx}`` suffix and clip_num
resets to 0 (same convention as assign_universal_ids).

Usage:
    python -m audio_tokenization.utils.prepare_data.postprocess.patch_yodas_clip_nums \
        --shard-dir /capstor/.../granary_yodas_interleave/en000 \
        --max-gap-sec 30 \
        --workers 128
"""

import argparse
import gzip
import logging
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import orjson

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_cuts_file(cuts_path: Path) -> list[dict]:
    dicts = []
    with gzip.open(str(cuts_path), "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                dicts.append(orjson.loads(line))
    return dicts


def _write_cuts_file(args: tuple):
    cuts_path, dicts = args
    tmp = Path(f"{cuts_path}.tmp.{os.getpid()}")
    with gzip.open(str(tmp), "wb") as f:
        for d in dicts:
            f.write(orjson.dumps(d, option=orjson.OPT_APPEND_NEWLINE))
    tmp.rename(cuts_path)


def _process_shard_dir(shard_dir: Path, max_gap_sec: float | None, num_workers: int):
    # 1. Discover all cuts files
    cuts_files = sorted(shard_dir.glob("**/cuts.*.jsonl.gz"))
    if not cuts_files:
        logger.warning(f"No cuts files in {shard_dir}")
        return

    logger.info(f"Found {len(cuts_files)} cuts files in {shard_dir.name}")

    # 2. Load all cuts
    t0 = time.time()
    file_cuts: dict[Path, list[dict]] = {}
    with Pool(min(num_workers, len(cuts_files))) as pool:
        results = pool.map(_load_cuts_file, cuts_files)
    for path, dicts in zip(cuts_files, results):
        file_cuts[path] = dicts
    total_cuts = sum(len(v) for v in file_cuts.values())
    logger.info(f"Loaded {total_cuts:,} cuts in {time.time() - t0:.1f}s")

    # 3. Group by original_audio_id -> [(offset, duration, file_path, idx)]
    groups: dict[str, list[tuple[float, float, Path, int]]] = defaultdict(list)
    skipped = 0
    for path, dicts in file_cuts.items():
        for idx, d in enumerate(dicts):
            custom = d.get("custom", {}) or {}
            audio_id = custom.get("original_audio_id")
            if audio_id is None:
                skipped += 1
                continue
            offset = custom.get("original_audio_offset", 0.0)
            duration = d.get("duration", 0.0)
            groups[audio_id].append((offset, duration, path, idx))

    logger.info(
        f"Grouped into {len(groups):,} unique original_audio_ids "
        f"({skipped} cuts skipped, no original_audio_id)"
    )

    # 4. Sort each group by offset, split runs on large gaps, assign clip_num.
    #    Only modifies custom fields — never touches cut.id.
    assignments = 0
    total_runs = 0
    for audio_id, entries in groups.items():
        entries.sort(key=lambda x: x[0])

        run_idx = 0
        clip_num = 0
        prev_end = None

        for offset, duration, path, idx in entries:
            if max_gap_sec is not None and prev_end is not None:
                gap = offset - prev_end
                if gap > max_gap_sec:
                    run_idx += 1
                    clip_num = 0

            source_id = f"{audio_id}_R{run_idx}" if run_idx > 0 else audio_id

            d = file_cuts[path][idx]
            custom = d.get("custom", {}) or {}
            custom["source_id"] = source_id
            custom["clip_num"] = clip_num
            custom["clip_start"] = offset
            d["custom"] = custom

            prev_end = offset + duration
            clip_num += 1
            assignments += 1

        total_runs += run_idx + 1

    logger.info(
        f"Assigned clip_num to {assignments:,} cuts, "
        f"{total_runs:,} runs across {len(groups):,} videos"
    )

    # 5. Write back
    t1 = time.time()
    write_args = [(path, dicts) for path, dicts in file_cuts.items()]
    with Pool(min(num_workers, len(write_args))) as pool:
        pool.map(_write_cuts_file, write_args)
    logger.info(f"Wrote {len(write_args)} files in {time.time() - t1:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Patch YODAS SHAR: reassign source_id/clip_num by original_audio_id + temporal order"
    )
    parser.add_argument("--shard-dir", type=Path, required=True,
                        help="Path to a single language shard (e.g. .../granary_yodas_interleave/en000)")
    parser.add_argument("--max-gap-sec", type=float, default=None,
                        help="Max gap (seconds) between consecutive segments before starting a new run")
    parser.add_argument("--workers", type=int, default=128,
                        help="Number of parallel workers for I/O")
    args = parser.parse_args()

    if not args.shard_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {args.shard_dir}")

    _process_shard_dir(args.shard_dir, args.max_gap_sec, args.workers)
    logger.info("Done")


if __name__ == "__main__":
    main()
