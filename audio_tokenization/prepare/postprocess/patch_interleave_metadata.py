#!/usr/bin/env python3
"""Patch existing SHAR with interleaving metadata (source_id/clip_num/clip_start).

Uses a clip_id_parser to derive (source_id, clip_num) from cut.id, groups globally
across all cuts files, sorts by clip_num, and assigns dense clip_num into cut.custom.
NEVER modifies cut.id, recording.id, or supervision IDs.

When --max-gap-sec is set, a temporal gap larger than this between consecutive
segments starts a new run: source_id gets a ``_R{run_idx}`` suffix and clip_num
resets to 0. The gap threshold is always specified in seconds; for parsers whose
clip_num is in other units (e.g. milliseconds), it is converted automatically.

Usage:
    python -m audio_tokenization.prepare.postprocess.patch_interleave_metadata \
        --shar-dir /capstor/.../libriheavy_large \
        --clip-id-parser libriheavy \
        --workers 128

    python -m audio_tokenization.prepare.postprocess.patch_interleave_metadata \
        --shar-dir /capstor/.../eurospeech \
        --clip-id-parser eurospeech \
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

from audio_tokenization.utils.clip_id_parsers import (
    get_clip_id_parser,
    get_clip_num_to_sec,
)

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


def _process_shar_dir(
    shar_dir: Path,
    parser_name: str,
    max_gap_clip_num: float | None,
    num_workers: int,
):
    parser = get_clip_id_parser(parser_name)

    # 1. Discover all cuts files
    cuts_files = sorted(shar_dir.glob("**/cuts.*.jsonl.gz"))
    if not cuts_files:
        logger.warning(f"No cuts files in {shar_dir}")
        return

    logger.info(f"Found {len(cuts_files)} cuts files")

    # 2. Load all cuts in parallel
    t0 = time.time()
    file_cuts: dict[Path, list[dict]] = {}
    with Pool(min(num_workers, len(cuts_files))) as pool:
        results = pool.map(_load_cuts_file, cuts_files)
    for path, dicts in zip(cuts_files, results):
        file_cuts[path] = dicts
    total_cuts = sum(len(v) for v in file_cuts.values())
    logger.info(f"Loaded {total_cuts:,} cuts in {time.time() - t0:.1f}s")

    # 3. Parse IDs and group globally: source_id -> [(raw_clip_num, duration, file_path, idx)]
    clip_num_scale = get_clip_num_to_sec(parser_name)
    groups: dict[str, list[tuple[int, float, Path, int]]] = defaultdict(list)
    errors = 0
    for path, dicts in file_cuts.items():
        for idx, d in enumerate(dicts):
            cut_id = d["id"]
            base_id = cut_id.rsplit("@", 1)[0] if "@" in cut_id else cut_id
            try:
                source_id, clip_num = parser(base_id)
                duration = d.get("duration", 0.0)
                groups[source_id].append((clip_num, duration, path, idx))
            except ValueError:
                errors += 1

    logger.info(
        f"Parsed into {len(groups):,} groups "
        f"({errors:,} parse errors)"
    )

    # 4. Sort each group, split runs on gaps, assign dense clip_num.
    #    Gap is computed as: start[i] - end[i-1], where end = start + duration.
    #    For parsers with time-based clip_num (ms), duration is converted to the
    #    same unit. For sequential clip_num, gap = clip_num difference directly.
    assignments = 0
    total_runs = 0
    for source_id, entries in groups.items():
        entries.sort(key=lambda x: x[0])

        run_idx = 0
        dense_clip_num = 0
        prev_end = None  # end position in clip_num units

        for raw_clip_num, duration, path, idx in entries:
            if max_gap_clip_num is not None and prev_end is not None:
                gap = raw_clip_num - prev_end
                if gap > max_gap_clip_num:
                    run_idx += 1
                    dense_clip_num = 0

            final_source_id = f"{source_id}_R{run_idx}" if run_idx > 0 else source_id

            d = file_cuts[path][idx]
            custom = d.get("custom", {}) or {}
            custom["source_id"] = final_source_id
            custom["clip_num"] = dense_clip_num
            custom["clip_start"] = float(d.get("start", 0.0))
            d["custom"] = custom

            # Compute end position in clip_num units
            if clip_num_scale is not None:
                # clip_num is time-based (e.g. ms): end = start + duration_in_clip_units
                prev_end = raw_clip_num + duration / clip_num_scale
            else:
                # clip_num is sequential: just use clip_num as-is
                prev_end = raw_clip_num
            dense_clip_num += 1
            assignments += 1

        total_runs += run_idx + 1

    logger.info(
        f"Assigned {assignments:,} cuts, "
        f"{total_runs:,} runs across {len(groups):,} groups"
    )

    # 5. Write back in parallel
    t1 = time.time()
    write_args = [(path, dicts) for path, dicts in file_cuts.items()]
    with Pool(min(num_workers, len(write_args))) as pool:
        pool.map(_write_cuts_file, write_args)
    logger.info(f"Wrote {len(write_args)} files in {time.time() - t1:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Patch SHAR with interleaving metadata using a clip_id_parser"
    )
    parser.add_argument("--shar-dir", type=Path, required=True,
                        help="Root SHAR directory to patch")
    parser.add_argument("--clip-id-parser", type=str, required=True,
                        help="Name of the clip ID parser (e.g. libriheavy, wenetspeech, eurospeech)")
    parser.add_argument("--max-gap-sec", type=float, default=None,
                        help="Max gap in seconds between consecutive clips before starting a new run")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse sample IDs and print stats without writing")
    args = parser.parse_args()

    if not args.shar_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {args.shar_dir}")

    # Validate parser
    get_clip_id_parser(args.clip_id_parser)

    # Convert max-gap-sec to clip_num units
    max_gap_clip_num = None
    if args.max_gap_sec is not None:
        scale = get_clip_num_to_sec(args.clip_id_parser)
        if scale is not None:
            max_gap_clip_num = args.max_gap_sec / scale
            logger.info(f"max-gap-sec={args.max_gap_sec}s -> clip_num threshold={max_gap_clip_num:.0f} (scale={scale})")
        else:
            max_gap_clip_num = args.max_gap_sec
            logger.info(f"max-gap-sec={args.max_gap_sec} (no time scale for this parser)")

    cuts_files = sorted(args.shar_dir.glob("**/cuts.*.jsonl.gz"))
    if not cuts_files:
        raise FileNotFoundError(f"No cuts files in {args.shar_dir}")

    logger.info(f"Found {len(cuts_files)} cuts files, parser={args.clip_id_parser}")

    if args.dry_run:
        parser_fn = get_clip_id_parser(args.clip_id_parser)
        with gzip.open(str(cuts_files[0]), "rb") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                d = orjson.loads(line.strip())
                cut_id = d["id"]
                base_id = cut_id.rsplit("@", 1)[0] if "@" in cut_id else cut_id
                try:
                    sid, cn = parser_fn(base_id)
                    logger.info(f"  {cut_id} -> source_id={sid}, clip_num={cn}")
                except ValueError as e:
                    logger.info(f"  {cut_id} -> PARSE ERROR: {e}")
        logger.info("Dry run — no files written.")
        return

    _process_shar_dir(args.shar_dir, args.clip_id_parser, max_gap_clip_num, args.workers)


if __name__ == "__main__":
    main()
