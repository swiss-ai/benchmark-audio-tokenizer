#!/usr/bin/env python3
"""Patch SHAR with interleaving metadata using custom fields for grouping and sorting.

Two-pass approach for memory efficiency:
  Pass 1: Scan all cuts in parallel workers, collect lightweight index tuples
  Pass 2: Compute assignments in main process, rewrite files in parallel workers

Only modifies cut.custom — never touches cut.id.

Usage:
    python -m audio_tokenization.prepare.postprocess.patch_interleave_by_field \
        --shar-dir /capstor/.../mls_v2 \
        --group-by speaker_id chapter_id \
        --sort-by begin_time \
        --max-gap-sec 10 \
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

# Module-level state for multiprocessing
_group_by = None
_sort_by = None


def _init_scanner(group_by, sort_by):
    global _group_by, _sort_by
    _group_by = group_by
    _sort_by = sort_by


def _scan_cuts_file(cuts_path: Path) -> list[tuple]:
    """Pass 1: extract lightweight (group_key, sort_val, duration, file_path, idx)."""
    entries = []
    with gzip.open(str(cuts_path), "rb") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = orjson.loads(line)
            custom = d.get("custom", {}) or {}

            key_parts = []
            skip = False
            for field in _group_by:
                val = custom.get(field)
                if val is None:
                    skip = True
                    break
                key_parts.append(str(val))
            if skip:
                continue

            group_key = "_".join(key_parts)
            sort_val = float(custom.get(_sort_by, 0.0))
            duration = d.get("duration", 0.0)
            entries.append((group_key, sort_val, duration, str(cuts_path), idx))
    return entries


def _apply_assignments(args: tuple):
    """Pass 2: re-read one cuts file, apply assignments, write back."""
    cuts_path_str, assignments_for_file = args
    cuts_path = Path(cuts_path_str)

    dicts = []
    with gzip.open(str(cuts_path), "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                dicts.append(orjson.loads(line))

    for idx, source_id, clip_num, clip_start in assignments_for_file:
        d = dicts[idx]
        custom = d.get("custom", {}) or {}
        custom["source_id"] = source_id
        custom["clip_num"] = clip_num
        custom["clip_start"] = clip_start
        d["custom"] = custom

    tmp = Path(f"{cuts_path}.tmp.{os.getpid()}")
    with gzip.open(str(tmp), "wb") as f:
        for d in dicts:
            f.write(orjson.dumps(d, option=orjson.OPT_APPEND_NEWLINE))
    tmp.rename(cuts_path)
    return len(assignments_for_file)


def _process_shar_dir(
    shar_dir: Path,
    group_by: list[str],
    sort_by: str,
    max_gap_sec: float | None,
    num_workers: int,
):
    cuts_files = sorted(shar_dir.glob("**/cuts.*.jsonl.gz"))
    if not cuts_files:
        logger.warning(f"No cuts files in {shar_dir}")
        return

    logger.info(f"Found {len(cuts_files)} cuts files")

    # Pass 1: scan all cuts to build lightweight index
    t0 = time.time()
    groups: dict[str, list[tuple[float, float, str, int]]] = defaultdict(list)
    total_scanned = 0

    with Pool(
        min(num_workers, len(cuts_files)),
        initializer=_init_scanner,
        initargs=(group_by, sort_by),
    ) as pool:
        for entries in pool.imap_unordered(_scan_cuts_file, cuts_files):
            for group_key, sort_val, duration, file_path, idx in entries:
                groups[group_key].append((sort_val, duration, file_path, idx))
                total_scanned += 1

    logger.info(
        f"Pass 1: scanned {total_scanned:,} cuts into {len(groups):,} groups "
        f"in {time.time() - t0:.1f}s"
    )

    # Compute assignments: sort each group, split runs, assign clip_num
    file_assignments: dict[str, list[tuple[int, str, int, float]]] = defaultdict(list)
    total_runs = 0

    for group_key, entries in groups.items():
        entries.sort(key=lambda x: x[0])

        run_idx = 0
        clip_num = 0
        prev_end = None

        for sort_val, duration, file_path, idx in entries:
            if max_gap_sec is not None and prev_end is not None:
                gap = sort_val - prev_end
                if gap > max_gap_sec:
                    run_idx += 1
                    clip_num = 0

            source_id = f"{group_key}_R{run_idx}" if run_idx > 0 else group_key
            file_assignments[file_path].append((idx, source_id, clip_num, sort_val))

            prev_end = sort_val + duration
            clip_num += 1

        total_runs += run_idx + 1

    del groups

    logger.info(
        f"Computed {total_scanned:,} assignments, "
        f"{total_runs:,} runs across {len(file_assignments)} files"
    )

    # Pass 2: rewrite files with assignments
    t1 = time.time()
    work = list(file_assignments.items())
    total_written = 0
    with Pool(min(num_workers, len(work))) as pool:
        for n in pool.imap_unordered(_apply_assignments, work):
            total_written += n
    logger.info(f"Pass 2: wrote {total_written:,} assignments in {time.time() - t1:.1f}s")

    # Generate top-level shar_index.json if missing
    top_index = shar_dir / "shar_index.json"
    if not top_index.exists():
        cuts_rel = sorted(str(f.relative_to(shar_dir)) for f in shar_dir.glob("**/cuts.*.jsonl.gz"))
        recs_rel = sorted(str(f.relative_to(shar_dir)) for f in shar_dir.glob("**/recording.*.tar"))
        if cuts_rel and recs_rel:
            index = {"version": 1, "fields": {"cuts": cuts_rel, "recording": recs_rel}}
            with open(top_index, "wb") as f:
                f.write(orjson.dumps(index, option=orjson.OPT_INDENT_2))
            logger.info(f"Created shar_index.json: {len(cuts_rel)} cuts, {len(recs_rel)} recordings")


def main():
    parser = argparse.ArgumentParser(
        description="Patch SHAR with interleaving metadata using custom fields"
    )
    parser.add_argument("--shar-dir", type=Path, required=True)
    parser.add_argument("--group-by", nargs="+", required=True,
                        help="Custom field(s) to group by (e.g. speaker_id chapter_id)")
    parser.add_argument("--sort-by", type=str, required=True,
                        help="Custom field to sort by within each group (e.g. begin_time)")
    parser.add_argument("--max-gap-sec", type=float, default=None,
                        help="Max gap in seconds between consecutive clips before starting a new run")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.shar_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {args.shar_dir}")

    cuts_files = sorted(args.shar_dir.glob("**/cuts.*.jsonl.gz"))
    if not cuts_files:
        raise FileNotFoundError(f"No cuts files in {args.shar_dir}")

    logger.info(f"Found {len(cuts_files)} cuts files, group_by={args.group_by}, sort_by={args.sort_by}")

    if args.dry_run:
        import gzip as gz
        with gz.open(str(cuts_files[0]), "rb") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                d = orjson.loads(line.strip())
                c = d.get("custom", {})
                key = "_".join(str(c.get(f, "?")) for f in args.group_by)
                sort_val = c.get(args.sort_by, "?")
                logger.info(f"  {d['id']} -> group={key}, {args.sort_by}={sort_val}")
        logger.info("Dry run — no files written.")
        return

    _process_shar_dir(args.shar_dir, args.group_by, args.sort_by, args.max_gap_sec, args.workers)


if __name__ == "__main__":
    main()
