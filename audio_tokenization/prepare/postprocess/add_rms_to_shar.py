#!/usr/bin/env python3
"""Add precomputed rms_db to existing Shar cuts without resharding.

Updates each cuts JSONL.gz shard in-place (atomic rename), leaving audio
TAR files untouched.  Supports multiprocessing for speed.

Usage:
    # Single Shar directory
    python -m audio_tokenization.prepare.recipes.add_rms_to_shar \
        /path/to/eurospeech/italy_train

    # Multiple directories (all eurospeech)
    python -m audio_tokenization.prepare.recipes.add_rms_to_shar \
        /capstor/.../eurospeech/italy_train \
        /capstor/.../eurospeech/uk_train \
        --workers 32
"""

import argparse
import gzip
import json
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _process_shard(args):
    """Process one (cuts_shard, recording_shard) pair: compute rms_db for each cut.

    Strategy: load cuts via CutSet.from_shar (gives audio access), but
    read/write the raw JSONL dicts to avoid serializing Shar audio bytes.
    """
    shar_root, cuts_rel, rec_rel = args
    cuts_path = Path(shar_root) / cuts_rel

    if not cuts_path.exists():
        logger.warning(f"Cuts shard missing: {cuts_path}")
        return 0, 0

    # Read raw JSONL dicts (metadata only, no audio bytes)
    raw_dicts = []
    with gzip.open(str(cuts_path), "rt") as f:
        for line in f:
            raw_dicts.append(json.loads(line))

    # Quick check: if all cuts already have rms_db, skip audio loading entirely
    needs_rms = [d for d in raw_dicts if "rms_db" not in (d.get("custom") or {})]
    if not needs_rms:
        return 0, len(raw_dicts)

    # Load the shard with audio access for RMS computation
    from lhotse import CutSet
    import numpy as np

    rec_path = Path(shar_root) / rec_rel
    cuts = CutSet.from_shar(
        fields={"cuts": [str(cuts_path)], "recording": [str(rec_path)]},
        split_for_dataloading=False,
        shuffle_shards=False,
    )

    # Build id -> rms_db map from audio
    rms_map = {}
    errors = 0
    for cut in cuts:
        if (cut.custom or {}).get("rms_db") is not None:
            continue
        try:
            if cut.duration <= 0:
                rms_map[cut.id] = -200.0
                continue
            audio = cut.load_audio()  # (channels, samples)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            rms_map[cut.id] = round(20.0 * np.log10(rms + 1e-10), 2)
        except Exception:
            rms_map[cut.id] = -200.0
            errors += 1

    # Patch the raw dicts and write back
    computed = 0
    for d in raw_dicts:
        if d["id"] in rms_map:
            custom = d.get("custom") or {}
            custom["rms_db"] = rms_map[d["id"]]
            d["custom"] = custom
            computed += 1

    tmp_path = str(cuts_path) + ".rms_tmp"
    with gzip.open(tmp_path, "wt") as f:
        for d in raw_dicts:
            f.write(json.dumps(d) + "\n")
    os.replace(tmp_path, str(cuts_path))

    return computed, len(raw_dicts) - computed


def _safe_process_shard(args):
    """Wrapper that catches transient errors so one bad shard doesn't kill the pool.

    Fatal I/O errors (permission, disk full, OOM) are re-raised to abort.
    """
    try:
        return _process_shard(args)
    except (PermissionError, OSError, MemoryError) as e:
        logger.error(f"Fatal error on shard {args[1]}: {e}")
        raise
    except Exception as e:
        logger.warning(f"Shard failed: {args[1]} — {e}")
        return None


def process_shar_dir(shar_dir: str, num_workers: int = 8):
    """Add rms_db to all cuts in a Shar directory."""
    shar_path = Path(shar_dir)
    index_path = shar_path / "shar_index.json"

    if not index_path.exists():
        logger.error(f"No shar_index.json in {shar_dir}")
        return

    with open(index_path) as f:
        index = json.load(f)

    fields = index.get("fields", {})
    cuts_shards = fields.get("cuts", [])
    rec_shards = fields.get("recording", [])

    if len(cuts_shards) != len(rec_shards):
        raise ValueError(
            f"Mismatched shard counts: {len(cuts_shards)} cuts vs {len(rec_shards)} recordings"
        )

    logger.info(f"Processing {shar_dir}: {len(cuts_shards)} shards with {num_workers} workers")
    t0 = time.time()

    tasks = [(shar_dir, c, r) for c, r in zip(cuts_shards, rec_shards)]

    total_computed = 0
    total_skipped = 0

    errors = 0
    if num_workers <= 1:
        for task in tasks:
            try:
                computed, skipped = _process_shard(task)
                total_computed += computed
                total_skipped += skipped
            except Exception as e:
                logger.warning(f"Shard failed: {task[1]} — {e}")
                errors += 1
    else:
        with Pool(num_workers) as pool:
            for result in pool.imap_unordered(_safe_process_shard, tasks):
                if result is None:
                    errors += 1
                else:
                    total_computed += result[0]
                    total_skipped += result[1]

    elapsed = time.time() - t0
    logger.info(
        f"Done: {shar_dir} — {total_computed} computed, {total_skipped} already had rms_db, "
        f"{errors} shard errors, {elapsed:.1f}s ({total_computed / max(elapsed, 1):.0f} cuts/s)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Add precomputed rms_db to existing Shar cuts (metadata-only update)."
    )
    parser.add_argument("shar_dirs", nargs="+", help="Shar directories to process")
    parser.add_argument("--workers", type=int, default=64, help="Parallel workers (default: 64)")
    args = parser.parse_args()

    for shar_dir in args.shar_dirs:
        process_shar_dir(shar_dir, num_workers=args.workers)


if __name__ == "__main__":
    main()
