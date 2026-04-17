#!/usr/bin/env python3
"""Repack WenetSpeech tar.gz files into proper Lhotse Shar format.

WenetSpeech tars have directory entries + wav-only files with nested paths.
Lhotse Shar expects flat paired entries: ``recording_id.wav`` + ``recording_id.json``.

This script reads each shard's cuts jsonl.gz (for recording metadata) and tar.gz
(for audio bytes), then writes a new tar in proper Lhotse Shar format.

Usage:
    python -m audio_tokenization.prepare.postprocess.repack_wenetspeech_shar \
        --data-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/wenetspeech/data \
        --output-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/wenetspeech/shar \
        --split L_fixed \
        --num-workers 20
"""

import argparse
import glob
import gzip
import io
import json
import logging
import multiprocessing as mp
import tarfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_recording_meta(cut: dict) -> dict:
    """Build Lhotse Shar-style recording JSON from a cut entry."""
    rec = cut["recording"]
    return {
        "id": rec["id"],
        "sources": [{"type": "shar", "channels": [0], "source": ""}],
        "sampling_rate": rec["sampling_rate"],
        "num_samples": rec["num_samples"],
        "duration": rec["duration"],
        "channel_ids": rec.get("channel_ids", [0]),
    }


def _repack_shard(args_tuple):
    shard_idx, cuts_path, tar_path, output_dir = args_tuple

    output_tar = output_dir / f"recording.{shard_idx:06d}.tar"
    output_cuts = output_dir / f"cuts.{shard_idx:06d}.jsonl.gz"

    if output_tar.exists() and output_cuts.exists():
        logger.info(f"Shard {shard_idx}: already exists, skipping")
        return {"shard": shard_idx, "written": 0, "skipped": True}

    t0 = time.time()

    # Load cuts metadata, keyed by recording ID
    cut_meta = {}
    cuts_list = []
    with gzip.open(cuts_path, "rt") as f:
        for line in f:
            cut = json.loads(line)
            cut_meta[cut["id"]] = cut
            cuts_list.append(cut)

    # Read audio bytes from source tar, keyed by recording ID
    audio_data = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isreg():
                continue
            rec_id = Path(member.name).stem
            audio_data[rec_id] = tf.extractfile(member).read()

    # Write new tar in proper Lhotse Shar format (data + json pairs)
    written = 0
    with tarfile.open(str(output_tar), "w") as out_tf:
        for cut in cuts_list:
            rec_id = cut["id"]
            wav_bytes = audio_data.get(rec_id)
            if wav_bytes is None:
                logger.warning(f"Shard {shard_idx}: missing audio for {rec_id}")
                continue

            # Write audio data
            wav_info = tarfile.TarInfo(name=f"{rec_id}.wav")
            wav_info.size = len(wav_bytes)
            out_tf.addfile(wav_info, io.BytesIO(wav_bytes))

            # Write JSON metadata
            meta = _build_recording_meta(cut)
            meta_bytes = json.dumps(meta).encode("utf-8")
            meta_info = tarfile.TarInfo(name=f"{rec_id}.json")
            meta_info.size = len(meta_bytes)
            out_tf.addfile(meta_info, io.BytesIO(meta_bytes))

            written += 1

    # Write new cuts jsonl.gz with updated recording sources
    with gzip.open(str(output_cuts), "wt") as f:
        for cut in cuts_list:
            rec = cut["recording"]
            rec["sources"] = [{"type": "shar", "channels": [0], "source": ""}]
            # Drop features field (external fbank paths won't be valid)
            cut.pop("features", None)
            f.write(json.dumps(cut) + "\n")

    elapsed = time.time() - t0
    logger.info(
        f"Shard {shard_idx}: {written} recordings in {elapsed:.1f}s "
        f"({written / elapsed:.0f} recs/s)"
    )
    return {"shard": shard_idx, "written": written, "skipped": False, "elapsed": elapsed}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Repack WenetSpeech tars into Lhotse Shar format",
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Source directory with cuts_*.jsonl.gz and cuts_*.tar.gz")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for repacked Shar files")
    parser.add_argument("--split", type=str, default="L_fixed",
                        help="Split prefix to repack (default: L_fixed)")
    parser.add_argument("--num-workers", type=int, default=20,
                        help="Number of parallel workers (default: 20)")
    args = parser.parse_args(argv)

    # Discover shards
    cuts_files = sorted(glob.glob(str(args.data_dir / f"cuts_{args.split}.*.jsonl.gz")))
    tar_files = sorted(glob.glob(str(args.data_dir / f"cuts_{args.split}.*.tar.gz")))

    if len(cuts_files) != len(tar_files):
        raise ValueError(f"Mismatch: {len(cuts_files)} cuts vs {len(tar_files)} tars")
    if not cuts_files:
        raise FileNotFoundError(f"No shards found for split {args.split} in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Repacking {len(cuts_files)} shards from {args.data_dir}")
    logger.info(f"Output: {args.output_dir}")

    worker_args = [
        (i, cuts_files[i], tar_files[i], args.output_dir)
        for i in range(len(cuts_files))
    ]

    num_workers = min(args.num_workers, len(worker_args))
    t0 = time.time()

    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_repack_shard, worker_args)

    elapsed = time.time() - t0
    total_written = sum(r["written"] for r in results if not r.get("skipped"))
    total_skipped = sum(1 for r in results if r.get("skipped"))
    logger.info(
        f"Done in {elapsed:.1f}s: {total_written} recordings repacked, "
        f"{total_skipped} shards skipped (already existed)"
    )

    # Build shar_index.json
    out_root = args.output_dir.resolve()
    index_cuts = sorted(
        str(p.resolve().relative_to(out_root))
        for p in args.output_dir.glob("cuts.*.jsonl.gz")
    )
    index_recs = sorted(
        str(p.resolve().relative_to(out_root))
        for p in args.output_dir.glob("recording.*.tar")
    )
    payload = {
        "version": 1,
        "fields": {
            "cuts": index_cuts,
            "recording": index_recs,
        },
    }
    index_path = args.output_dir / "shar_index.json"
    index_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote {index_path} ({len(index_cuts)} shards)")


if __name__ == "__main__":
    main()
