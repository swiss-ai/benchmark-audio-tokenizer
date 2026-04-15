#!/usr/bin/env python3
"""Patch SHAR cuts with captions and universal clip IDs.

Reads captions from a separate caption directory (produced by an annotation
pipeline), merges them into existing SHAR cuts as SupervisionSegments, rewrites
cut IDs to the universal format, and optionally pre-tokenizes the caption text.
Audio TAR files are left untouched — only cuts JSONL.gz files are rewritten.

Usage:
    python -m audio_tokenization.utils.prepare_data.postprocess.add_captions_to_shar \
        --shar-dir /capstor/.../SHAR_TODO/annotate/mrsaudio_music \
        --caption-dir /capstor/.../SHAR_TODO_captions/annotate/mrsaudio_music \
        --output-dir /capstor/.../SHAR/stage_2/mrsaudio_music \
        --text-tokenizer /capstor/.../tokenizer.json \
        --workers 32
"""

import argparse
import gzip
import json
import logging
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_captions(caption_dir: Path) -> dict[str, str]:
    """Load all captions from worker_*/caption.*.jsonl.gz into a {cut_id: caption} dict."""
    captions = {}
    for caption_file in sorted(caption_dir.glob("**/caption.*.jsonl.gz")):
        with gzip.open(caption_file, "rt") as f:
            for line in f:
                d = json.loads(line)
                captions[d["cut_id"]] = d["caption"]
    logger.info(f"Loaded {len(captions):,} captions from {caption_dir}")
    return captions


def _rewrite_cut_id_to_universal(cut_id: str) -> str:
    """Convert legacy cut IDs like 'MRSMusic/music093/chunk_000' to 'MRSMusic/music093@000000'."""
    # Split on last '/' to get source and chunk part
    parts = cut_id.rsplit("/", 1)
    if len(parts) == 2 and parts[1].startswith("chunk_"):
        chunk_num = int(parts[1].replace("chunk_", ""))
        return f"{parts[0]}@{chunk_num:06d}"
    # Already universal or unknown format — return as-is
    if "@" in cut_id:
        return cut_id
    return f"{cut_id}@000000"


def _assert_id_stability_for_symlinked_recordings(raw_dicts: list[dict], *, cuts_path: Path) -> None:
    """Reject ID rewrites when recording tar shards are left untouched.

    This script currently symlinks the original ``recording.*.tar`` files, so
    changing ``cut.id`` would make the output SHAR unreadable. Scripts that need
    to rewrite IDs must rebuild the recording tar shards too.
    """
    for d in raw_dicts:
        old_id = d["id"]
        new_id = _rewrite_cut_id_to_universal(old_id)
        if new_id != old_id:
            raise RuntimeError(
                "add_captions_to_shar cannot rewrite cut IDs while symlinking "
                f"recording tar shards unchanged (first mismatch in {cuts_path}: "
                f"{old_id!r} -> {new_id!r}). Full shard rewrite is required."
            )


def _process_shard(args):
    """Process one cuts shard: add captions, rewrite IDs, optionally tokenize."""
    cuts_path, captions, tokenizer_path, output_path, caption_custom_key = args

    from audio_tokenization.utils.prepare_data.text_ops import load_text_tokenizer
    tokenizer = load_text_tokenizer(tokenizer_path)

    raw_dicts = []
    with gzip.open(str(cuts_path), "rt") as f:
        for line in f:
            if line.strip():
                raw_dicts.append(json.loads(line))

    _assert_id_stability_for_symlinked_recordings(raw_dicts, cuts_path=cuts_path)

    patched = 0
    missing = 0
    for d in raw_dicts:
        old_id = d["id"]
        new_id = _rewrite_cut_id_to_universal(old_id)
        d["id"] = new_id

        # Update recording ID to match
        if "recording" in d and isinstance(d["recording"], dict):
            d["recording"]["id"] = new_id

        # Update supervision IDs
        for sup in d.get("supervisions", []):
            sup["id"] = new_id
            sup["recording_id"] = d.get("recording", {}).get("id", new_id)

        # Get caption: from external lookup or from cut.custom field
        if caption_custom_key:
            caption = (d.get("custom") or {}).get(caption_custom_key)
        else:
            caption = captions.get(old_id) if captions else None

        if caption:
            sup = {
                "id": new_id,
                "recording_id": d.get("recording", {}).get("id", new_id),
                "start": 0,
                "duration": d["duration"],
                "text": caption,
            }
            d["supervisions"] = [sup]

            # Pre-tokenize caption
            if tokenizer:
                tokens = tokenizer.encode(caption, add_special_tokens=False).ids
                d.setdefault("custom", {})["text_tokens"] = tokens

            patched += 1
        else:
            missing += 1

    # Write to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{output_path}.tmp.{os.getpid()}")
    with gzip.open(str(tmp), "wt") as f:
        for d in raw_dicts:
            print(json.dumps(d, ensure_ascii=False), file=f)
    tmp.rename(output_path)

    return patched, missing, len(raw_dicts)


def main():
    parser = argparse.ArgumentParser(
        description="Patch SHAR cuts with captions without rewriting recording tar shards"
    )
    parser.add_argument("--shar-dir", type=Path, required=True,
                        help="Input SHAR directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--caption-dir", type=Path, default=None,
                       help="External caption dir with worker_*/caption.*.jsonl.gz")
    group.add_argument("--caption-from-custom", type=str, default=None,
                       help="Read caption from cut.custom[KEY] instead of external dir "
                            "(e.g. --caption-from-custom text)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output SHAR directory")
    parser.add_argument("--text-tokenizer", type=str, default=None,
                        help="Path to tokenizer.json for pre-tokenizing captions")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if args.output_dir == args.shar_dir:
        raise ValueError("--output-dir must differ from --shar-dir")

    # Load captions from external dir if provided
    captions = _load_captions(args.caption_dir) if args.caption_dir else None

    # Find all cuts shards
    cuts_shards = sorted(args.shar_dir.glob("**/cuts.*.jsonl.gz"))
    if not cuts_shards:
        raise FileNotFoundError(f"No cuts shards in {args.shar_dir}")
    logger.info(f"Found {len(cuts_shards)} cuts shards")

    # Build work items
    work = []
    for cuts_path in cuts_shards:
        rel = cuts_path.relative_to(args.shar_dir)
        output_path = args.output_dir / rel
        work.append((cuts_path, captions, args.text_tokenizer, output_path, args.caption_from_custom))

    # Process in parallel. ID-rewriting modes are rejected inside workers until
    # this script is upgraded to rebuild recording tar shards as well.
    t0 = time.time()
    total_patched = total_missing = total_cuts = 0
    with Pool(min(args.workers, len(work))) as pool:
        for patched, missing, n_cuts in pool.imap_unordered(_process_shard, work):
            total_patched += patched
            total_missing += missing
            total_cuts += n_cuts

    # Copy audio tars (symlink to save space)
    for tar_path in sorted(args.shar_dir.glob("**/recording.*.tar")):
        rel = tar_path.relative_to(args.shar_dir)
        dst = args.output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(str(tar_path.resolve()), str(dst))

    # Copy other metadata files
    for name in ["shar_index.json", "_SUCCESS", "_PREPARE_STATE.json", "_worker_assignment.json"]:
        for src in args.shar_dir.glob(f"**/{name}"):
            rel = src.relative_to(args.shar_dir)
            dst = args.output_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)

    elapsed = time.time() - t0
    logger.info(
        f"Done in {elapsed:.1f}s: {total_patched:,} patched, "
        f"{total_missing:,} missing captions, {total_cuts:,} total cuts"
    )


if __name__ == "__main__":
    main()
