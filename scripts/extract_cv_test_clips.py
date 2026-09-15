#!/usr/bin/env python3
"""Extract longer test clips from Common Voice 24 tar.zst archives.

For each language, selects test clips >= min_duration_ms and copies/extracts
them into an output directory along with a metadata.tsv for ground truth.

Usage:
    python scripts/extract_cv_test_clips.py \
        --lang zh-HK de \
        --min-duration-ms 7000 \
        --num-samples 10 \
        --output-root results/inference
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import tarfile

import zstandard as zstd


CV_ROOT = "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24"


def read_tsv(path: str) -> list[dict[str, str]]:
    """Read a TSV file into a list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def get_test_clips_with_duration(
    lang_dir: str, min_duration_ms: int, num_samples: int
) -> list[dict]:
    """Return test clips >= min_duration_ms, sorted longest first."""
    # Read durations
    dur_rows = read_tsv(os.path.join(lang_dir, "clip_durations.tsv"))
    dur_map = {}
    for row in dur_rows:
        dur_map[row["clip"]] = int(row["duration[ms]"])

    # Read test split
    test_rows = read_tsv(os.path.join(lang_dir, "test.tsv"))

    # Filter and sort by duration (longest first)
    candidates = []
    for row in test_rows:
        clip = row["path"]
        dur = dur_map.get(clip, 0)
        if dur >= min_duration_ms:
            candidates.append({
                "clip": clip,
                "duration_ms": dur,
                "sentence": row.get("sentence", ""),
            })

    candidates.sort(key=lambda x: x["duration_ms"], reverse=True)
    return candidates[:num_samples]


def extract_from_tar_zst(
    tar_zst_path: str, clip_names: set[str], output_dir: str
) -> int:
    """Extract specific clips from a tar.zst archive."""
    dctx = zstd.ZstdDecompressor()
    extracted = 0
    with open(tar_zst_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    basename = os.path.basename(member.name)
                    if basename in clip_names:
                        # Extract to output dir with flat name
                        out_path = os.path.join(output_dir, basename)
                        if not os.path.exists(out_path):
                            f = tar.extractfile(member)
                            if f:
                                with open(out_path, "wb") as out_f:
                                    shutil.copyfileobj(f, out_f)
                                extracted += 1
                        else:
                            extracted += 1  # already exists
                        if extracted >= len(clip_names):
                            break
    return extracted


def copy_from_clips_dir(
    clips_dir: str, clip_names: set[str], output_dir: str
) -> int:
    """Copy specific clips from an already-extracted clips directory."""
    copied = 0
    for clip in clip_names:
        src = os.path.join(clips_dir, clip)
        dst = os.path.join(output_dir, clip)
        if os.path.exists(dst):
            copied += 1
        elif os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  WARNING: {clip} not found in {clips_dir}")
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", nargs="+", required=True,
                        help="Language codes (e.g. zh-HK de)")
    parser.add_argument("--min-duration-ms", type=int, default=7000)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-root", type=str, default="results/inference")
    parser.add_argument("--cv-root", type=str, default=CV_ROOT)
    args = parser.parse_args()

    for lang in args.lang:
        lang_dir = os.path.join(args.cv_root, lang)
        if not os.path.isdir(lang_dir):
            print(f"Skipping {lang}: directory not found at {lang_dir}")
            continue

        dataset_name = f"commonvoice_{lang.replace('-', '_')}"
        output_dir = os.path.join(args.output_root, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Language: {lang}  ->  {output_dir}")
        print(f"{'='*60}")

        # Get candidate clips
        clips = get_test_clips_with_duration(
            lang_dir, args.min_duration_ms, args.num_samples
        )
        print(f"  Selected {len(clips)} clips >= {args.min_duration_ms}ms")
        if not clips:
            continue

        for c in clips:
            print(f"    {c['clip']}  {c['duration_ms']}ms  {c['sentence'][:60]}")

        clip_names = {c["clip"] for c in clips}

        # Extract or copy audio files
        clips_dir = os.path.join(lang_dir, "clips")
        if os.path.isdir(clips_dir):
            print(f"\n  Copying from extracted clips/ directory ...")
            n = copy_from_clips_dir(clips_dir, clip_names, output_dir)
        else:
            tar_path = os.path.join(lang_dir, "test_clips.tar.zst")
            if not os.path.isfile(tar_path):
                print(f"  ERROR: neither clips/ dir nor {tar_path} found")
                continue
            print(f"\n  Extracting from {tar_path} ...")
            n = extract_from_tar_zst(tar_path, clip_names, output_dir)

        print(f"  {n}/{len(clips)} audio files ready")

        # Write metadata.tsv
        meta_path = os.path.join(output_dir, "metadata.tsv")
        with open(meta_path, "w") as f:
            f.write("filename\ttext\n")
            for c in clips:
                f.write(f"{c['clip']}\t{c['sentence']}\n")
        print(f"  Wrote {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
