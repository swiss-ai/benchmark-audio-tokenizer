#!/usr/bin/env python3
"""CLI wrapper for merging Megatron indexed datasets."""

from __future__ import annotations

import argparse
import logging

from audio_tokenization.utils.indexed_dataset.merge_indexed_dataset import (
    merge_indexed_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for merge discovery and execution."""
    parser = argparse.ArgumentParser(
        description="Merge Megatron indexed datasets (.bin/.idx) without temp staging.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory to scan for prefixes. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix (without .bin/.idx suffix).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directories.",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Use multimodal index mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output .bin/.idx files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered prefixes; do not merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Keep logging minimal; this is primarily an ops utility.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("merge_indexed_datasets")

    if args.dry_run:
        # Dry-run validates directories and shard completeness, then prints
        # deterministic merge order without writing output files.
        from audio_tokenization.utils.indexed_dataset import discover_indexed_prefixes

        prefixes, missing_idx = discover_indexed_prefixes(
            input_dirs=args.input_dir,
            recursive=args.recursive,
        )
        print(f"Discovered prefixes: {len(prefixes)}")
        print(f"Output prefix: {args.output_prefix}")
        for p in prefixes:
            print(p)
        return

    # Real merge path: build one consolidated .bin/.idx dataset.
    summary = merge_indexed_dataset(
        input_dirs=args.input_dir,
        output_prefix=args.output_prefix,
        recursive=args.recursive,
        multimodal=args.multimodal,
        force=args.force,
        logger=logger,
    )
    print("Merge completed.")
    print(f"  {summary['output_bin']}")
    print(f"  {summary['output_idx']}")


if __name__ == "__main__":
    main()
