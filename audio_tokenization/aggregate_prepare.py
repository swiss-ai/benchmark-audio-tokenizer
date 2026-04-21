#!/usr/bin/env python3
"""Post-hoc stats aggregator for multi-node prepare runs.

Reads ``prepare_summary.json`` from each ``node_*/`` subdirectory under a
SHAR root, sums totals, and prints the rollup. Extracted from the family
runners (``prepare_wds_to_shar``, ``prepare_audio_dir_to_shar``) because
it is an operator tool, not a ``DatasetSpec``-driven stage — keeping it on
the family runners forced their arg schemas to treat every other input
as conditionally-required.

Usage::

    python -m audio_tokenization.aggregate_prepare --shar-root <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_tokenization.prepare.runtime import run_aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate prepare_summary.json across node_*/ dirs for a "
            "multi-node prepare run."
        ),
    )
    parser.add_argument(
        "--shar-root",
        type=Path,
        required=True,
        metavar="SHAR_ROOT",
        help="Root SHAR directory containing node_*/ subdirs (or a single "
        "node's output with prepare_summary.json directly inside).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_aggregate(args.shar_root)


if __name__ == "__main__":
    main()
