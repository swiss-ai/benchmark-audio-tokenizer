"""Reduce per-rank stats into a single ``stats_summary.json``.

Each rank writes ``rank_XXXX_stats.json`` at the end of tokenization.
The last rank to finish aggregates all per-rank files into one summary.

CLI for recomputing summaries on old runs::

    python -m audio_tokenization.pipelines.lhotse.stats_reducer /path/to/output_dir
"""

from __future__ import annotations

import logging
from pathlib import Path

from audio_tokenization.utils.io import atomic_write_json
from audio_tokenization.utils.stats import load_json_records, max_field, sum_mapped_fields

logger = logging.getLogger(__name__)

_SUM_FIELDS = (
    ("samples_processed", "samples_processed"),
    ("audio_tokens", "tokens_generated"),
    ("text_tokens", "text_tokens_generated"),
    ("errors", "errors"),
    ("samples_skipped", "samples_skipped"),
    ("rms_skipped", "rms_skipped"),
    ("no_text_skipped", "no_text_skipped"),
    ("chunks_written", "chunks_written"),
)


def load_rank_stats(output_dir: Path) -> list[dict]:
    """Load all ``rank_XXXX_stats.json`` files, sorted by rank."""
    return load_json_records(
        sorted(output_dir.glob("rank_*_stats.json")),
        required_key="rank",
        logger=logger,
    )


def build_aggregate(rank_stats: list[dict]) -> dict:
    """Aggregate per-rank stats in a single pass."""
    agg = sum_mapped_fields(rank_stats, _SUM_FIELDS)
    agg["num_ranks"] = len(rank_stats)
    max_elapsed = max_field(rank_stats, "elapsed_time")

    agg["total_tokens"] = agg["audio_tokens"] + agg["text_tokens"]
    agg["max_elapsed_s"] = max_elapsed
    if max_elapsed > 0:
        agg["audio_tokens_per_second"] = agg["audio_tokens"] / max_elapsed
        agg["samples_per_second"] = agg["samples_processed"] / max_elapsed
    agg["per_rank"] = rank_stats
    return agg


def write_rank_stats(output_dir: str | Path, result: dict) -> Path:
    """Write a single rank's stats to ``rank_XXXX_stats.json``."""
    output_dir = Path(output_dir)
    rank = result.get("rank", 0)
    stats_path = output_dir / f"rank_{rank:04d}_stats.json"
    atomic_write_json(stats_path, result, sort_keys=False)
    return stats_path


def maybe_publish_terminal_artifacts(
    output_dir: str | Path,
    expected_ranks: int,
) -> dict | None:
    """Convoy-leader publish: write ``stats_summary.json`` and, if every rank
    reported ``success=True``, the partition ``_SUCCESS`` marker.

    Each rank calls this after ``write_rank_stats``. The single rank whose
    call observes that all ranks have reported wins the race and publishes
    both artifacts. ``_SUCCESS`` is never written when any rank's stats
    record ``success=False``; the failed rank's process re-raises so the
    job exit code signals failure.
    """
    output_dir = Path(output_dir)
    rank_stats = load_rank_stats(output_dir)
    if len(rank_stats) < expected_ranks:
        return None
    aggregate = build_aggregate(rank_stats)
    atomic_write_json(output_dir / "stats_summary.json", aggregate, sort_keys=False)
    if all(s.get("success") is True for s in aggregate["per_rank"]):
        from audio_tokenization.prepare.runtime import mark_partition_success

        mark_partition_success(output_dir)
    else:
        failed = [s.get("rank") for s in aggregate["per_rank"] if s.get("success") is not True]
        logger.warning(
            "Skipping _SUCCESS publication; ranks reported failure: %s",
            failed,
        )
    return aggregate


def main(argv: list[str] | None = None) -> int:
    """Recompute stats_summary.json from per-rank files."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Tokenization output directory")
    parser.add_argument("--expected-ranks", type=int, default=None)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    rank_stats = load_rank_stats(output_dir)
    if not rank_stats:
        print(f"No rank stats found in {output_dir}")
        return 1

    if args.expected_ranks and len(rank_stats) < args.expected_ranks:
        print(f"Found {len(rank_stats)} ranks, expected {args.expected_ranks}")
        return 1

    aggregate = build_aggregate(rank_stats)
    atomic_write_json(output_dir / "stats_summary.json", aggregate, sort_keys=False)

    print(f"  Ranks: {aggregate['num_ranks']}")
    print(f"  Samples: {aggregate['samples_processed']:,}")
    print(f"  Audio tokens: {aggregate['audio_tokens']:,}")
    print(f"  Text tokens: {aggregate['text_tokens']:,}")
    print(f"  Total tokens: {aggregate['total_tokens']:,}")
    print(f"  Errors: {aggregate['errors']}")
    if aggregate.get("audio_tokens_per_second"):
        print(f"  Throughput: {aggregate['audio_tokens_per_second']:,.0f} audio tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
