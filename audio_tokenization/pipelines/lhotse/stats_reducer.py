"""Reduce per-rank stats into a single ``stats_summary.json``.

Each rank writes ``rank_XXXX_stats.json`` at the end of tokenization.
The last rank to finish aggregates all per-rank files into one summary.

CLI for recomputing summaries on old runs::

    python -m audio_tokenization.pipelines.lhotse.stats_reducer /path/to/output_dir
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

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


def _atomic_json_write(path: Path, data: dict) -> None:
    """Write JSON atomically via tmp + rename."""
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str))
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def load_rank_stats(output_dir: Path) -> list[dict]:
    """Load all ``rank_XXXX_stats.json`` files, sorted by rank."""
    stats = []
    for f in sorted(output_dir.glob("rank_*_stats.json")):
        try:
            record = json.loads(f.read_text())
            if "rank" in record:
                stats.append(record)
        except Exception:
            logger.debug("Failed to read %s", f, exc_info=True)
    return stats


def build_aggregate(rank_stats: list[dict]) -> dict:
    """Aggregate per-rank stats in a single pass."""
    agg = {out_key: 0 for out_key, _ in _SUM_FIELDS}
    agg["num_ranks"] = len(rank_stats)
    max_elapsed = 0.0

    for s in rank_stats:
        for out_key, src_key in _SUM_FIELDS:
            agg[out_key] += s.get(src_key, 0)
        max_elapsed = max(max_elapsed, s.get("elapsed_time", 0))

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
    _atomic_json_write(stats_path, result)
    return stats_path


def maybe_write_stats_summary(
    output_dir: str | Path,
    expected_ranks: int,
) -> dict | None:
    """Write summary if all ranks have reported. Returns None if not ready."""
    output_dir = Path(output_dir)
    rank_stats = load_rank_stats(output_dir)
    if len(rank_stats) < expected_ranks:
        return None
    aggregate = build_aggregate(rank_stats)
    _atomic_json_write(output_dir / "stats_summary.json", aggregate)
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
    _atomic_json_write(output_dir / "stats_summary.json", aggregate)

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
