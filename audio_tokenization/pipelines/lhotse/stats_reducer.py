"""Reduce per-rank stats into a single ``stats_summary.json``.

Each rank writes ``rank_XXXX_stats.json`` at the end of tokenization. Rank 0
(through ``run_stage``) polls until all expected rank stats are present, then
aggregates them. The stage-level ``_SUCCESS`` + ``_STAGE_MANIFEST.json`` are
written exclusively by ``run_stage`` — this module no longer publishes them.

CLI for recomputing summaries on old runs::

    python -m audio_tokenization.pipelines.lhotse.stats_reducer /path/to/output_dir
"""

from __future__ import annotations

import logging
import time
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


def zero_rank_stats(*, rank: int, world_size: int) -> dict:
    """Per-rank stats template with all aggregated keys zeroed.

    Single source of truth for the rank-stats schema: keys come from
    ``_SUM_FIELDS`` so a new aggregated key can't drift past fallback writers.
    """
    stats: dict = {source: 0 for _, source in _SUM_FIELDS}
    stats["elapsed_time"] = 0.0
    stats["rank"] = rank
    stats["world_size"] = world_size
    stats["success"] = True
    return stats


RANK_STATS_WAIT_TIMEOUT_SEC = 14_400.0  # 4h: covers an entire SLURM allocation


def wait_for_rank_stats(
    output_dir: str | Path,
    *,
    expected_ranks: int,
    timeout_sec: float = RANK_STATS_WAIT_TIMEOUT_SEC,
    poll_interval_sec: float = 2.0,
) -> list[dict]:
    """Block until every rank in ``range(expected_ranks)`` has written stats.

    Called by rank 0 (through ``run_stage(work=...)``) after it finishes its
    own slice. Non-zero ranks write their stats and exit; only rank 0 polls.

    Tracks parsed records across polls so each ``rank_XXXX_stats.json`` is
    read exactly once even on long waits. Identity (rank IDs), not count,
    is the completion contract: a stray ``rank_9999_stats.json`` doesn't
    satisfy a missing rank 1.

    Raises ``RuntimeError`` after ``timeout_sec`` listing the missing rank
    IDs. ``run_stage``'s ``work`` then aborts, so ``_SUCCESS`` is not
    published and the job exits with a clear diagnostic instead of hanging
    until the allocation runs out.
    """
    output_dir = Path(output_dir)
    expected = set(range(expected_ranks))
    parsed: dict[int, dict] = {}
    deadline = time.monotonic() + timeout_sec
    while True:
        # Probe each missing rank by exact path instead of globbing the dir.
        # On Lustre at world_size=128 over hours of polling this matters:
        # readdir cost scales with N, per-name stat is constant.
        for r in list(expected - parsed.keys()):
            path = output_dir / f"rank_{r:04d}_stats.json"
            if not path.is_file():
                continue
            for record in load_json_records([path], required_key="rank", logger=logger):
                parsed[int(record["rank"])] = record
        if expected <= parsed.keys():
            return [parsed[r] for r in sorted(expected)]
        if time.monotonic() >= deadline:
            missing = sorted(expected - parsed.keys())
            raise RuntimeError(
                f"Timed out after {timeout_sec:.0f}s waiting for rank stats at "
                f"{output_dir}; missing ranks: {missing} "
                f"(saw {sorted(parsed.keys())} of {expected_ranks}). "
                "A rank likely crashed before writing rank_XXXX_stats.json; "
                "check that rank's SLURM stderr."
            )
        time.sleep(poll_interval_sec)


def aggregate_rank_stats(
    output_dir: str | Path,
    *,
    rank_stats: list[dict] | None = None,
) -> dict:
    """Aggregate rank stats into ``stats_summary.json`` (writes the summary).

    Accepts pre-parsed *rank_stats* (e.g. from ``wait_for_rank_stats``) to
    avoid re-globbing and re-parsing the same files on Lustre right before
    publishing ``_SUCCESS``. The CLI / standalone recompute path passes
    ``None`` and falls back to ``load_rank_stats``.

    Does NOT publish ``_SUCCESS`` or ``_STAGE_MANIFEST.json`` — those are owned
    by ``run_stage``. Raises if any rank's stats record ``success != True``.
    """
    output_dir = Path(output_dir)
    if rank_stats is None:
        rank_stats = load_rank_stats(output_dir)
    aggregate = build_aggregate(rank_stats)
    atomic_write_json(output_dir / "stats_summary.json", aggregate, sort_keys=False)
    failed = [s.get("rank") for s in rank_stats if s.get("success") is not True]
    if failed:
        raise RuntimeError(
            f"Tokenize ranks reported failure: {failed}. See per-rank stats files for details."
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
