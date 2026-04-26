import json

from audio_tokenization.pipelines.lhotse.stats_reducer import (
    build_aggregate,
    load_rank_stats,
    maybe_publish_terminal_artifacts,
    write_rank_stats,
)
from audio_tokenization.prepare.runtime import build_prepare_rollup, build_prepare_summary


def _tokenize_rank_stats(rank: int, *, success: bool) -> dict:
    return {
        "rank": rank,
        "success": success,
        "samples_processed": 1,
        "tokens_generated": 10,
        "text_tokens_generated": 0,
        "errors": 0,
        "samples_skipped": 0,
        "rms_skipped": 0,
        "no_text_skipped": 0,
        "chunks_written": 1,
        "elapsed_time": 1.0,
    }


def test_tokenize_stats_reducer_uses_shared_json_loader(tmp_path):
    (tmp_path / "rank_0001_stats.json").write_text(
        json.dumps(
            {
                "rank": 1,
                "samples_processed": 2,
                "tokens_generated": 11,
                "text_tokens_generated": 3,
                "errors": 1,
                "elapsed_time": 4.0,
                "chunks_written": 2,
            }
        )
    )
    (tmp_path / "rank_bad_stats.json").write_text("{not json")
    (tmp_path / "rank_missing_stats.json").write_text(json.dumps({"samples_processed": 9}))

    aggregate = build_aggregate(load_rank_stats(tmp_path))

    assert aggregate["num_ranks"] == 1
    assert aggregate["samples_processed"] == 2
    assert aggregate["audio_tokens"] == 11
    assert aggregate["text_tokens"] == 3
    assert aggregate["total_tokens"] == 14
    assert aggregate["errors"] == 1
    assert aggregate["chunks_written"] == 2
    assert aggregate["max_elapsed_s"] == 4.0


def test_terminal_artifacts_publish_success_only_after_all_ranks_success(tmp_path):
    write_rank_stats(tmp_path, _tokenize_rank_stats(0, success=True))
    assert maybe_publish_terminal_artifacts(tmp_path, expected_ranks=2) is None
    assert not (tmp_path / "_SUCCESS").exists()

    write_rank_stats(tmp_path, _tokenize_rank_stats(1, success=True))
    summary = maybe_publish_terminal_artifacts(tmp_path, expected_ranks=2)

    assert summary is not None
    assert (tmp_path / "stats_summary.json").is_file()
    assert (tmp_path / "_SUCCESS").is_file()


def test_terminal_artifacts_do_not_publish_success_for_failed_rank(tmp_path):
    write_rank_stats(tmp_path, _tokenize_rank_stats(0, success=True))
    write_rank_stats(tmp_path, _tokenize_rank_stats(1, success=False))

    summary = maybe_publish_terminal_artifacts(tmp_path, expected_ranks=2)

    assert summary is not None
    assert (tmp_path / "stats_summary.json").is_file()
    assert not (tmp_path / "_SUCCESS").exists()


def test_prepare_summary_counts_reused_worker_from_persisted_stats():
    results = [
        {
            "worker_id": 0,
            "written": 5,
            "skipped": 1,
            "errors": 0,
            "total_duration_sec": 10.0,
            "reused": False,
            "reason_counts": {"vad": 1},
            "worker_stats": {"runtime_counts": {"decoded": 5}},
        },
        {
            "worker_id": 1,
            "written": -1,
            "skipped": 0,
            "errors": 0,
            "total_duration_sec": 0.0,
            "reused": True,
            "worker_stats": {
                "written": 7,
                "skipped": 2,
                "errors": 1,
                "total_duration_sec": 20.0,
                "runtime_counts": {"decoded": 7},
                "reason_counts": {"quiet": 2},
            },
        },
    ]

    summary = build_prepare_summary(results, elapsed_sec=3.0, num_workers=2)

    assert summary["workers_reused"] == 1
    assert summary["total_written"] == 12
    assert summary["total_skipped"] == 3
    assert summary["total_errors"] == 1
    assert summary["total_duration_sec"] == 30.0
    assert summary["runtime_counts"] == {"decoded": 12}
    assert summary["reason_counts"] == {"vad": 1, "quiet": 2}


def test_prepare_rollup_sums_node_summaries():
    rollup = build_prepare_rollup(
        [
            {
                "total_written": 5,
                "total_skipped": 1,
                "total_errors": 0,
                "total_duration_sec": 10.0,
                "elapsed_sec": 4.0,
                "runtime_counts": {"decoded": 5},
            },
            {
                "total_written": 7,
                "total_skipped": 2,
                "total_errors": 1,
                "total_duration_sec": 20.0,
                "elapsed_sec": 3.0,
                "reason_counts": {"quiet": 2},
            },
        ]
    )

    assert rollup["num_nodes"] == 2
    assert rollup["total_written"] == 12
    assert rollup["total_skipped"] == 3
    assert rollup["total_errors"] == 1
    assert rollup["total_duration_sec"] == 30.0
    assert rollup["max_elapsed_sec"] == 4.0
    assert rollup["runtime_counts"] == {"decoded": 5}
    assert rollup["reason_counts"] == {"quiet": 2}
