import json
import multiprocessing as mp

from audio_tokenization.contracts.artifacts import SUCCESS_MARKER_FILE
from audio_tokenization.prepare import runtime as prepare_runtime
import pytest

from audio_tokenization.pipelines.lhotse.stats_reducer import (
    aggregate_rank_stats,
    build_aggregate,
    load_rank_stats,
    wait_for_rank_stats,
    write_rank_stats,
)
from audio_tokenization.prepare.runtime import build_prepare_rollup, build_prepare_summary


class _CaptureLogger:
    def __init__(self):
        self.infos = []

    def info(self, message, *args):
        self.infos.append(message % args if args else message)


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


def test_aggregate_rank_stats_writes_summary_when_all_ranks_succeed(tmp_path):
    write_rank_stats(tmp_path, _tokenize_rank_stats(0, success=True))
    write_rank_stats(tmp_path, _tokenize_rank_stats(1, success=True))

    summary = aggregate_rank_stats(tmp_path)

    assert summary is not None
    assert summary["num_ranks"] == 2
    assert (tmp_path / "stats_summary.json").is_file()
    # The stage SUCCESS marker is owned by run_stage, not stats_reducer:
    assert not (tmp_path / "_SUCCESS").exists()


def test_aggregate_rank_stats_raises_for_failed_rank(tmp_path):
    write_rank_stats(tmp_path, _tokenize_rank_stats(0, success=True))
    write_rank_stats(tmp_path, _tokenize_rank_stats(1, success=False))

    with pytest.raises(RuntimeError, match="ranks reported failure"):
        aggregate_rank_stats(tmp_path)

    # Summary is still written so debug can read it.
    assert (tmp_path / "stats_summary.json").is_file()
    assert not (tmp_path / "_SUCCESS").exists()


def test_wait_for_rank_stats_returns_when_all_present(tmp_path):
    write_rank_stats(tmp_path, _tokenize_rank_stats(0, success=True))
    write_rank_stats(tmp_path, _tokenize_rank_stats(1, success=True))

    stats = wait_for_rank_stats(tmp_path, expected_ranks=2, poll_interval_sec=0.01)
    assert len(stats) == 2
    assert [s["rank"] for s in stats] == [0, 1]


def test_wait_for_rank_stats_raises_on_timeout_with_missing_ranks(tmp_path):
    # Only ranks 0 and 2 wrote stats; rank 1 "crashed" before writing.
    write_rank_stats(tmp_path, _tokenize_rank_stats(0, success=True))
    write_rank_stats(tmp_path, _tokenize_rank_stats(2, success=True))

    with pytest.raises(RuntimeError, match=r"missing ranks: \[1\]"):
        wait_for_rank_stats(
            tmp_path,
            expected_ranks=3,
            timeout_sec=0.1,
            poll_interval_sec=0.02,
        )


def test_wait_for_rank_stats_requires_exact_rank_ids(tmp_path):
    # Count is not enough: rank 99 is not a substitute for missing rank 1.
    write_rank_stats(tmp_path, _tokenize_rank_stats(0, success=True))
    write_rank_stats(tmp_path, _tokenize_rank_stats(2, success=True))
    write_rank_stats(tmp_path, _tokenize_rank_stats(99, success=True))

    with pytest.raises(RuntimeError, match=r"missing ranks: \[1\]"):
        wait_for_rank_stats(
            tmp_path,
            expected_ranks=3,
            timeout_sec=0.1,
            poll_interval_sec=0.02,
        )


def test_prepare_summary_counts_current_worker_results():
    results = [
        {
            "worker_id": 0,
            "written": 5,
            "skipped": 1,
            "errors": 0,
            "total_duration_sec": 10.0,
            "reason_counts": {"vad": 1},
            "worker_stats": {"runtime_counts": {"decoded": 5}},
        },
        {
            "worker_id": 1,
            "written": 7,
            "skipped": 2,
            "errors": 1,
            "total_duration_sec": 20.0,
            "worker_stats": {
                "runtime_counts": {"decoded": 7},
                "reason_counts": {"quiet": 2},
            },
        },
    ]

    summary = build_prepare_summary(results, elapsed_sec=3.0, num_workers=2)

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


def test_prepare_worker_progress_logs_when_written_crosses_threshold(monkeypatch):
    logger = _CaptureLogger()
    monkeypatch.setattr(prepare_runtime.time, "time", lambda: 20.0)

    next_log_at = prepare_runtime.maybe_log_worker_progress(
        logger=logger,
        worker_id=3,
        written=1007,
        skipped=2,
        errors=1,
        t0=10.0,
        next_log_at=1000,
    )

    assert next_log_at == 2000
    assert logger.infos == [
        "Worker 3: 1007 written, 2 skipped, 1 errors (100.7 samples/s)"
    ]


def test_prepare_pool_reports_workers_as_they_finish(monkeypatch, tmp_path):
    class _FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, *_args, **_kwargs):
            raise AssertionError("run_pool_and_finalize must not block on pool.map")

        def imap_unordered(self, fn, worker_args):
            for arg in reversed(worker_args):
                yield fn(arg)

    class _FakeContext:
        def Pool(self, processes):
            return _FakePool(processes)

    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork"])
    monkeypatch.setattr(mp, "get_context", lambda _method: _FakeContext())
    monkeypatch.setattr(
        prepare_runtime, "build_shar_index_for_worker_dirs", lambda *_args, **_kwargs: None
    )

    import audio_tokenization.prepare.validate_shar as validate_shar

    monkeypatch.setattr(validate_shar, "validate_shar_directory", lambda _path: {})

    logger = _CaptureLogger()
    monkeypatch.setattr(prepare_runtime, "logger", logger)

    def worker(worker_id):
        return {
            "worker_id": worker_id,
            "written": worker_id + 1,
            "skipped": 0,
            "errors": 0,
            "elapsed": 1.0,
            "total_duration_sec": 10.0,
            "worker_stats": {"runtime_counts": {}},
        }

    results = prepare_runtime.run_pool_and_finalize(
        worker,
        [0, 1],
        tmp_path,
        num_workers=2,
        mp_start_method="fork",
    )

    assert [result["worker_id"] for result in results] == [1, 0]
    assert "Worker 1 finished (1/2)" in logger.infos
    assert "Worker 0 finished (2/2)" in logger.infos


def test_prepare_pool_indexes_actual_worker_dirs_for_sparse_ids(monkeypatch, tmp_path):
    """Empty assignment buckets should not make finalize expect missing workers."""

    class _FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, fn, worker_args):
            for arg in worker_args:
                yield fn(arg)

    class _FakeContext:
        def Pool(self, processes):
            return _FakePool(processes)

    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork"])
    monkeypatch.setattr(mp, "get_context", lambda _method: _FakeContext())
    indexed_dirs = []

    def _capture_index(_shar_dir, worker_dirs, **_kwargs):
        indexed_dirs.extend(path.name for path in worker_dirs)

    monkeypatch.setattr(prepare_runtime, "build_shar_index_for_worker_dirs", _capture_index)

    import audio_tokenization.prepare.validate_shar as validate_shar

    monkeypatch.setattr(validate_shar, "validate_shar_directory", lambda _path: {})

    def worker(worker_id):
        worker_dir = tmp_path / f"worker_{worker_id:02d}"
        worker_dir.mkdir()
        (worker_dir / SUCCESS_MARKER_FILE).write_text("ok\n")
        return {
            "worker_id": worker_id,
            "written": 1,
            "skipped": 0,
            "errors": 0,
            "elapsed": 1.0,
            "total_duration_sec": 10.0,
            "worker_stats": {"runtime_counts": {}},
        }

    prepare_runtime.run_pool_and_finalize(
        worker,
        [0, 2],
        tmp_path,
        num_workers=3,
        mp_start_method="fork",
    )

    assert indexed_dirs == ["worker_00", "worker_02"]
