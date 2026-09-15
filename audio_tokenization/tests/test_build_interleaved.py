"""Tests for interleave/common.py helper functions."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from audio_tokenization.interleave.common import (
    _detect_runs,
    compute_ratio_adjustment,
    list_interleave_cache_partitions,
    load_interleave_cache,
    prepare_interleave_cache_and_runs,
    prepare_length_metadata,
)


# ── _detect_runs ────────────────────────────────────────────────────────


class TestDetectRuns:
    def test_single_source_consecutive(self):
        df = pl.DataFrame(
            {
                "source_id": ["s1"] * 4,
                "clip_num": [0, 1, 2, 3],
            }
        )
        _, starts, lengths = _detect_runs(df)
        assert list(starts) == [0]
        assert list(lengths) == [4]

    def test_single_source_with_gap(self):
        df = pl.DataFrame(
            {
                "source_id": ["s1"] * 4,
                "clip_num": [0, 1, 5, 6],
            }
        )
        _, starts, lengths = _detect_runs(df)
        assert list(lengths) == [2, 2]

    def test_two_sources(self):
        df = pl.DataFrame(
            {
                "source_id": ["s1", "s1", "s2", "s2"],
                "clip_num": [0, 1, 0, 1],
            }
        )
        _, starts, lengths = _detect_runs(df)
        assert list(lengths) == [2, 2]

    def test_single_row(self):
        df = pl.DataFrame(
            {
                "source_id": ["s1"],
                "clip_num": [0],
            }
        )
        _, starts, lengths = _detect_runs(df)
        assert list(starts) == [0]
        assert list(lengths) == [1]

    def test_unsorted_input_gets_sorted(self):
        df = pl.DataFrame(
            {
                "source_id": ["s2", "s1", "s1", "s2"],
                "clip_num": [0, 1, 0, 1],
            }
        )
        sorted_df, starts, lengths = _detect_runs(df)
        # After sorting: s1/0, s1/1, s2/0, s2/1
        assert list(sorted_df["source_id"]) == ["s1", "s1", "s2", "s2"]
        assert list(sorted_df["clip_num"]) == [0, 1, 0, 1]
        assert list(lengths) == [2, 2]

    def test_lengths_sum_equals_nrows(self):
        df = pl.DataFrame(
            {
                "source_id": ["a", "a", "b", "b", "b", "a"],
                "clip_num": [0, 1, 0, 1, 5, 10],
            }
        )
        _, starts, lengths = _detect_runs(df)
        assert int(lengths.sum()) == len(df)


class TestInterleaveCacheReaders:
    def test_load_interleave_cache_rejects_unversioned_legacy_cache(self, tmp_path: Path):
        df = pl.DataFrame(
            {
                "source_id": ["s1", "s1"],
                "clip_num": [0, 1],
                "audio_tokens": [[1, 2], [3]],
                "text_tokens": [[10], [11, 12]],
            }
        )
        df.write_parquet(tmp_path / "part_00000.parquet")

        with pytest.raises(RuntimeError, match="missing _CACHE_LAYOUT.json"):
            load_interleave_cache(tmp_path)

        with pytest.raises(RuntimeError, match="missing _CACHE_LAYOUT.json"):
            list_interleave_cache_partitions(tmp_path)

    def test_prepare_length_metadata_accepts_v2_length_columns(self):
        df = pl.DataFrame(
            {
                "source_id": ["s1", "s1"],
                "clip_num": [0, 1],
                "audio_token_length": [2, 3],
                "text_token_length": [4, 5],
            }
        )

        lengths_df = prepare_length_metadata(df)

        assert lengths_df.columns == ["source_id", "clip_num", "_alen", "_tlen"]
        assert lengths_df["_alen"].to_list() == [2, 3]
        assert lengths_df["_tlen"].to_list() == [4, 5]

    def test_load_interleave_cache_reads_v2_metadata_and_payload(self, tmp_path: Path):
        from audio_tokenization.pipelines.shard_io import (
            INTERLEAVE_CACHE_SCHEMA_VERSION,
            StructuredCacheChunkWriter,
        )

        writer = StructuredCacheChunkWriter(str(tmp_path), rank=0, writer_state=0)
        writer.add_rows([
            {
                "clip_id": "s1@000000",
                "source_id": "s1",
                "clip_num": 0,
                "clip_start": 0.0,
                "clip_duration": None,
                "speaker": "",
                "duration": 1.0,
                "text": "a",
                "text_tokens": [10, 11],
                "audio_tokens": [1, 2, 3],
                "dataset": "ds",
            },
            {
                "clip_id": "s1@000001",
                "source_id": "s1",
                "clip_num": 1,
                "clip_start": 1.0,
                "clip_duration": None,
                "speaker": "",
                "duration": 1.0,
                "text": "b",
                "text_tokens": [12],
                "audio_tokens": [4, 5],
                "dataset": "ds",
            },
        ])
        writer.finalize()

        partition_dir = list_interleave_cache_partitions(tmp_path)[0]
        df, reader = load_interleave_cache(partition_dir)
        cache, starts, lengths, n_clips, n_sources = prepare_interleave_cache_and_runs(df, reader)

        assert reader.__class__.__name__ == "_V2InterleaveCacheReader"
        assert "clip_start" in df.columns
        assert "clip_duration" in df.columns
        assert json.loads((tmp_path / "_CACHE_LAYOUT.json").read_text())["schema_version"] == (
            INTERLEAVE_CACHE_SCHEMA_VERSION
        )
        assert n_clips == 2
        assert n_sources == 1
        assert starts.tolist() == [0]
        assert lengths.tolist() == [2]
        assert cache.audio_lengths.tolist() == [3, 2]
        assert cache.text_lengths.tolist() == [2, 1]
        assert cache.audio.slice(0, 2).to_pylist() == [[1, 2, 3], [4, 5]]
        assert cache.text.slice(0, 2).to_pylist() == [[10, 11], [12]]

    def test_load_interleave_cache_raises_on_unknown_layout_version(self, tmp_path: Path):
        (tmp_path / "_CACHE_LAYOUT.json").write_text(json.dumps({"version": "v3"}))
        (tmp_path / "rank_0000").mkdir()

        assert list_interleave_cache_partitions(tmp_path) == [tmp_path]

        with pytest.raises(RuntimeError, match="Unsupported interleave cache layout version"):
            load_interleave_cache(tmp_path)

    def test_load_interleave_cache_rejects_v2_layout_without_schema_version(self, tmp_path: Path):
        from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter

        writer = StructuredCacheChunkWriter(str(tmp_path), rank=0, writer_state=0)
        writer.add_rows([
            {
                "clip_id": "s1@000000",
                "source_id": "s1",
                "clip_num": 0,
                "clip_start": 0.0,
                "clip_duration": None,
                "speaker": "",
                "duration": 1.0,
                "text": "a",
                "text_tokens": [10],
                "audio_tokens": [1, 2, 3],
                "dataset": "ds",
            }
        ])
        writer.finalize()

        partition_dir = list_interleave_cache_partitions(tmp_path)[0]
        layout_path = partition_dir / "_CACHE_LAYOUT.json"
        layout = json.loads(layout_path.read_text())
        layout.pop("schema_version")
        layout_path.write_text(json.dumps(layout))

        with pytest.raises(RuntimeError, match="schema version"):
            load_interleave_cache(partition_dir)

    def test_v2_reader_raises_when_fd_budget_is_too_low(self, tmp_path: Path, monkeypatch):
        from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter
        import audio_tokenization.interleave.common as bic

        writer = StructuredCacheChunkWriter(str(tmp_path), rank=0, writer_state=0)
        writer.add_rows([
            {
                "clip_id": "s1@000000",
                "source_id": "s1",
                "clip_num": 0,
                "clip_start": 0.0,
                "clip_duration": None,
                "speaker": "",
                "duration": 1.0,
                "text": "a",
                "text_tokens": [10],
                "audio_tokens": [1, 2, 3],
                "dataset": "ds",
            }
        ])
        writer.finalize()

        partition_dir = list_interleave_cache_partitions(tmp_path)[0]
        df, reader = load_interleave_cache(partition_dir)
        monkeypatch.setattr(bic.resource, "getrlimit", lambda *_args: (10, 10))
        monkeypatch.setattr(
            bic.os,
            "listdir",
            lambda _path: ["fd0", "fd1", "fd2", "fd3", "fd4", "fd5", "fd6", "fd7", "fd8"],
        )
        with pytest.raises(RuntimeError, match="file descriptor budget"):
            prepare_interleave_cache_and_runs(df, reader)


# ── compute_ratio_adjustment ─────────────────────────────────────────


class TestComputeRatioAdjustment:
    """Tests for compute_ratio_adjustment()."""

    def test_already_above_target(self):
        """If natural ratio >= target, return empty set."""
        # 10 runs: 5 single-clip (transcribe), 5 multi-clip (interleaved)
        il_per_run = np.array([2, 2, 2, 2, 2, 0, 0, 0, 0, 0])
        tr_per_run = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        run_lengths = np.array([4, 4, 4, 4, 4, 1, 1, 1, 1, 1])
        # natural: il=10, tr=5, ratio=5/15=0.333
        result = compute_ratio_adjustment(il_per_run, tr_per_run, run_lengths, 0.3)
        assert result == set()

    def test_below_target_converts_runs(self):
        """If natural ratio < target, some runs should be converted."""
        # 10 runs: 9 multi-clip (producing 2 interleaved each), 1 single-clip
        il_per_run = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 0])
        tr_per_run = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        run_lengths = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 1])
        # natural: il=18, tr=1, ratio=1/19=0.053
        result = compute_ratio_adjustment(il_per_run, tr_per_run, run_lengths, 0.3, seed=42)
        assert len(result) > 0
        # All converted runs must be multi-clip (length >= 2)
        for r in result:
            assert run_lengths[r] >= 2

    def test_converts_enough_to_meet_target(self):
        """Converted runs should bring ratio to at least the target."""
        il_per_run = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 0])
        tr_per_run = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        run_lengths = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 1])
        target = 0.3
        result = compute_ratio_adjustment(il_per_run, tr_per_run, run_lengths, target, seed=42)

        # Recompute the adjusted ratio
        il = int(il_per_run.sum())
        tr = int(tr_per_run.sum())
        for r in result:
            il -= int(il_per_run[r])
            tr += int(run_lengths[r]) - int(tr_per_run[r])
        total = il + tr
        assert total > 0
        assert tr / total >= target

    def test_empty_input(self):
        """Empty arrays should return empty set."""
        result = compute_ratio_adjustment(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            0.1,
        )
        assert result == set()

    def test_all_single_clip(self):
        """All single-clip runs → ratio is 1.0, no conversion needed."""
        il_per_run = np.array([0, 0, 0])
        tr_per_run = np.array([1, 1, 1])
        run_lengths = np.array([1, 1, 1])
        result = compute_ratio_adjustment(il_per_run, tr_per_run, run_lengths, 0.5)
        assert result == set()

    def test_deterministic_with_seed(self):
        """Same seed should produce same result."""
        il_per_run = np.array([2, 2, 2, 2, 2, 0])
        tr_per_run = np.array([0, 0, 0, 0, 0, 1])
        run_lengths = np.array([4, 4, 4, 4, 4, 1])
        r1 = compute_ratio_adjustment(il_per_run, tr_per_run, run_lengths, 0.3, seed=123)
        r2 = compute_ratio_adjustment(il_per_run, tr_per_run, run_lengths, 0.3, seed=123)
        assert r1 == r2

    def test_does_not_convert_single_clip_runs(self):
        """Only multi-clip runs with interleaved seqs should be candidates."""
        # Mix of single-clip and multi-clip
        il_per_run = np.array([2, 0, 2, 0])
        tr_per_run = np.array([0, 1, 0, 1])
        run_lengths = np.array([4, 1, 4, 1])
        result = compute_ratio_adjustment(il_per_run, tr_per_run, run_lengths, 0.5, seed=42)
        for r in result:
            assert run_lengths[r] >= 2
