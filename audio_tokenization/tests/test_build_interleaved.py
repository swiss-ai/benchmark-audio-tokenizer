"""Tests for build_interleaved helper functions (pattern + common)."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from audio_tokenization.interleave.common import (
    find_consecutive_runs,
    _detect_runs,
    compute_ratio_adjustment,
    list_interleave_cache_partitions,
    load_parquets,
    load_interleave_cache,
    prepare_interleave_cache_and_runs,
    prepare_length_metadata,
)
from audio_tokenization.interleave.pattern import (
    build_sequence,
    derive_sub_patterns,
    group_patterns_by_size,
    _pattern_constants,
)


# ── build_sequence ──────────────────────────────────────────────────────


class TestBuildSequence:
    """Tests for build_sequence()."""

    BOS, EOS, STT_CONT, TTS_CONT = 1, 2, 99, 97

    def _build(self, audio, text, pattern):
        return build_sequence(
            audio, text, pattern,
            self.BOS, self.EOS, self.STT_CONT, self.TTS_CONT,
        )

    def test_at_pattern(self):
        # 2 positions: pos0=A, pos1=T → audio[0], stt_continue, text[1]
        seq = self._build([[10, 11], []], [[], [20, 21]], "AT")
        assert seq == [1, 10, 11, 99, 20, 21, 2]

    def test_ta_pattern(self):
        # 2 positions: pos0=T, pos1=A → text[0], tts_continue, audio[1]
        seq = self._build([[], [10, 11]], [[20, 21], []], "TA")
        assert seq == [1, 20, 21, 97, 10, 11, 2]

    def test_atat_pattern(self):
        # 4 positions: A T A T — clips 0,1,2,3
        audio = [[10, 11], [], [12, 13], []]
        text = [[], [20, 21], [], [22, 23]]
        seq = self._build(audio, text, "ATAT")
        # A0 stt_cont T1 tts_cont A2 stt_cont T3
        assert seq == [1, 10, 11, 99, 20, 21, 97, 12, 13, 99, 22, 23, 2]

    def test_tata_pattern(self):
        # 4 positions: T A T A — clips 0,1,2,3
        audio = [[], [10, 11], [], [12, 13]]
        text = [[20, 21], [], [22, 23], []]
        seq = self._build(audio, text, "TATA")
        # T0 tts_cont A1 stt_cont T2 tts_cont A3
        assert seq == [1, 20, 21, 97, 10, 11, 99, 22, 23, 97, 12, 13, 2]

    def test_aa_no_transition_token(self):
        seq = self._build([[10], [20]], [[], []], "AA")
        assert seq == [1, 10, 20, 2]

    def test_tt_no_transition_token(self):
        seq = self._build([[], []], [[10], [20]], "TT")
        assert seq == [1, 10, 20, 2]

    def test_bos_eos_always_present(self):
        seq = self._build([[]], [[]], "A")
        assert seq[0] == self.BOS
        assert seq[-1] == self.EOS

    def test_stt_count_matches_at_transitions(self):
        """Number of stt_continue tokens == number of A→T transitions."""
        for pattern, expected in [
            ("AT", 1),
            ("TA", 0),
            ("ATAT", 2),
            ("TATA", 1),
            ("AAT", 1),
            ("TAA", 0),
            ("AATT", 1),
        ]:
            n_pos = len(pattern)
            audio = [[100 + i] for i in range(n_pos)]
            text = [[200 + i] for i in range(n_pos)]
            seq = self._build(audio, text, pattern)
            actual = seq.count(self.STT_CONT)
            assert actual == expected, (
                f"Pattern {pattern}: expected {expected} stt_continue, got {actual}"
            )

    def test_tts_count_matches_ta_transitions(self):
        """Number of tts_continue tokens == number of T→A transitions."""
        for pattern, expected in [
            ("AT", 0),
            ("TA", 1),
            ("ATAT", 1),
            ("TATA", 2),
            ("AAT", 0),
            ("TAA", 1),
            ("TTAA", 1),
        ]:
            n_pos = len(pattern)
            audio = [[100 + i] for i in range(n_pos)]
            text = [[200 + i] for i in range(n_pos)]
            seq = self._build(audio, text, pattern)
            actual = seq.count(self.TTS_CONT)
            assert actual == expected, (
                f"Pattern {pattern}: expected {expected} tts_continue, got {actual}"
            )

    def test_single_a_no_transition(self):
        """Single 'A' pattern should have no transition tokens."""
        seq = self._build([[10, 11]], [[]], "A")
        assert seq == [1, 10, 11, 2]

    def test_single_t_no_transition(self):
        """Single 'T' pattern should have no transition tokens."""
        seq = self._build([[]], [[10, 11]], "T")
        assert seq == [1, 10, 11, 2]


# ── find_consecutive_runs ───────────────────────────────────────────────


class TestFindConsecutiveRuns:
    def test_docstring_example(self):
        assert find_consecutive_runs([3, 4, 5, 8, 9, 15]) == [
            [3, 4, 5],
            [8, 9],
            [15],
        ]

    def test_empty(self):
        assert find_consecutive_runs([]) == []

    def test_single(self):
        assert find_consecutive_runs([7]) == [[7]]

    def test_all_consecutive(self):
        assert find_consecutive_runs([0, 1, 2, 3]) == [[0, 1, 2, 3]]

    def test_all_disjoint(self):
        assert find_consecutive_runs([0, 5, 10]) == [[0], [5], [10]]

    def test_two_runs(self):
        assert find_consecutive_runs([1, 2, 10, 11, 12]) == [[1, 2], [10, 11, 12]]


# ── group_patterns_by_size ──────────────────────────────────────────────


class TestGroupPatternsBySize:
    def test_docstring_example(self):
        result = group_patterns_by_size(["AT", "TA", "AAT", "TTA"])
        assert result == {2: ["AT", "TA"], 3: ["AAT", "TTA"]}

    def test_single_pattern(self):
        assert group_patterns_by_size(["ATAT"]) == {4: ["ATAT"]}

    def test_empty(self):
        assert group_patterns_by_size([]) == {}

    def test_same_size(self):
        result = group_patterns_by_size(["AT", "TA"])
        assert result == {2: ["AT", "TA"]}


# ── derive_sub_patterns ────────────────────────────────────────────────


class TestDeriveSubPatterns:
    def test_docstring_example(self):
        result = derive_sub_patterns(["ATAT", "TATA"], 4)
        assert result == {3: ["ATA", "TAT"], 2: ["AT", "TA"]}

    def test_min_window_2_no_sub_patterns(self):
        result = derive_sub_patterns(["AT", "TA"], 2)
        assert result == {}

    def test_min_window_3(self):
        result = derive_sub_patterns(["ATA", "TAT"], 3)
        assert result == {2: ["AT", "TA"]}

    def test_dedup(self):
        # Both "AAT" and "AAT" truncated to 2 give "AA" — should deduplicate
        result = derive_sub_patterns(["AAT", "AAT"], 3)
        assert result == {2: ["AA"]}


# ── _pattern_constants ──────────────────────────────────────────────────


class TestPatternConstants:
    def test_atat(self):
        a_pos, t_pos, n_at, n_ta = _pattern_constants("ATAT")
        assert a_pos == [0, 2]
        assert t_pos == [1, 3]
        assert n_at == 2
        assert n_ta == 1

    def test_tata(self):
        a_pos, t_pos, n_at, n_ta = _pattern_constants("TATA")
        assert a_pos == [1, 3]
        assert t_pos == [0, 2]
        assert n_at == 1
        assert n_ta == 2

    def test_at(self):
        a_pos, t_pos, n_at, n_ta = _pattern_constants("AT")
        assert a_pos == [0]
        assert t_pos == [1]
        assert n_at == 1
        assert n_ta == 0

    def test_ta(self):
        a_pos, t_pos, n_at, n_ta = _pattern_constants("TA")
        assert a_pos == [1]
        assert t_pos == [0]
        assert n_at == 0
        assert n_ta == 1

    def test_all_audio(self):
        a_pos, t_pos, n_at, n_ta = _pattern_constants("AAA")
        assert a_pos == [0, 1, 2]
        assert t_pos == []
        assert n_at == 0
        assert n_ta == 0

    def test_all_text(self):
        a_pos, t_pos, n_at, n_ta = _pattern_constants("TTT")
        assert a_pos == []
        assert t_pos == [0, 1, 2]
        assert n_at == 0
        assert n_ta == 0


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


class TestLoadParquets:
    def test_falls_back_to_generic_parquet_names(self, tmp_path: Path):
        df = pl.DataFrame(
            {
                "source_id": ["s1", "s1"],
                "clip_num": [0, 1],
                "audio_tokens": [[1, 2], [3, 4]],
                "text_tokens": [[5], [6]],
            }
        )
        df.write_parquet(tmp_path / "part_00000.parquet")

        loaded = load_parquets(tmp_path)

        assert loaded.shape == (2, 4)
        assert loaded["source_id"].to_list() == ["s1", "s1"]
        assert loaded["clip_num"].to_list() == [0, 1]


class TestInterleaveCacheReaders:
    def test_load_interleave_cache_uses_v1_without_layout_file(self, tmp_path: Path):
        df = pl.DataFrame(
            {
                "source_id": ["s1", "s1"],
                "clip_num": [0, 1],
                "audio_tokens": [[1, 2], [3]],
                "text_tokens": [[10], [11, 12]],
            }
        )
        df.write_parquet(tmp_path / "part_00000.parquet")

        loaded_df, reader = load_interleave_cache(tmp_path)

        assert reader.__class__.__name__ == "_V1InterleaveCacheReader"
        assert loaded_df.shape == (2, 4)

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
        from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter

        writer = StructuredCacheChunkWriter(str(tmp_path), rank=0, writer_state=0)
        writer.add_rows([
            {
                "clip_id": "s1@000000",
                "source_id": "s1",
                "clip_num": 0,
                "clip_start": 0.0,
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

        with pytest.raises(RuntimeError, match="Unsupported interleave cache layout version"):
            load_interleave_cache(tmp_path)

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
