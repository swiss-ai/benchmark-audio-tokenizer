"""Coverage for the variable-length accumulate-mode interleave builder.

Mirrors the structure of ``test_build_interleaved_shift_by_one.py`` so the two
strategy implementations are exercised the same way: pure logic for sequence
shaping, mocked-cache drive of the chunk worker, and a real v2-cache
end-to-end through ``main()``.
"""
from pathlib import Path

import json
import sys

import numpy as np
import pytest

from audio_tokenization.interleave import greedy
from audio_tokenization.interleave.common import (
    list_interleave_cache_partitions,
    load_interleave_cache,
    prepare_interleave_cache_and_runs,
)
from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter


class _FakeSliceableAccessor:
    def __init__(self, rows):
        self._rows = rows

    def slice(self, start: int, length: int):
        return self._rows[start:start + length]


class _FakePreparedCache:
    def __init__(self, audio_rows, text_rows):
        self.audio = _FakeSliceableAccessor(audio_rows)
        self.text = _FakeSliceableAccessor(text_rows)


# ---------------------------------------------------------------------------
# _accumulate_sequences: pure even-clip / max_seq_len logic.
# ---------------------------------------------------------------------------


def test_accumulate_sequences_at_direction_two_clips_emits_one_sequence() -> None:
    sequences, leftovers = greedy._accumulate_sequences(
        run_audio=[["a"], ["b"]],
        run_text=[["1"], ["2"]],
        direction="AT",
        max_seq_len=100,
        bos_id="BOS",
        eos_id="EOS",
        stt_continue_id="STT",
        tts_continue_id="TTS",
    )
    assert sequences == [["BOS", "a", "STT", "2", "EOS"]]
    assert leftovers == []


def test_accumulate_sequences_ta_direction_mirrors_at() -> None:
    sequences, leftovers = greedy._accumulate_sequences(
        run_audio=[["a"], ["b"]],
        run_text=[["1"], ["2"]],
        direction="TA",
        max_seq_len=100,
        bos_id="BOS",
        eos_id="EOS",
        stt_continue_id="STT",
        tts_continue_id="TTS",
    )
    assert sequences == [["BOS", "1", "TTS", "b", "EOS"]]
    assert leftovers == []


def test_accumulate_sequences_odd_run_reverts_to_last_even_checkpoint() -> None:
    # The even-clip invariant (sequences end on opposite modality from
    # start) is what forces clip 2 to roll back to the 2-clip checkpoint.
    sequences, leftovers = greedy._accumulate_sequences(
        run_audio=[["a"], ["b"], ["c"]],
        run_text=[["1"], ["2"], ["3"]],
        direction="AT",
        max_seq_len=100,
        bos_id="BOS",
        eos_id="EOS",
        stt_continue_id="STT",
        tts_continue_id="TTS",
    )
    assert sequences == [["BOS", "a", "STT", "2", "EOS"]]
    assert leftovers == [2]


def test_accumulate_sequences_max_seq_len_cut_continues_into_new_sequence() -> None:
    sequences, leftovers = greedy._accumulate_sequences(
        run_audio=[[10], [11], [12], [13]],
        run_text=[[20], [21], [22], [23]],
        direction="AT",
        max_seq_len=8,
        bos_id=1,
        eos_id=2,
        stt_continue_id=99,
        tts_continue_id=97,
    )
    assert sequences == [
        [1, 10, 99, 21, 2],
        [1, 12, 99, 23, 2],
    ]
    assert leftovers == []


def test_accumulate_sequences_single_clip_run_routes_to_transcribe() -> None:
    sequences, leftovers = greedy._accumulate_sequences(
        run_audio=[["only"]],
        run_text=[["one"]],
        direction="AT",
        max_seq_len=100,
        bos_id="BOS",
        eos_id="EOS",
        stt_continue_id="STT",
        tts_continue_id="TTS",
    )
    assert sequences == []
    assert leftovers == [0]


# ---------------------------------------------------------------------------
# _dry_run_accumulate_lengths: must produce identical token-count totals to
# the materializing version (it's the planner's pre-pass for routing).
# ---------------------------------------------------------------------------


def _seq_total_tokens(sequences: list[list]) -> list[int]:
    return [len(s) for s in sequences]


def test_dry_run_lengths_match_accumulate_for_simple_run() -> None:
    audio = [[10, 11], [12], [13, 14], [15]]
    text = [[20], [21, 22], [23], [24, 25]]
    sequences, single = greedy._accumulate_sequences(
        audio, text, direction="AT", max_seq_len=100,
        bos_id=1, eos_id=2, stt_continue_id=99, tts_continue_id=97,
    )
    audio_lens = [len(c) for c in audio]
    text_lens = [len(c) for c in text]
    seq_lengths, single_lengths = greedy._dry_run_accumulate_lengths(
        audio_lens, text_lens, direction="AT", max_seq_len=100,
    )
    assert _seq_total_tokens(sequences) == seq_lengths
    assert single_lengths == single


def test_dry_run_lengths_match_accumulate_under_max_seq_len_cut() -> None:
    audio = [[10], [11], [12], [13]]
    text = [[20], [21], [22], [23]]
    sequences, single = greedy._accumulate_sequences(
        audio, text, direction="AT", max_seq_len=8,
        bos_id=1, eos_id=2, stt_continue_id=99, tts_continue_id=97,
    )
    audio_lens = [len(c) for c in audio]
    text_lens = [len(c) for c in text]
    seq_lengths, single_lengths = greedy._dry_run_accumulate_lengths(
        audio_lens, text_lens, direction="AT", max_seq_len=8,
    )
    assert _seq_total_tokens(sequences) == seq_lengths
    assert single_lengths == single


# ---------------------------------------------------------------------------
# _compute_per_run_stats_accumulate: vectorized fast path + slow fallback.
# ---------------------------------------------------------------------------


def test_per_run_stats_single_clip_run_routes_only_to_transcribe() -> None:
    audio_lens = np.array([3], dtype=np.int64)
    text_lens = np.array([4], dtype=np.int64)
    run_starts = np.array([0], dtype=np.int64)
    run_lengths = np.array([1], dtype=np.int64)

    il, tr = greedy._compute_per_run_stats_accumulate(
        audio_lens, text_lens, run_starts, run_lengths,
        directions=greedy.DIRECTIONS, max_seq_len=100,
    )
    assert il.tolist() == [0]
    assert tr.tolist() == [1]


def test_per_run_stats_even_multi_clip_run_uses_fast_path() -> None:
    audio_lens = np.array([3, 3], dtype=np.int64)
    text_lens = np.array([3, 3], dtype=np.int64)
    run_starts = np.array([0], dtype=np.int64)
    run_lengths = np.array([2], dtype=np.int64)

    il, tr = greedy._compute_per_run_stats_accumulate(
        audio_lens, text_lens, run_starts, run_lengths,
        directions=greedy.DIRECTIONS, max_seq_len=200,
    )
    assert il.tolist() == [len(greedy.DIRECTIONS)]
    assert tr.tolist() == [0]


def test_per_run_stats_odd_multi_clip_run_emits_one_transcribe_remainder() -> None:
    audio_lens = np.array([3, 3, 3], dtype=np.int64)
    text_lens = np.array([3, 3, 3], dtype=np.int64)
    run_starts = np.array([0], dtype=np.int64)
    run_lengths = np.array([3], dtype=np.int64)

    il, tr = greedy._compute_per_run_stats_accumulate(
        audio_lens, text_lens, run_starts, run_lengths,
        directions=greedy.DIRECTIONS, max_seq_len=200,
    )
    assert il.tolist() == [len(greedy.DIRECTIONS)]
    assert tr.tolist() == [1]


def test_per_run_stats_long_run_falls_back_to_slow_path() -> None:
    # Token costs exceed the per-run upper-bound check → fast-path skip,
    # slow-path runs per-direction simulation. Verified against a direct
    # _dry_run_accumulate_lengths call as the oracle.
    audio_lens = np.array([5, 5, 5, 5, 5], dtype=np.int64)
    text_lens = np.array([5, 5, 5, 5, 5], dtype=np.int64)
    run_starts = np.array([0], dtype=np.int64)
    run_lengths = np.array([5], dtype=np.int64)

    il, tr = greedy._compute_per_run_stats_accumulate(
        audio_lens, text_lens, run_starts, run_lengths,
        directions=greedy.DIRECTIONS, max_seq_len=20,
    )

    expected_il = 0
    expected_singles: set[int] = set()
    for d in greedy.DIRECTIONS:
        seq_lens, singles = greedy._dry_run_accumulate_lengths(
            audio_lens.tolist(), text_lens.tolist(), d, 20,
        )
        expected_il += len(seq_lens)
        expected_singles.update(singles)
    assert il.tolist() == [expected_il]
    assert tr.tolist() == [len(expected_singles)]


# ---------------------------------------------------------------------------
# _accumulate_run_chunk: drives the worker via mocked module-level globals.
# ---------------------------------------------------------------------------


def test_accumulate_run_chunk_emits_at_ta_and_transcribe(
    tmp_path: Path, reset_interleave_globals,
) -> None:
    reset_interleave_globals(greedy)
    greedy._shared_cache = _FakePreparedCache(
        audio_rows=[[10], [11], [12]],
        text_rows=[[20], [21], [22]],
    )
    greedy._shared_run_starts = np.array([0], dtype=np.int64)
    greedy._shared_run_lengths = np.array([3], dtype=np.int64)
    greedy._shared_transcribe_only_runs = set()

    result = greedy._accumulate_run_chunk(
        worker_id=0,
        run_start=0,
        run_end=1,
        all_keys=list(greedy.DIRECTIONS) + [greedy.TR_KEY],
        directions=greedy.DIRECTIONS,
        max_seq_len=100,
        bos_id=1,
        eos_id=2,
        stt_continue_id=99,
        stt_transcribe_id=98,
        tts_continue_id=97,
        dtype=np.int32,
        tmp_dir=str(tmp_path),
        seq_threshold=None,
    )

    # Each direction emits one sequence from clips 0+1; clip 2 is the
    # odd-clip remainder, routed once to transcribe (deduplicated across
    # directions).
    assert result["AT"]["seqs"] == 1
    assert result["TA"]["seqs"] == 1
    assert result[greedy.TR_KEY]["seqs"] == 1

    for key in (*greedy.DIRECTIONS, greedy.TR_KEY):
        prefix = Path(result[key]["shard_prefix"])
        assert prefix.with_suffix(".bin").exists()
        assert Path(f"{prefix}_seqlens.npy").exists()
        assert Path(f"{prefix}_docidx.npy").exists()


def test_accumulate_run_chunk_routes_by_seq_threshold(
    tmp_path: Path, reset_interleave_globals,
) -> None:
    reset_interleave_globals(greedy)
    greedy._shared_cache = _FakePreparedCache(
        audio_rows=[[10], [11]],
        text_rows=[[20], [21]],
    )
    greedy._shared_run_starts = np.array([0], dtype=np.int64)
    greedy._shared_run_lengths = np.array([2], dtype=np.int64)
    greedy._shared_transcribe_only_runs = set()

    result = greedy._accumulate_run_chunk(
        worker_id=0,
        run_start=0,
        run_end=1,
        all_keys=list(greedy.DIRECTIONS) + [greedy.TR_KEY],
        directions=greedy.DIRECTIONS,
        max_seq_len=100,
        bos_id=1,
        eos_id=2,
        stt_continue_id=99,
        stt_transcribe_id=98,
        tts_continue_id=97,
        dtype=np.int32,
        tmp_dir=str(tmp_path),
        seq_threshold=4,  # 5-token AT/TA seqs exceed the 4-token stage2 ceiling
    )

    for key in greedy.DIRECTIONS:
        assert result[f"stage2/{key}"]["seqs"] == 0
        assert result[f"lct/{key}"]["seqs"] == 1
    assert result[f"stage2/{greedy.TR_KEY}"]["seqs"] == 0
    assert result[f"lct/{greedy.TR_KEY}"]["seqs"] == 0


def test_accumulate_run_chunk_transcribe_only_runs_force_per_clip_transcribe(
    tmp_path: Path, reset_interleave_globals,
) -> None:
    reset_interleave_globals(greedy)
    greedy._shared_cache = _FakePreparedCache(
        audio_rows=[[10], [11], [12]],
        text_rows=[[20], [21], [22]],
    )
    greedy._shared_run_starts = np.array([0], dtype=np.int64)
    greedy._shared_run_lengths = np.array([3], dtype=np.int64)
    greedy._shared_transcribe_only_runs = {0}

    result = greedy._accumulate_run_chunk(
        worker_id=0,
        run_start=0,
        run_end=1,
        all_keys=list(greedy.DIRECTIONS) + [greedy.TR_KEY],
        directions=greedy.DIRECTIONS,
        max_seq_len=100,
        bos_id=1,
        eos_id=2,
        stt_continue_id=99,
        stt_transcribe_id=98,
        tts_continue_id=97,
        dtype=np.int32,
        tmp_dir=str(tmp_path),
        seq_threshold=None,
    )

    assert result["AT"]["seqs"] == 0
    assert result["TA"]["seqs"] == 0
    assert result[greedy.TR_KEY]["seqs"] == 3


def test_accumulate_run_chunk_consumes_real_v2_cache(
    tmp_path: Path, reset_interleave_globals,
) -> None:
    from audio_tokenization.tests.conftest import make_v2_cache_rows

    writer = StructuredCacheChunkWriter(str(tmp_path), rank=0, writer_state=0)
    writer.add_rows(make_v2_cache_rows(3))
    writer.finalize()

    partition_dir = list_interleave_cache_partitions(tmp_path)[0]
    df, reader = load_interleave_cache(partition_dir)
    cache, starts, lengths, _n_clips, _n_sources = prepare_interleave_cache_and_runs(df, reader)

    reset_interleave_globals(greedy)
    greedy._shared_cache = cache
    greedy._shared_run_starts = starts
    greedy._shared_run_lengths = lengths
    greedy._shared_transcribe_only_runs = set()

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = greedy._accumulate_run_chunk(
        worker_id=0,
        run_start=0,
        run_end=1,
        all_keys=list(greedy.DIRECTIONS) + [greedy.TR_KEY],
        directions=greedy.DIRECTIONS,
        max_seq_len=100,
        bos_id=1,
        eos_id=2,
        stt_continue_id=99,
        stt_transcribe_id=98,
        tts_continue_id=97,
        dtype=np.int32,
        tmp_dir=str(out_dir),
        seq_threshold=None,
    )

    assert result["AT"]["seqs"] == 1
    assert result["TA"]["seqs"] == 1
    assert result[greedy.TR_KEY]["seqs"] == 1


# ---------------------------------------------------------------------------
# Module-level imports: guard against the formerly-missing symbols that
# would NameError at runtime when main()/dry-run was invoked.
# ---------------------------------------------------------------------------


def test_detect_runs_is_imported_from_common() -> None:
    """If this fails, _detect_runs got dropped from greedy.py imports again
    and main()/dry-run will NameError at runtime.
    """
    assert hasattr(greedy, "_detect_runs"), (
        "greedy must import _detect_runs from interleave.common; without it, "
        "_dry_run_accumulate and main() raise NameError at runtime."
    )


def test_os_module_is_imported() -> None:
    """main() uses os.getpid() to scope the tmp_dir; if `import os` is
    dropped again the multi-partition build path NameErrors.
    """
    assert hasattr(greedy, "os"), (
        "greedy must `import os`; main() uses os.getpid() to build a "
        "rank-unique tmp dir name."
    )


# ---------------------------------------------------------------------------
# main(): end-to-end through the CLI surface on a real v2 cache.
# Catches the previously-latent missing imports (os, _detect_runs) and any
# regression in the merge/idx-write path.
# ---------------------------------------------------------------------------


def test_greedy_main_builds_outputs_from_v2_cache(monkeypatch, tmp_path: Path) -> None:
    from audio_tokenization.tests.conftest import make_v2_cache_rows

    writer = StructuredCacheChunkWriter(str(tmp_path), rank=0, writer_state=0)
    writer.add_rows(make_v2_cache_rows(3))
    writer.finalize()

    monkeypatch.setattr(greedy, "load_token_ids", lambda _path: (1, 2, 99, 98, 97, 256))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "greedy.py",
            "--parquet-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--tokenizer-path", "ignored",
            "--max-seq-len", "100",
            "--num-workers", "1",
        ],
    )

    greedy.main()

    out = tmp_path / "out"
    for stem in (*greedy.DIRECTIONS, "transcribe"):
        assert (out / f"{stem}.bin").exists(), f"{stem}.bin missing"
        assert (out / f"{stem}.idx").exists(), f"{stem}.idx missing"
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["mode"] == "accumulate"
    assert metadata["max_seq_len"] == 100
    assert set(metadata["outputs"].keys()) == set(greedy.DIRECTIONS) | {"transcribe"}
    # 3-clip run produces 1 AT + 1 TA + 1 transcribe leftover.
    assert metadata["outputs"]["AT"]["sequences"] == 1
    assert metadata["outputs"]["TA"]["sequences"] == 1
    assert metadata["outputs"]["transcribe"]["sequences"] == 1


def test_greedy_main_dry_run_writes_stats_without_output_bins(monkeypatch, tmp_path: Path) -> None:
    """Dry-run path exercises ``_dry_run_accumulate`` (which calls
    ``_detect_runs`` directly) without materializing tokens. Catches both
    the formerly-missing ``_detect_runs`` import and any regression in the
    plain-text stats writer.
    """
    from audio_tokenization.tests.conftest import make_v2_cache_rows

    writer = StructuredCacheChunkWriter(str(tmp_path), rank=0, writer_state=0)
    writer.add_rows(make_v2_cache_rows(2))
    writer.finalize()

    partition_dir = list_interleave_cache_partitions(tmp_path)[0]

    monkeypatch.setattr(greedy, "load_token_ids", lambda _path: (1, 2, 99, 98, 97, 256))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "greedy.py",
            "--parquet-dir", str(partition_dir),
            "--output-dir", str(tmp_path / "out"),
            "--tokenizer-path", "ignored",
            "--max-seq-len", "100",
            "--dry-run",
        ],
    )

    greedy.main()

    # Dry-run writes only the stats sidecar inside the partition dir; no
    # output bin/idx should be materialized.
    assert (partition_dir / "dry_run_stats.txt").exists()
    out = tmp_path / "out"
    assert not out.exists() or not any(out.iterdir())


def test_greedy_main_rejects_invalid_transcribe_ratio(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(greedy, "load_token_ids", lambda _path: (1, 2, 99, 98, 97, 256))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "greedy.py",
            "--parquet-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--tokenizer-path", "ignored",
            "--transcribe-ratio", "1.5",
        ],
    )
    with pytest.raises(SystemExit):
        greedy.main()
