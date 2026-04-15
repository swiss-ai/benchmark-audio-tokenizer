from pathlib import Path

import json
import numpy as np
import sys

from audio_tokenization.utils.build_interleaved import shift_by_one as sbo
from audio_tokenization.utils.build_interleaved.common import (
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


def test_shift_by_one_mock_symbols_cover_multiple_interleavings() -> None:
    sequences_0, leftovers_0 = sbo._accumulate_shift_sequences(
        run_audio=[["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"]],
        run_text=[["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"]],
        offset=0,
        max_seq_len=10,
        bos_id="BOS",
        eos_id="EOS",
        stt_continue_id="STT",
        tts_continue_id="TTS",
    )
    sequences_1, leftovers_1 = sbo._accumulate_shift_sequences(
        run_audio=[["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"]],
        run_text=[["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"]],
        offset=1,
        max_seq_len=10,
        bos_id="BOS",
        eos_id="EOS",
        stt_continue_id="STT",
        tts_continue_id="TTS",
    )

    assert sequences_0 == [
        ["BOS", "a", "STT", "2", "TTS", "c", "STT", "4", "EOS"],
        ["BOS", "e", "STT", "6", "EOS"],
    ]
    assert leftovers_0 == [6]
    assert sequences_1 == [
        ["BOS", "b", "STT", "3", "TTS", "d", "STT", "5", "EOS"],
        ["BOS", "f", "STT", "7", "EOS"],
    ]
    assert leftovers_1 == []


def test_build_shift_sequence_orders_audio_then_next_text() -> None:
    seq = sbo._build_shift_sequence(
        run_audio=[[10, 11], [12], [13, 14], [15]],
        run_text=[[20], [21, 22], [23], [24, 25]],
        start=0,
        count=4,
        bos_id=1,
        eos_id=2,
        stt_continue_id=99,
        tts_continue_id=97,
    )

    assert seq == [
        1,
        10, 11,
        99, 21, 22,
        97, 13, 14,
        99, 24, 25,
        2,
    ]


def test_accumulate_shift_sequences_tracks_leftovers() -> None:
    sequences, leftovers = sbo._accumulate_shift_sequences(
        run_audio=[[10], [11], [12]],
        run_text=[[20], [21], [22]],
        offset=0,
        max_seq_len=100,
        bos_id=1,
        eos_id=2,
        stt_continue_id=99,
        tts_continue_id=97,
    )

    assert sequences == [[1, 10, 99, 21, 2]]
    assert leftovers == [2]


def test_shift_run_chunk_emits_offsets_and_transcribe(tmp_path: Path) -> None:
    sbo._shared_cache = _FakePreparedCache([[10], [11], [12]], [[20], [21], [22]])
    sbo._shared_run_starts = np.array([0], dtype=np.int64)
    sbo._shared_run_lengths = np.array([3], dtype=np.int64)
    sbo._shared_transcribe_only_runs = set()

    try:
        result = sbo._shift_run_chunk(
            worker_id=0,
            run_start=0,
            run_end=1,
            all_keys=sbo.OFFSET_KEYS + [sbo.TR_KEY],
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
    finally:
        sbo._shared_cache = None
        sbo._shared_run_starts = None
        sbo._shared_run_lengths = None
        sbo._shared_transcribe_only_runs = set()

    assert result["offset_0"]["seqs"] == 1
    assert result["offset_1"]["seqs"] == 1
    assert result["transcribe"]["seqs"] == 1
    assert result["offset_0"]["tokens"] == 5
    assert result["offset_1"]["tokens"] == 5
    assert result["transcribe"]["tokens"] == 5

    for key in sbo.OFFSET_KEYS + [sbo.TR_KEY]:
        prefix = Path(result[key]["shard_prefix"])
        assert prefix.with_suffix(".bin").exists()
        assert Path(f"{prefix}_seqlens.npy").exists()
        assert Path(f"{prefix}_docidx.npy").exists()


def test_shift_run_chunk_routes_by_seq_threshold(tmp_path: Path) -> None:
    sbo._shared_cache = _FakePreparedCache([[10], [11]], [[20], [21]])
    sbo._shared_run_starts = np.array([0], dtype=np.int64)
    sbo._shared_run_lengths = np.array([2], dtype=np.int64)
    sbo._shared_transcribe_only_runs = set()

    try:
        result = sbo._shift_run_chunk(
            worker_id=0,
            run_start=0,
            run_end=1,
            all_keys=sbo.OFFSET_KEYS + [sbo.TR_KEY],
            max_seq_len=100,
            bos_id=1,
            eos_id=2,
            stt_continue_id=99,
            stt_transcribe_id=98,
            tts_continue_id=97,
            dtype=np.int32,
            tmp_dir=str(tmp_path),
            seq_threshold=4,
        )
    finally:
        sbo._shared_cache = None
        sbo._shared_run_starts = None
        sbo._shared_run_lengths = None
        sbo._shared_transcribe_only_runs = set()

    assert result["stage2/offset_0"]["seqs"] == 0
    assert result["lct/offset_0"]["seqs"] == 1
    assert result["stage2/offset_1"]["seqs"] == 0
    assert result["lct/offset_1"]["seqs"] == 0
    assert result["stage2/transcribe"]["seqs"] == 0
    assert result["lct/transcribe"]["seqs"] == 1


def test_shift_run_chunk_consumes_real_v2_cache(tmp_path: Path) -> None:
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
            "text_tokens": [20],
            "audio_tokens": [10],
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
            "text_tokens": [21],
            "audio_tokens": [11],
            "dataset": "ds",
        },
        {
            "clip_id": "s1@000002",
            "source_id": "s1",
            "clip_num": 2,
            "clip_start": 2.0,
            "speaker": "",
            "duration": 1.0,
            "text": "c",
            "text_tokens": [22],
            "audio_tokens": [12],
            "dataset": "ds",
        },
    ])
    writer.finalize()

    partition_dir = list_interleave_cache_partitions(tmp_path)[0]
    df, reader = load_interleave_cache(partition_dir)
    cache, starts, lengths, _n_clips, _n_sources = prepare_interleave_cache_and_runs(df, reader)

    sbo._shared_cache = cache
    sbo._shared_run_starts = starts
    sbo._shared_run_lengths = lengths
    sbo._shared_transcribe_only_runs = set()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    try:
        result = sbo._shift_run_chunk(
            worker_id=0,
            run_start=0,
            run_end=1,
            all_keys=sbo.OFFSET_KEYS + [sbo.TR_KEY],
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
    finally:
        sbo._shared_cache = None
        sbo._shared_run_starts = None
        sbo._shared_run_lengths = None
        sbo._shared_transcribe_only_runs = set()

    assert result["offset_0"]["seqs"] == 1
    assert result["offset_1"]["seqs"] == 1
    assert result["transcribe"]["seqs"] == 1


def test_shift_main_builds_from_partitioned_root(monkeypatch, tmp_path: Path) -> None:
    writer = StructuredCacheChunkWriter(
        str(tmp_path),
        rank=0,
        writer_state=0,
        partitioning={"type": "field", "field": "language"},
    )
    writer.add_rows([
        {
            "clip_id": "en@000000",
            "source_id": "en",
            "clip_num": 0,
            "clip_start": 0.0,
            "speaker": "",
            "duration": 1.0,
            "text": "a",
            "text_tokens": [20],
            "audio_tokens": [10],
            "dataset": "ds",
            "_partition_value": "en",
        },
        {
            "clip_id": "en@000001",
            "source_id": "en",
            "clip_num": 1,
            "clip_start": 1.0,
            "speaker": "",
            "duration": 1.0,
            "text": "b",
            "text_tokens": [21],
            "audio_tokens": [11],
            "dataset": "ds",
            "_partition_value": "en",
        },
        {
            "clip_id": "fr@000000",
            "source_id": "fr",
            "clip_num": 0,
            "clip_start": 0.0,
            "speaker": "",
            "duration": 1.0,
            "text": "c",
            "text_tokens": [22],
            "audio_tokens": [12],
            "dataset": "ds",
            "_partition_value": "fr",
        },
        {
            "clip_id": "fr@000001",
            "source_id": "fr",
            "clip_num": 1,
            "clip_start": 1.0,
            "speaker": "",
            "duration": 1.0,
            "text": "d",
            "text_tokens": [23],
            "audio_tokens": [13],
            "dataset": "ds",
            "_partition_value": "fr",
        },
    ])
    writer.finalize()

    monkeypatch.setattr(sbo, "load_token_ids", lambda _path: (1, 2, 99, 98, 97, 256))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shift_by_one.py",
            "--parquet-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--tokenizer-path", "ignored",
            "--max-seq-len", "100",
            "--num-workers", "1",
        ],
    )

    sbo.main()

    assert (tmp_path / "out" / "offset_0.bin").exists()
    assert (tmp_path / "out" / "offset_0.idx").exists()
    assert (tmp_path / "out" / "transcribe.bin").exists()
    assert (tmp_path / "out" / "transcribe.idx").exists()
    metadata = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert metadata["partition_summary"]["num_partitions"] == 2
    assert metadata["partition_summary"]["total_clips"] == 4
