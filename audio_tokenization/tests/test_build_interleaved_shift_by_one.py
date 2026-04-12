from pathlib import Path

import numpy as np
import pyarrow as pa

from audio_tokenization.utils.build_interleaved import shift_by_one as sbo


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
    sbo._shared_audio_arrow = pa.array([[10], [11], [12]])
    sbo._shared_text_arrow = pa.array([[20], [21], [22]])
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
        sbo._shared_audio_arrow = None
        sbo._shared_text_arrow = None
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
    sbo._shared_audio_arrow = pa.array([[10], [11]])
    sbo._shared_text_arrow = pa.array([[20], [21]])
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
        sbo._shared_audio_arrow = None
        sbo._shared_text_arrow = None
        sbo._shared_run_starts = None
        sbo._shared_run_lengths = None
        sbo._shared_transcribe_only_runs = set()

    assert result["stage2/offset_0"]["seqs"] == 0
    assert result["lct/offset_0"]["seqs"] == 1
    assert result["stage2/offset_1"]["seqs"] == 0
    assert result["lct/offset_1"]["seqs"] == 0
    assert result["stage2/transcribe"]["seqs"] == 0
    assert result["lct/transcribe"]["seqs"] == 1
