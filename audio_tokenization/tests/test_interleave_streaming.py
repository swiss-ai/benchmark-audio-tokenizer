"""Array ownership, bounded emission, and frozen materialization output parity."""

import hashlib
import json
from pathlib import Path
import weakref

import numpy as np
import pytest

from audio_tokenization.interleave import common, shift_by_one as sbo
from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter
from audio_tokenization.utils.indexed_dataset.indexed_dataset_megatron import IndexedDatasetBuilder


def test_token_run_spans_share_mapped_payload(tmp_path):
    path = tmp_path / "tokens.bin"
    np.asarray([10, 11, 12, 13, 14], dtype=np.int32).tofile(path)
    mapped = np.memmap(path, dtype=np.int32, mode="r")
    accessor = common._MemmapTokenAccessor(
        [mapped], np.zeros(3, dtype=np.int32), np.array([0, 2, 2]), np.array([2, 0, 3]),
    )
    run = accessor.slice(0, 3)[1:]

    span = run[-1]
    assert isinstance(span, np.ndarray)
    assert np.shares_memory(span, mapped)
    assert not span.flags.writeable
    assert span.tolist() == [12, 13, 14]
    assert run[0].dtype == np.int32
    assert run[0].size == 0
    assert run.to_pylist() == [[], [12, 13, 14]]


def test_worker_writes_first_sequence_before_reading_rest_of_large_run(monkeypatch, tmp_path):
    payload_reads = []

    class CountingAccessor(common._MemmapTokenAccessor):
        def get(self, idx):
            payload_reads.append(idx)
            return super().get(idx)

    count = 2000
    accessor = CountingAccessor(
        [np.arange(count, dtype=np.int32)], np.zeros(count, dtype=np.int32),
        np.arange(count), np.ones(count, dtype=np.int32),
    )
    monkeypatch.setattr(sbo, "_shared_cache", common.PreparedInterleaveCache(accessor, accessor))
    monkeypatch.setattr(sbo, "_shared_run_starts", np.array([0]))
    monkeypatch.setattr(sbo, "_shared_run_lengths", np.array([count]))
    monkeypatch.setattr(sbo, "_shared_transcribe_only_runs", set())
    original_add = IndexedDatasetBuilder.add_item
    emitted = []

    def observe_write(builder, sequence):
        if not emitted:
            assert len(payload_reads) == 2
        if len(emitted) > 1:
            assert emitted[-2]() is None
        assert isinstance(sequence, np.ndarray)
        emitted.append(weakref.ref(sequence))
        original_add(builder, sequence)

    monkeypatch.setattr(IndexedDatasetBuilder, "add_item", observe_write)
    result = sbo._shift_run_chunk(
        0, 0, 1, sbo.OFFSET_KEYS + [sbo.TR_KEY], 5,
        1, 2, 99, 98, 97, np.int32, str(tmp_path),
    )

    assert [result[key]["seqs"] for key in sbo.OFFSET_KEYS + [sbo.TR_KEY]] == [1000, 999, 1]


@pytest.mark.parametrize("token", [-1, 65536])
@pytest.mark.parametrize("kind", ["audio", "text"])
@pytest.mark.parametrize("count,transcribe_only", [(1, False), (2, False), (2, True)])
def test_worker_rejects_cache_tokens_outside_uint16(monkeypatch, tmp_path, token, kind, count, transcribe_only):
    accessors = []
    for payload_kind in ("audio", "text"):
        values = np.full(count, 10, dtype=np.int32)
        if kind == payload_kind:
            values[0 if kind == "audio" else count - 1] = token
        accessors.append(common._MemmapTokenAccessor(
            [values], np.zeros(count, dtype=np.int32), np.arange(count), np.ones(count, dtype=np.int32),
        ))
    monkeypatch.setattr(sbo, "_shared_cache", common.PreparedInterleaveCache(*accessors))
    monkeypatch.setattr(sbo, "_shared_run_starts", np.array([0]))
    monkeypatch.setattr(sbo, "_shared_run_lengths", np.array([count]))
    monkeypatch.setattr(sbo, "_shared_transcribe_only_runs", {0} if transcribe_only else set())

    with pytest.raises(OverflowError):
        sbo._shift_run_chunk(
            0, 0, 1, sbo.OFFSET_KEYS + [sbo.TR_KEY], 100,
            1, 2, 99, 98, 97, np.uint16, str(tmp_path),
        )


@pytest.mark.parametrize("dtype,low,high", [(np.uint16, 0, 65535), (np.int32, -(2**31), 2**31 - 1)])
def test_worker_preserves_output_dtype_boundaries(monkeypatch, tmp_path, dtype, low, high):
    accessor = common._MemmapTokenAccessor(
        [np.array([low, high], dtype=np.int32)], np.zeros(2, dtype=np.int32),
        np.arange(2), np.ones(2, dtype=np.int32),
    )
    monkeypatch.setattr(sbo, "_shared_cache", common.PreparedInterleaveCache(accessor, accessor))
    monkeypatch.setattr(sbo, "_shared_run_starts", np.array([0]))
    monkeypatch.setattr(sbo, "_shared_run_lengths", np.array([2]))
    monkeypatch.setattr(sbo, "_shared_transcribe_only_runs", set())
    result = sbo._shift_run_chunk(
        0, 0, 1, sbo.OFFSET_KEYS + [sbo.TR_KEY], 100,
        1, 2, 99, 98, 97, dtype, str(tmp_path),
    )

    assert np.fromfile(result["offset_0"]["shard_prefix"] + ".bin", dtype=dtype).tolist() == [1, low, 99, high, 2]
    assert np.fromfile(result["transcribe"]["shard_prefix"] + ".bin", dtype=dtype).tolist() == [1, high, 98, high, 2]


def _materialize_case(tmp_path, dtype, threshold, ratio):
    # Deliberately unsorted, with even/odd runs, a singleton, empty spans,
    # a timestamp gap and a pair larger than the sequence limit.
    specs = [
        ("odd", 0, 0.0, [10, 11], []),
        ("odd", 1, 1.0, [], [21]),
        ("odd", 2, 2.0, [12], [22, 23]),
        ("odd", 3, 3.0, [13, 14], []),
        ("odd", 4, 4.0, [15], [24]),
        ("even", 0, 0.0, list(range(30, 45)), [50]),
        ("even", 1, 1.0, [45], [51, 52]),
        ("even", 2, 2.0, [46], [53]),
        ("even", 3, 3.0, [], [54, 55]),
        ("gap", 0, 0.0, [], []),
        ("gap", 1, 1.0, [60], []),
        ("gap", 2, 20.0, [61, 62], [63]),
        ("single", 0, 0.0, [70], [71]),
    ]
    cache_dir = tmp_path / "cache"
    for rank, subset in enumerate((specs[::2], specs[1::2])):
        writer = StructuredCacheChunkWriter(
            str(cache_dir), rank=rank, writer_state=0,
            partitioning={"type": "hash", "field": "source_id", "num_buckets": 1},
        )
        writer.add_rows([
            {
                "clip_id": f"{source}@{clip_num:06d}", "source_id": source,
                "clip_num": clip_num, "clip_start": start, "clip_duration": 1.0,
                "speaker": "", "duration": 1.0, "text": "fixture",
                "audio_tokens": audio, "text_tokens": text, "dataset": "ds",
            }
            for source, clip_num, start, audio, text in reversed(subset)
        ])
        writer.finalize()
    leaf = common.list_interleave_cache_partitions(cache_dir)[0]
    df, reader = common.load_interleave_cache(leaf)
    cache, starts, lengths, _, _ = common.prepare_interleave_cache_and_runs(df, reader, max_gap_sec=5.0)
    converted = set()
    if ratio is not None:
        il_counts, tr_counts = sbo._compute_per_run_stats_shift(lengths)
        converted = common.compute_ratio_adjustment(il_counts, tr_counts, lengths, ratio)

    output = tmp_path / "output"
    shards = tmp_path / "shards"
    output.mkdir()
    shards.mkdir()
    keys = sbo.OFFSET_KEYS + [sbo.TR_KEY]
    merge_keys = keys
    if threshold is not None:
        merge_keys = [f"{bucket}/{key}" for bucket in ("stage2", "lct") for key in keys]
        for bucket in ("stage2", "lct"):
            (output / bucket).mkdir()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sbo, "_shared_cache", cache)
        patch.setattr(sbo, "_shared_run_starts", starts)
        patch.setattr(sbo, "_shared_run_lengths", lengths)
        patch.setattr(sbo, "_shared_transcribe_only_runs", converted)
        results = [
            sbo._shift_run_chunk(
                worker, begin, end, keys, 12, 1, 2, 99, 98, 97, dtype, str(shards), threshold,
            )
            for worker, (begin, end) in enumerate(common._partition_runs(lengths, 2))
        ]
    counters = common._merge_shards(results, merge_keys, output, dtype, shards)
    sbo._dry_run_shift(
        df, 12, 5.0, 1, 2, 99, 98, 97, dtype, cache_dir,
        transcribe_ratio=ratio, seq_threshold=threshold,
    )
    return {
        "run_lengths": lengths.tolist(),
        "converted": sorted(converted),
        "counters": counters,
        "dry_run_stats": (cache_dir / "dry_run_shift_stats.txt").read_text(),
        "files": {
            f"{directory.name}/{path.relative_to(directory)}": hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (shards, output)
            for path in sorted(directory.rglob("*")) if path.is_file()
        },
    }


@pytest.mark.parametrize("dtype", [np.int32, np.uint16])
@pytest.mark.parametrize("threshold,ratio", [(None, None), (9, None), (9, 0.8)])
def test_complete_worker_and_merged_index_match_frozen_baseline(tmp_path, dtype, threshold, ratio):
    baseline_path = Path(__file__).with_name("fixtures") / "interleave_materialization_baseline.json"
    baseline = json.loads(baseline_path.read_text())
    case = f"{np.dtype(dtype).name}:{threshold}:{ratio}"

    assert _materialize_case(tmp_path, dtype, threshold, ratio) == baseline["cases"][case]
