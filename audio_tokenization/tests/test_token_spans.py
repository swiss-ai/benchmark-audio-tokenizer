"""Reader regressions for malformed int32 spans and metadata-only packing."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from audio_tokenization.token_cache import AudioTokenCacheWriter, load_audio_token_cache


def _replace_index_value(index_path: Path, column: str, value, *, arrow_type=None) -> None:
    table = pq.read_table(index_path)
    field = table.schema.field(column) if arrow_type is None else pa.field(column, arrow_type)
    table = table.set_column(
        table.schema.get_field_index(column), field,
        pa.array([value], type=field.type),
    )
    pq.write_table(table, index_path)


@pytest.mark.parametrize(
    "column,value",
    [
        ("token_offset", 1),
        ("token_offset", -4),
        ("token_count", -1),
        ("token_count", 0),
        ("token_offset", None),
        ("token_count", None),
        ("token_offset", 2**63 - 4),
    ],
)
def test_sft_reader_rejects_invalid_span_at_load(tmp_path, column, value):
    writer = AudioTokenCacheWriter(tmp_path, rank=0)
    writer.add(audio_id="a", tokens=[101, 102], duration_sec=1.0)
    writer.finalize()
    index_path = next(tmp_path.glob("rank_*/audio_index.*.parquet"))
    _replace_index_value(index_path, "token_count", 1)
    _replace_index_value(index_path, column, value)

    with pytest.raises(ValueError):
        load_audio_token_cache(tmp_path)


def test_sft_reader_rejects_non_int32_payload_size_at_load(tmp_path):
    writer = AudioTokenCacheWriter(tmp_path, rank=0)
    writer.add(audio_id="a", tokens=[101, 102], duration_sec=1.0)
    writer.finalize()
    with next(tmp_path.glob("rank_*/audio_tokens.*.bin")).open("ab") as stream:
        stream.write(b"x")

    with pytest.raises(ValueError, match="align"):
        load_audio_token_cache(tmp_path)


def test_audio_cache_abort_discards_pending_chunk_and_preserves_committed_payload(tmp_path):
    writer = AudioTokenCacheWriter(tmp_path, rank=0)
    writer.add(audio_id="committed", tokens=[101, 102], duration_sec=1.0)
    writer.finalize()
    committed = {path: path.read_bytes() for path in (tmp_path / "rank_0000").iterdir()}
    writer.add(audio_id="pending", tokens=[201], duration_sec=1.0)
    pending_handle = writer._fh

    writer.abort()
    writer.abort()
    writer.finalize()

    assert pending_handle.closed
    assert not list(tmp_path.glob("rank_*/*.tmp"))
    assert {path: path.read_bytes() for path in (tmp_path / "rank_0000").iterdir()} == committed
    cache = load_audio_token_cache(tmp_path)
    assert cache.audio_ids == {"committed"}
    assert cache.read("committed").tolist() == [101, 102]


def _write_interleave_fixture(tmp_path, *, empty_text=False):
    from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter

    writer = StructuredCacheChunkWriter(str(tmp_path), rank=0, writer_state=0)
    writer.add_rows([{
        "clip_id": "s@000000", "source_id": "s", "clip_num": 0,
        "clip_start": 0.0, "clip_duration": None, "speaker": "",
        "duration": 1.0, "text": "", "dataset": "ds",
        "audio_tokens": [101, 102], "text_tokens": [] if empty_text else [201, 202],
    }])
    writer.finalize()


@pytest.mark.parametrize("cache_kind", ["sft", "interleave_audio", "interleave_text"])
@pytest.mark.parametrize("span_part", ["offset", "count"])
@pytest.mark.parametrize("arrow_type,convert", [
    pytest.param(pa.float64(), float, id="float"),
    pytest.param(pa.bool_(), bool, id="bool"),
    pytest.param(pa.uint64(), int, id="unsigned"),
    pytest.param(pa.string(), str, id="string"),
])
def test_reader_rejects_non_signed_physical_span_type(
    tmp_path, cache_kind, span_part, arrow_type, convert,
):
    if cache_kind == "sft":
        writer = AudioTokenCacheWriter(tmp_path, rank=0)
        writer.add(audio_id="a", tokens=[101, 102], duration_sec=1.0)
        writer.finalize()
        index_path = next(tmp_path.glob("rank_*/audio_index.*.parquet"))
        column = f"token_{span_part}"
        load = lambda: load_audio_token_cache(tmp_path)
    else:
        from audio_tokenization.interleave.common import list_interleave_cache_partitions, load_interleave_cache

        _write_interleave_fixture(tmp_path)
        partition = list_interleave_cache_partitions(tmp_path)[0]
        index_path = next(partition.glob("rank_*/clips.*.parquet"))
        kind = cache_kind.removeprefix("interleave_")
        suffix = "length" if span_part == "count" else "offset"
        column = f"{kind}_token_{suffix}"
        load = lambda: load_interleave_cache(partition)

    # These values are aligned and bounded after coercion. Rejection must
    # depend on the physical type, before a cast can disguise it as an integer.
    _replace_index_value(
        index_path, column, convert(0 if span_part == "offset" else 1),
        arrow_type=arrow_type,
    )
    with pytest.raises(ValueError):
        load()


@pytest.mark.parametrize("kind", ["audio", "text"])
@pytest.mark.parametrize(
    "suffix,value", [("offset", 1), ("offset", -4), ("length", -1),
                     ("length", 3), ("offset", 2**63 - 4), ("offset", None), ("length", None)],
)
def test_interleave_reader_rejects_invalid_span_at_metadata_load(tmp_path, kind, suffix, value):
    from audio_tokenization.interleave.common import list_interleave_cache_partitions, load_interleave_cache

    _write_interleave_fixture(tmp_path)
    partition = list_interleave_cache_partitions(tmp_path)[0]
    _replace_index_value(next(partition.glob("rank_*/clips.*.parquet")), f"{kind}_token_{suffix}", value)

    with pytest.raises(ValueError):
        load_interleave_cache(partition)


def test_interleave_reader_supports_empty_text_payload(tmp_path):
    from audio_tokenization.interleave.common import (
        list_interleave_cache_partitions, load_interleave_cache, prepare_interleave_cache_and_runs,
    )

    _write_interleave_fixture(tmp_path, empty_text=True)
    partition = list_interleave_cache_partitions(tmp_path)[0]
    df, reader = load_interleave_cache(partition)
    cache, *_ = prepare_interleave_cache_and_runs(df, reader)

    assert cache.audio.slice(0, 1).to_pylist() == [[101, 102]]
    assert cache.text.slice(0, 1).to_pylist() == [[]]
    assert cache.text_lengths.tolist() == [0]


@pytest.mark.parametrize("audio_ids,expected", [(["fresh", "fresh"], "fresh"), (["fresh", "old"], "old"), (["fresh"], None)])
def test_duplicate_detection_does_not_rescan_previous_chunks(audio_ids, expected):
    from audio_tokenization.token_cache import _first_duplicate

    class CountingKeys(dict):
        keys_visited = 0

        def __iter__(self):
            for key in super().__iter__():
                self.keys_visited += 1
                yield key

    existing = CountingKeys.fromkeys(["old", "older"])
    assert _first_duplicate(audio_ids, existing=existing) == expected
    assert existing.keys_visited == 0


@pytest.mark.parametrize("offset,max_seq_len,expected,leftovers", [
    (0, 100, [[1, 10, 11, 99, 97, 13, 99, 23, 2]], [4]),
    (0, 5, [[1, 10, 11, 99, 2], [1, 13, 99, 23, 2]], [4]),
    (1, 5, [[1, 12, 99, 21, 22, 2], [1, 14, 99, 24, 2]], []),
])
def test_packing_reads_selected_memmap_spans_once(tmp_path, offset, max_seq_len, expected, leftovers):
    from audio_tokenization.interleave.common import _MemmapTokenAccessor
    from audio_tokenization.interleave.shift_by_one import _build_shift_sequence, _iter_shift_spans

    accesses = []

    class CountingAccessor(_MemmapTokenAccessor):
        def get(self, idx):
            accesses.append((self.kind, idx))
            return super().get(idx)

    runs = []
    for kind, payload, starts, lengths in [
        ("audio", [10, 11, 12, 13, 14, 15], [0, 2, 3, 4, 5], [2, 1, 1, 1, 1]),
        ("text", [20, 21, 22, 23, 24], [0, 1, 1, 3, 4], [1, 0, 2, 1, 1]),
    ]:
        path = tmp_path / f"{kind}.bin"
        np.asarray(payload, dtype=np.int32).tofile(path)
        accessor = CountingAccessor(
            [np.memmap(path, dtype=np.int32, mode="r")], np.zeros(5, dtype=np.int32),
            np.asarray(starts), np.asarray(lengths),
        )
        accessor.kind = kind
        # Exercise nested views; lengths must use the view's row origin.
        runs.append(accessor.slice(0, 5)[1:][0:] if offset == 1 else accessor.slice(0, 5))

    result, actual_leftovers = [], []
    for start, count, _ in _iter_shift_spans(runs[0].lengths, runs[1].lengths, 0, max_seq_len):
        if count == 1:
            actual_leftovers.append(start)
        else:
            result.append(_build_shift_sequence(*runs, start, count, 1, 2, 99, 97).tolist())

    assert result == expected
    assert actual_leftovers == leftovers
    assert len(accesses) == len(set(accesses)) == 4
