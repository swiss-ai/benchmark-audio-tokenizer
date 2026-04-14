import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter


def _read_int32_tokens(path: Path, offset: int, length: int) -> list[int]:
    itemsize = np.dtype(np.int32).itemsize
    with open(path, "rb") as f:
        f.seek(offset)
        return np.frombuffer(f.read(length * itemsize), dtype=np.int32).tolist()


def test_structured_cache_chunk_writer_writes_layout_and_offsets(tmp_path):
    writer = StructuredCacheChunkWriter(str(tmp_path), rank=0, chunk_id=0)
    writer.add_rows([
        {
            "clip_id": "a@000000",
            "source_id": "a",
            "clip_num": 0,
            "clip_start": 0.0,
            "speaker": "",
            "duration": 1.0,
            "text": "hello",
            "text_tokens": [11, 12],
            "audio_tokens": [1, 2, 3],
            "dataset": "ds",
        },
        {
            "clip_id": "a@000001",
            "source_id": "a",
            "clip_num": 1,
            "clip_start": 1.0,
            "speaker": "spk",
            "duration": 1.5,
            "text": "world",
            "text_tokens": [21],
            "audio_tokens": [4, 5],
            "dataset": "ds",
        },
    ])

    done = writer.finalize()
    assert done == 0

    layout = json.loads((tmp_path / "_CACHE_LAYOUT.json").read_text())
    assert layout["version"] == "v2"
    assert layout["commit_marker"] == "clips.parquet"

    rank_dir = tmp_path / "rank_0000"
    clips_path = rank_dir / "clips.000000.parquet"
    audio_path = rank_dir / "audio_tokens.000000.bin"
    text_path = rank_dir / "text_tokens.000000.bin"
    assert clips_path.exists()
    assert audio_path.exists()
    assert text_path.exists()

    table = pq.read_table(clips_path)
    rows = table.to_pylist()
    assert [row["clip_id"] for row in rows] == ["a@000000", "a@000001"]
    assert rows[0]["audio_token_offset"] == 0
    assert rows[0]["audio_token_length"] == 3
    assert rows[1]["audio_token_offset"] == 3 * np.dtype(np.int32).itemsize
    assert rows[1]["audio_token_length"] == 2
    assert rows[0]["text_token_offset"] == 0
    assert rows[0]["text_token_length"] == 2
    assert rows[1]["text_token_offset"] == 2 * np.dtype(np.int32).itemsize
    assert rows[1]["text_token_length"] == 1

    assert _read_int32_tokens(audio_path, rows[0]["audio_token_offset"], rows[0]["audio_token_length"]) == [1, 2, 3]
    assert _read_int32_tokens(audio_path, rows[1]["audio_token_offset"], rows[1]["audio_token_length"]) == [4, 5]
    assert _read_int32_tokens(text_path, rows[0]["text_token_offset"], rows[0]["text_token_length"]) == [11, 12]
    assert _read_int32_tokens(text_path, rows[1]["text_token_offset"], rows[1]["text_token_length"]) == [21]


def test_structured_cache_chunk_writer_removes_orphan_bins_without_commit_marker(tmp_path):
    rank_dir = tmp_path / "rank_0003"
    rank_dir.mkdir(parents=True)
    (rank_dir / "audio_tokens.000000.bin").write_bytes(b"orphan-audio")
    (rank_dir / "text_tokens.000000.bin").write_bytes(b"orphan-text")
    (rank_dir / "clips.000001.parquet.tmp").write_text("stale")

    writer = StructuredCacheChunkWriter(str(tmp_path), rank=3, chunk_id=0)

    assert not (rank_dir / "audio_tokens.000000.bin").exists()
    assert not (rank_dir / "text_tokens.000000.bin").exists()
    assert not (rank_dir / "clips.000001.parquet.tmp").exists()
    assert writer.num_rows == 0


def test_structured_cache_chunk_writer_raises_on_missing_bins_for_committed_parquet(tmp_path):
    rank_dir = tmp_path / "rank_0001"
    rank_dir.mkdir(parents=True)
    table = pa.table(
        {
            "clip_id": ["a@000000"],
            "source_id": ["a"],
            "clip_num": [0],
            "clip_start": [0.0],
            "speaker": [""],
            "duration": [1.0],
            "text": ["hello"],
            "dataset": ["ds"],
            "audio_token_offset": [0],
            "audio_token_length": [3],
            "text_token_offset": [0],
            "text_token_length": [2],
        },
        schema=StructuredCacheChunkWriter._get_schema(),
    )
    pq.write_table(table, rank_dir / "clips.000000.parquet")

    with pytest.raises(RuntimeError, match="exists without both token bins"):
        StructuredCacheChunkWriter(str(tmp_path), rank=1, chunk_id=0)


def test_structured_cache_chunk_writer_raises_on_layout_version_mismatch(tmp_path):
    (tmp_path / "_CACHE_LAYOUT.json").write_text(
        json.dumps({"version": "v1", "kind": "structured_interleave_cache"})
    )

    with pytest.raises(RuntimeError, match="layout mismatch"):
        StructuredCacheChunkWriter(str(tmp_path), rank=0, chunk_id=0)
