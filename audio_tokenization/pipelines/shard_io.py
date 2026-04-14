"""Shared shard writer helpers for Megatron indexed dataset output and cache formats."""

import json
import logging
import numpy as np
import os
from pathlib import Path
from typing import Any, Dict, List

from audio_tokenization.utils.indexed_dataset import DType, IndexedDatasetBuilder

logger = logging.getLogger(__name__)

INTERLEAVE_CACHE_LAYOUT_V2 = "v2"


def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_dir(path.parent)



def finalize_shard_writer(
    builder: IndexedDatasetBuilder,
    tmp_bin: str,
    tmp_idx: str,
    bin_path: str,
    idx_path: str,
) -> None:
    """Finalize index and atomically move temporary shard files in place.

    Calls ``fsync`` on both temp files before renaming to ensure data is
    durable on network filesystems (e.g. Lustre) where client write-back
    caching can lose data if the process is killed before a flush.
    """
    builder.finalize(tmp_idx)
    for p in (tmp_bin, tmp_idx):
        _fsync_file(Path(p))
    os.replace(tmp_bin, bin_path)
    os.replace(tmp_idx, idx_path)


# ---------------------------------------------------------------------------
# Parquet chunk writer for audio_text_interleaving pre-tokenization cache
# ---------------------------------------------------------------------------


class ParquetChunkWriter:
    """Streaming Parquet writer with periodic row group flushing.

    Buffers rows in columnar form and flushes them as row groups to a
    ``ParquetWriter`` when the buffer exceeds ``row_group_size``.  This
    bounds memory usage regardless of how many samples are written
    between checkpoints.

    ``finalize()`` flushes remaining rows, closes the writer, fsyncs,
    and atomically renames ``.tmp`` → ``.parquet``.

    Schema columns:
        clip_id (str), source_id (str), clip_num (int), speaker (str),
        duration (float), text (str), text_tokens (list<int32>),
        audio_tokens (list<int32>), dataset (str)
    """

    _SCHEMA = None

    @classmethod
    def _get_schema(cls):
        if cls._SCHEMA is None:
            import pyarrow as pa
            cls._SCHEMA = pa.schema([
                ("clip_id", pa.string()),
                ("source_id", pa.string()),
                ("clip_num", pa.int64()),
                ("clip_start", pa.float64()),
                ("speaker", pa.string()),
                ("duration", pa.float64()),
                ("text", pa.string()),
                ("text_tokens", pa.list_(pa.int32())),
                ("audio_tokens", pa.list_(pa.int32())),
                ("dataset", pa.string()),
            ])
        return cls._SCHEMA

    def __init__(self, output_dir: str, rank: int, chunk_id: int = 0,
                 row_group_size: int = 10000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.chunk_id = chunk_id
        self.row_group_size = row_group_size
        self._columns: Dict[str, list] = {name: [] for name in self._get_schema().names}
        self._buffered: int = 0
        self._total_rows: int = 0
        self._writer = None
        self._tmp_path = None
        self._final_path = None

    def _open_writer(self):
        """Lazily open a ParquetWriter for the current chunk."""
        import pyarrow.parquet as pq
        self._final_path = self.output_dir / f"rank_{self.rank:04d}_chunk_{self.chunk_id:04d}.parquet"
        self._tmp_path = self._final_path.with_suffix(".parquet.tmp")
        self._writer = pq.ParquetWriter(str(self._tmp_path), self._get_schema())

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Append a batch of rows to the column buffer."""
        if not rows:
            return
        if self._writer is None:
            self._open_writer()
        for row in rows:
            for key in self._get_schema().names:
                self._columns[key].append(row[key])
        self._buffered += len(rows)
        self._total_rows += len(rows)

    def flush_if_needed(self) -> None:
        """Write a row group to disk if the buffer exceeds ``row_group_size``."""
        if self._buffered >= self.row_group_size:
            self._flush_row_group()

    def _flush_row_group(self) -> None:
        """Write the current column buffer as a row group and clear it."""
        if self._buffered == 0:
            return
        import pyarrow as pa
        table = pa.table(self._columns, schema=self._get_schema())
        self._writer.write_table(table)
        for col in self._columns.values():
            col.clear()
        self._buffered = 0

    @property
    def num_rows(self) -> int:
        return self._total_rows

    @property
    def num_samples(self) -> int:
        """Alias for ``num_rows``."""
        return self._total_rows

    def finalize(self) -> int:
        """Flush remaining rows, close writer, fsync, rename. Returns finalized chunk_id."""
        if self._writer is None:
            self._open_writer()
        self._flush_row_group()
        self._writer.close()
        _fsync_file(self._tmp_path)
        os.replace(str(self._tmp_path), str(self._final_path))

        finalized_id = self.chunk_id
        logger.info(
            f"[rank {self.rank}] Wrote {self._total_rows} rows to {self._final_path.name}"
        )

        # Reset for next chunk
        self.chunk_id += 1
        for col in self._columns.values():
            col.clear()
        self._buffered = 0
        self._total_rows = 0
        self._writer = None
        self._tmp_path = None
        self._final_path = None
        return finalized_id


class StructuredCacheChunkWriter:
    """Structured v2 interleave cache writer.

    Stores lightweight clip metadata in Parquet and token payloads in flat
    ``.bin`` files with byte offsets/lengths. The Parquet file is the chunk
    commit marker: it is renamed only after both token bins have been fsync'ed
    and renamed into place.
    """

    _SCHEMA = None

    @classmethod
    def _get_schema(cls):
        if cls._SCHEMA is None:
            import pyarrow as pa

            cls._SCHEMA = pa.schema([
                ("clip_id", pa.string()),
                ("source_id", pa.string()),
                ("clip_num", pa.int64()),
                ("clip_start", pa.float64()),
                ("speaker", pa.string()),
                ("duration", pa.float64()),
                ("text", pa.string()),
                ("dataset", pa.string()),
                ("audio_token_offset", pa.int64()),
                ("audio_token_length", pa.int32()),
                ("text_token_offset", pa.int64()),
                ("text_token_length", pa.int32()),
            ])
        return cls._SCHEMA

    def __init__(self, output_dir: str, rank: int, chunk_id: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.chunk_id = chunk_id
        self.rank_dir = self.output_dir / f"rank_{rank:04d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)

        self._audio_tmp_path = None
        self._text_tmp_path = None
        self._clips_tmp_path = None
        self._audio_final_path = None
        self._text_final_path = None
        self._clips_final_path = None
        self._audio_fh = None
        self._text_fh = None
        self._audio_offset = 0
        self._text_offset = 0
        self._rows: List[Dict[str, Any]] = []
        self._total_rows = 0
        self._opened = False

        self._write_layout_metadata()
        self._cleanup_incomplete_rank_dir()

    def _write_layout_metadata(self) -> None:
        path = self.output_dir / "_CACHE_LAYOUT.json"
        payload = {
            "version": INTERLEAVE_CACHE_LAYOUT_V2,
            "kind": "structured_interleave_cache",
            "commit_marker": "clips.parquet",
            "rank_dirs": True,
            "token_dtype": "int32",
            "metadata_columns": self._get_schema().names,
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid structured cache layout metadata at {path}") from exc
            if existing.get("version") != INTERLEAVE_CACHE_LAYOUT_V2:
                raise RuntimeError(
                    f"Structured cache layout mismatch at {path}: "
                    f"expected version {INTERLEAVE_CACHE_LAYOUT_V2!r}, "
                    f"found {existing.get('version')!r}"
                )
            return
        _write_json_atomic(path, payload)

    def _cleanup_incomplete_rank_dir(self) -> None:
        # Each rank owns exactly one rank-local cache directory. Cleanup is intentionally
        # lock-free and assumes there is no concurrent writer for the same rank.
        for tmp in self.rank_dir.glob("*.tmp"):
            tmp.unlink()

        for clips_path in self.rank_dir.glob("clips.*.parquet"):
            stem = clips_path.name.split(".")[1]
            audio_path = self.rank_dir / f"audio_tokens.{stem}.bin"
            text_path = self.rank_dir / f"text_tokens.{stem}.bin"
            if not audio_path.exists() or not text_path.exists():
                raise RuntimeError(
                    f"Incomplete structured cache chunk detected for rank {self.rank}: "
                    f"{clips_path.name} exists without both token bins."
                )

        clip_ids = {p.name.split(".")[1] for p in self.rank_dir.glob("clips.*.parquet")}
        for bin_path in list(self.rank_dir.glob("audio_tokens.*.bin")) + list(self.rank_dir.glob("text_tokens.*.bin")):
            stem = bin_path.name.split(".")[1]
            if stem not in clip_ids:
                logger.warning(
                    f"[rank {self.rank}] Removing orphan structured cache payload with no commit marker: {bin_path.name}"
                )
                bin_path.unlink()

    def _open_chunk(self) -> None:
        stem = f"{self.chunk_id:06d}"
        self._audio_tmp_path = self.rank_dir / f"audio_tokens.{stem}.bin.tmp"
        self._text_tmp_path = self.rank_dir / f"text_tokens.{stem}.bin.tmp"
        self._clips_tmp_path = self.rank_dir / f"clips.{stem}.parquet.tmp"
        self._audio_final_path = self.rank_dir / f"audio_tokens.{stem}.bin"
        self._text_final_path = self.rank_dir / f"text_tokens.{stem}.bin"
        self._clips_final_path = self.rank_dir / f"clips.{stem}.parquet"
        self._audio_fh = open(self._audio_tmp_path, "wb")
        self._text_fh = open(self._text_tmp_path, "wb")
        self._audio_offset = 0
        self._text_offset = 0
        self._rows = []
        self._total_rows = 0
        self._opened = True

    @property
    def num_rows(self) -> int:
        return self._total_rows

    @property
    def num_samples(self) -> int:
        return self._total_rows

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        if not self._opened:
            self._open_chunk()

        for row in rows:
            audio_tokens = np.asarray(row["audio_tokens"], dtype=np.int32)
            text_tokens = np.asarray(row["text_tokens"], dtype=np.int32)
            self._audio_fh.write(audio_tokens.tobytes())
            self._text_fh.write(text_tokens.tobytes())
            self._rows.append({
                "clip_id": row["clip_id"],
                "source_id": row["source_id"],
                "clip_num": row["clip_num"],
                "clip_start": row["clip_start"],
                "speaker": row["speaker"],
                "duration": row["duration"],
                "text": row["text"],
                "dataset": row["dataset"],
                "audio_token_offset": self._audio_offset,
                "audio_token_length": int(audio_tokens.shape[0]),
                "text_token_offset": self._text_offset,
                "text_token_length": int(text_tokens.shape[0]),
            })
            self._audio_offset += len(audio_tokens) * np.dtype(np.int32).itemsize
            self._text_offset += len(text_tokens) * np.dtype(np.int32).itemsize
            self._total_rows += 1

    def flush_if_needed(self) -> None:
        """No-op for v2; metadata is flushed once per finalized chunk."""
        return None

    def finalize(self) -> int:
        if not self._opened:
            return self.chunk_id

        self._audio_fh.flush()
        self._text_fh.flush()
        os.fsync(self._audio_fh.fileno())
        os.fsync(self._text_fh.fileno())
        self._audio_fh.close()
        self._text_fh.close()

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({name: [row[name] for row in self._rows] for name in self._get_schema().names}, schema=self._get_schema())
        pq.write_table(table, self._clips_tmp_path)
        _fsync_file(self._clips_tmp_path)

        os.replace(self._audio_tmp_path, self._audio_final_path)
        os.replace(self._text_tmp_path, self._text_final_path)
        os.replace(self._clips_tmp_path, self._clips_final_path)
        _fsync_dir(self.rank_dir)

        finalized_id = self.chunk_id
        logger.info(
            f"[rank {self.rank}] Wrote structured cache chunk {self._clips_final_path.name} "
            f"with {self._total_rows} rows"
        )

        self.chunk_id += 1
        self._audio_tmp_path = None
        self._text_tmp_path = None
        self._clips_tmp_path = None
        self._audio_final_path = None
        self._text_final_path = None
        self._clips_final_path = None
        self._audio_fh = None
        self._text_fh = None
        self._audio_offset = 0
        self._text_offset = 0
        self._rows = []
        self._total_rows = 0
        self._opened = False
        return finalized_id


def parquet_cache_exists(parquet_dir: Path) -> bool:
    """Check if a Parquet cache directory has at least one .parquet file."""
    if not parquet_dir.is_dir():
        return False
    return any(parquet_dir.glob("*.parquet"))
