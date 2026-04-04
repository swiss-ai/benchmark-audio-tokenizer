"""Shared shard writer helpers for Megatron indexed dataset output and Parquet cache."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from audio_tokenization.utils.indexed_dataset import DType, IndexedDatasetBuilder

logger = logging.getLogger(__name__)



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
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
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

        fd = os.open(str(self._tmp_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
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


def parquet_cache_exists(parquet_dir: Path) -> bool:
    """Check if a Parquet cache directory has at least one .parquet file."""
    if not parquet_dir.is_dir():
        return False
    return any(parquet_dir.glob("*.parquet"))
