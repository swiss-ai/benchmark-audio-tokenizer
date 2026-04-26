"""Shared shard writer helpers for Megatron indexed dataset output and cache formats."""

import hashlib
import json
import logging
import numpy as np
import os
from pathlib import Path
from typing import Any, Dict, List

from audio_tokenization.contracts.artifacts import (
    INTERLEAVE_CACHE_LAYOUT_FILENAME,
    INTERLEAVE_CACHE_LAYOUT_V2,
    INTERLEAVE_CACHE_OUTPUT_STEM,
    INTERLEAVE_CACHE_SCHEMA_VERSION,
    next_chunk_id,
    prune_orphan_bin_files,
    validate_v2_chunks_complete,
)
from audio_tokenization.utils.io import (
    atomic_replace_files,
    atomic_streaming_write,
    atomic_write_json,
    cleanup_tmp_files,
    fsync_file,
)
from audio_tokenization.utils.indexed_dataset import DType, IndexedDatasetBuilder
from audio_tokenization.utils.indexed_dataset.constants import CUT_ID_SIDECAR_SUFFIX

logger = logging.getLogger(__name__)


class CutIdSidecarWriter:
    """Write one JSON-encoded cut ID per Megatron document.

    The line number is the local document index inside the corresponding
    ``rank_XXXX_chunk_YYYY.{bin,idx}`` pair. This sidecar is only for
    Megatron-format outputs; interleave cache rows carry identity natively.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stream = atomic_streaming_write(self.path, mode="wb", compression="zst")
        self._file = self._stream.__enter__()
        self.count = 0
        self._closed = False

    @property
    def tmp_path(self) -> Path:
        return self._stream.tmp_path

    def write(self, cut_id: str) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed CutIdSidecarWriter")
        line = json.dumps(str(cut_id)) + "\n"
        self._file.write(line.encode("utf-8"))
        self.count += 1

    def finalize(self) -> None:
        if not self._closed:
            self._stream.commit()
            self._closed = True

    def close_temp(self) -> None:
        if not self._closed:
            self._stream.close_temp()

    def mark_committed(self) -> None:
        self._stream.mark_committed()
        self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self._stream.abort()
            self._closed = True


def finalize_shard_writer(
    builder: IndexedDatasetBuilder,
    tmp_bin: str,
    tmp_idx: str,
    bin_path: str,
    idx_path: str,
    cut_id_writer: CutIdSidecarWriter | None = None,
) -> None:
    """Finalize index and atomically move temporary shard files in place.

    Calls ``fsync`` on both temp files before renaming to ensure data is
    durable on network filesystems (e.g. Lustre) where client write-back
    caching can lose data if the process is killed before a flush.
    """
    try:
        if cut_id_writer is not None:
            expected_docs = max(0, len(builder.document_indices) - 1)
            if cut_id_writer.count != expected_docs:
                cut_id_writer.abort()
                raise RuntimeError(
                    "Cut ID sidecar/document count mismatch for "
                    f"{bin_path}: sidecar={cut_id_writer.count}, indexed_docs={expected_docs}"
                )

        builder.finalize(tmp_idx)
        if cut_id_writer is not None:
            cut_id_writer.close_temp()
            atomic_replace_files(
                [
                    (tmp_bin, bin_path),
                    (cut_id_writer.tmp_path, cut_id_writer.path),
                    (tmp_idx, idx_path),
                ],
                fsync_sources=True,
            )
            cut_id_writer.mark_committed()
        else:
            atomic_replace_files(
                [(tmp_bin, bin_path), (tmp_idx, idx_path)],
                fsync_sources=True,
            )
    except BaseException:
        if cut_id_writer is not None:
            cut_id_writer.abort()
        raise


class StructuredCacheChunkWriter:
    """Structured v2 interleave cache writer.

    This is a partition-aware wrapper. Rows are routed into source-local
    partition directories, and each ``(partition, rank)`` pair owns its own
    chunk stream:

        <cache_root>/<partition>/rank_<rank>/clips.NNNNNN.parquet
        <cache_root>/<partition>/rank_<rank>/audio_tokens.NNNNNN.bin
        <cache_root>/<partition>/rank_<rank>/text_tokens.NNNNNN.bin
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
                ("clip_duration", pa.float64()),
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

    @staticmethod
    def _sanitize_partition_value(value: Any) -> str:
        text = str(value).strip()
        if not text:
            text = "unknown"
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)

    @staticmethod
    def _hash_bucket_name(value: str, num_buckets: int) -> str:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "little") % num_buckets
        return f"bucket_{bucket:04d}"

    class _PartitionShardWriter:
        def __init__(self, partition_dir: Path, rank: int, chunk_id: int):
            self.partition_dir = partition_dir
            self.partition_dir.mkdir(parents=True, exist_ok=True)
            self.rank = rank
            self.chunk_id = chunk_id
            self.rank_dir = self.partition_dir / f"rank_{rank:04d}"
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

            self._cleanup_incomplete_rank_dir()

        @staticmethod
        def infer_next_chunk_id(partition_dir: Path, rank: int) -> int:
            return next_chunk_id(partition_dir / f"rank_{rank:04d}")

        def _cleanup_incomplete_rank_dir(self) -> None:
            # Each rank owns exactly one rank-local directory within a partition.
            cleanup_tmp_files(self.rank_dir, "*.tmp", logger=logger)
            validate_v2_chunks_complete(
                self.rank_dir,
                error_label=f"structured cache (rank {self.rank})",
            )
            prune_orphan_bin_files(
                self.rank_dir,
                logger=logger,
                label=f"[rank {self.rank}] structured cache payload with no commit marker",
            )

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
                    "clip_duration": row.get("clip_duration"),
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

            table = pa.table(
                {name: [row[name] for row in self._rows] for name in StructuredCacheChunkWriter._get_schema().names},
                schema=StructuredCacheChunkWriter._get_schema(),
            )
            pq.write_table(table, self._clips_tmp_path)
            fsync_file(self._clips_tmp_path)

            atomic_replace_files(
                [
                    (self._audio_tmp_path, self._audio_final_path),
                    (self._text_tmp_path, self._text_final_path),
                    (self._clips_tmp_path, self._clips_final_path),
                ],
                fsync_sources=False,
            )

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

        def get_state(self) -> int:
            return self.chunk_id

    def __init__(
        self,
        output_dir: str,
        rank: int,
        writer_state: int | Dict[str, int] = 0,
        partitioning: Dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.partitioning = partitioning or {"type": "hash", "field": "source_id", "num_buckets": 16}
        self._initial_writer_state = (
            dict(writer_state) if isinstance(writer_state, dict) else {"__default__": int(writer_state)}
        )
        self._partition_writers: Dict[str, StructuredCacheChunkWriter._PartitionShardWriter] = {}
        self._num_rows = 0
        self._chunks_written = 0
        self._write_layout_metadata(
            self.output_dir,
            {
                "version": INTERLEAVE_CACHE_LAYOUT_V2,
                "schema_version": INTERLEAVE_CACHE_SCHEMA_VERSION,
                "kind": "structured_interleave_cache",
                "commit_marker": "clips.parquet",
                "rank_dirs": False,
                "token_dtype": "int32",
                "metadata_columns": self._get_schema().names,
                "partitioned": True,
                "partitioning": self.partitioning,
            },
        )
        self._bootstrap_existing_partition_writers()

    def _write_layout_metadata(self, path: Path, payload: Dict[str, Any]) -> None:
        layout_path = path / INTERLEAVE_CACHE_LAYOUT_FILENAME
        if layout_path.exists():
            try:
                existing = json.loads(layout_path.read_text())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid structured cache layout metadata at {layout_path}") from exc
            if existing.get("version") != INTERLEAVE_CACHE_LAYOUT_V2:
                raise RuntimeError(
                    f"Structured cache layout mismatch at {layout_path}: "
                    f"expected version {INTERLEAVE_CACHE_LAYOUT_V2!r}, "
                    f"found {existing.get('version')!r}"
                )
            if existing.get("schema_version") != INTERLEAVE_CACHE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Structured cache schema mismatch at {layout_path}: "
                    f"expected schema_version {INTERLEAVE_CACHE_SCHEMA_VERSION!r}, "
                    f"found {existing.get('schema_version')!r}. "
                    "Rerun tokenize into a fresh interleave cache directory."
                )
            return
        atomic_write_json(layout_path, payload)

    def _partition_name_for_row(self, row: Dict[str, Any]) -> str:
        if self.partitioning["type"] == "hash":
            field = self.partitioning["field"]
            return self._hash_bucket_name(str(row[field]), self.partitioning["num_buckets"])
        field_value = row.get("_partition_value", row.get(self.partitioning["field"]))
        if field_value is None:
            raise ValueError(f"Missing partition field value for {self.partitioning['field']!r}")
        return f"{self.partitioning['field']}={self._sanitize_partition_value(field_value)}"

    def _get_partition_writer(self, partition_name: str) -> _PartitionShardWriter:
        writer = self._partition_writers.get(partition_name)
        if writer is not None:
            return writer

        partition_dir = self.output_dir / partition_name
        partition_dir.mkdir(parents=True, exist_ok=True)
        self._write_layout_metadata(
            partition_dir,
            {
                "version": INTERLEAVE_CACHE_LAYOUT_V2,
                "schema_version": INTERLEAVE_CACHE_SCHEMA_VERSION,
                "kind": "structured_interleave_cache",
                "commit_marker": "clips.parquet",
                "rank_dirs": True,
                "partitioned_root": str(self.output_dir),
                "token_dtype": "int32",
                "metadata_columns": self._get_schema().names,
                "partitioning": self.partitioning,
            },
        )

        if partition_name in self._initial_writer_state:
            chunk_id = int(self._initial_writer_state[partition_name])
        elif "__default__" in self._initial_writer_state:
            chunk_id = int(self._initial_writer_state["__default__"])
        else:
            chunk_id = self._PartitionShardWriter.infer_next_chunk_id(partition_dir, self.rank)

        writer = self._PartitionShardWriter(partition_dir, self.rank, chunk_id)
        self._partition_writers[partition_name] = writer
        return writer

    def _bootstrap_existing_partition_writers(self) -> None:
        for partition_dir in sorted(
            p for p in self.output_dir.iterdir()
            if p.is_dir() and (p / INTERLEAVE_CACHE_LAYOUT_FILENAME).exists()
        ):
            rank_dir = partition_dir / f"rank_{self.rank:04d}"
            if not rank_dir.exists():
                continue
            partition_name = partition_dir.name
            self._get_partition_writer(partition_name)

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @property
    def num_samples(self) -> int:
        return self._num_rows

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            partition_name = self._partition_name_for_row(row)
            grouped.setdefault(partition_name, []).append(row)
        for partition_name, part_rows in grouped.items():
            writer = self._get_partition_writer(partition_name)
            writer.add_rows(part_rows)
            self._num_rows += len(part_rows)

    def finalize(self) -> Dict[str, int]:
        done: Dict[str, int] = {}
        for partition_name, writer in sorted(self._partition_writers.items()):
            finalized = writer.finalize()
            if writer.get_state() != finalized:
                self._chunks_written += 1
            done[partition_name] = finalized
        for partition_name, chunk_id in self._initial_writer_state.items():
            if partition_name != "__default__" and partition_name not in done:
                done[partition_name] = int(chunk_id)
        self._num_rows = 0
        return done

    def get_state(self) -> Dict[str, int]:
        state = {
            partition_name: int(chunk_id)
            for partition_name, chunk_id in self._initial_writer_state.items()
            if partition_name != "__default__"
        }
        for partition_name, writer in self._partition_writers.items():
            state[partition_name] = writer.get_state()
        return state
