"""Shared utilities for interleaved-tokenization indexed dataset builders.

Provides functions used by both the fixed-window *pattern* mode and the
variable-length *accumulate* mode:

- Token ID loading (BOS/EOS/transition tokens)
- Consecutive-run detection
- Megatron .idx writing
- Shard merging
- Worker partitioning
- Parquet loading and Arrow preparation
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import struct
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
from transformers import AutoTokenizer

from audio_tokenization.utils.indexed_dataset.indexed_dataset_megatron import (
    _INDEX_HEADER,
    DType,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)

# Re-export indexed dataset symbols so callers import from one place
__all__ = [
    "_INDEX_HEADER",
    "DType",
    "IndexedDatasetBuilder",
    "get_bin_path",
    "get_idx_path",
    "TR_KEY",
    "_HIST_BINS",
    "_HIST_LABELS",
    "_PERCENTILES",
    "format_distribution",
    "load_token_ids",
    "find_consecutive_runs",
    "_detect_runs",
    "_write_idx_file",
    "_merge_shards",
    "_partition_runs",
    "list_interleave_cache_partitions",
    "summarize_partition_stats",
    "print_partition_stats",
    "load_parquets",
    "prepare_arrow_and_runs",
    "load_interleave_cache",
    "prepare_interleave_cache_and_runs",
    "prepare_length_metadata",
    "compute_ratio_adjustment",
    "_LazyArrowList",
]

# ---------------------------------------------------------------------------
# Lazy Arrow helper
# ---------------------------------------------------------------------------


class _LazyArrowList:
    """Lazy list-like wrapper around an Arrow chunked array slice.

    Converts one element at a time via ``as_py()`` on access instead of
    materializing the entire slice upfront with ``to_pylist()``.
    Reduces peak memory for long runs.
    """
    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, idx):
        return self._arr[idx].as_py()

    def __len__(self):
        return len(self._arr)


class _TokenRunView:
    """Lazy list-like view over token sequences in a prepared cache."""

    __slots__ = ("_accessor", "_start", "_length")

    def __init__(self, accessor, start: int, length: int):
        self._accessor = accessor
        self._start = start
        self._length = length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._length)
            if step != 1:
                return [self._accessor.get(self._start + i) for i in range(start, stop, step)]
            return _TokenRunView(self._accessor, self._start + start, stop - start)
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        return self._accessor.get(self._start + idx)

    def __len__(self):
        return self._length

    def to_pylist(self) -> list[list[int]]:
        return [self._accessor.get(self._start + i) for i in range(self._length)]


class _ArrowTokenAccessor:
    __slots__ = ("_arr", "_lengths")

    def __init__(self, arr):
        import pyarrow.compute as pc

        self._arr = arr
        self._lengths = pc.list_value_length(arr).to_numpy()

    def get(self, idx: int) -> list[int]:
        return self._arr[idx].as_py()

    def slice(self, start: int, length: int) -> _TokenRunView:
        return _TokenRunView(self, start, length)

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths


class _MemmapTokenAccessor:
    __slots__ = ("_mmaps", "_chunk_indices", "_starts", "_lengths")

    def __init__(
        self,
        mmaps: list[np.memmap],
        chunk_indices: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
    ):
        self._mmaps = mmaps
        self._chunk_indices = chunk_indices
        self._starts = starts
        self._lengths = lengths

    def get(self, idx: int) -> list[int]:
        chunk_idx = int(self._chunk_indices[idx])
        start = int(self._starts[idx])
        length = int(self._lengths[idx])
        return self._mmaps[chunk_idx][start:start + length].tolist()

    def slice(self, start: int, length: int) -> _TokenRunView:
        return _TokenRunView(self, start, length)

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths


class PreparedInterleaveCache:
    __slots__ = ("audio", "text")

    def __init__(self, audio, text):
        self.audio = audio
        self.text = text

    @property
    def audio_lengths(self) -> np.ndarray:
        return self.audio.lengths

    @property
    def text_lengths(self) -> np.ndarray:
        return self.text.lengths


class _V1InterleaveCacheReader:
    def __init__(self, parquet_dir: Path):
        self.parquet_dir = parquet_dir
        self.parquet_files = sorted(
            p
            for p in parquet_dir.glob("rank_*_chunk_*.parquet")
            if not p.name.endswith(".tmp")
        )
        if not self.parquet_files:
            self.parquet_files = sorted(
                p
                for p in parquet_dir.glob("*.parquet")
                if not p.name.endswith(".tmp")
            )

    def load_metadata(self) -> pl.DataFrame:
        return pl.read_parquet(
            self.parquet_files,
            columns=["source_id", "clip_num", "audio_tokens", "text_tokens"],
        )

    def prepare(self, sorted_df: pl.DataFrame) -> PreparedInterleaveCache:
        _audio_raw = sorted_df["audio_tokens"].to_arrow()
        _text_raw = sorted_df["text_tokens"].to_arrow()
        audio_arrow = _audio_raw.combine_chunks() if isinstance(_audio_raw, pa.ChunkedArray) else _audio_raw
        text_arrow = _text_raw.combine_chunks() if isinstance(_text_raw, pa.ChunkedArray) else _text_raw
        del _audio_raw, _text_raw
        return PreparedInterleaveCache(
            _ArrowTokenAccessor(audio_arrow),
            _ArrowTokenAccessor(text_arrow),
        )


class _V2InterleaveCacheReader:
    def __init__(self, cache_dir: Path, layout: dict[str, object]):
        self.cache_dir = cache_dir
        self.layout = layout
        self.chunks: list[tuple[Path, Path, Path]] = []
        for rank_dir in sorted(p for p in cache_dir.glob("rank_*") if p.is_dir()):
            for clips_path in sorted(rank_dir.glob("clips.*.parquet")):
                stem = clips_path.name.split(".")[1]
                audio_path = rank_dir / f"audio_tokens.{stem}.bin"
                text_path = rank_dir / f"text_tokens.{stem}.bin"
                if not audio_path.exists() or not text_path.exists():
                    raise RuntimeError(
                        f"Incomplete v2 structured cache chunk under {rank_dir}: "
                        f"{clips_path.name} missing token bins"
                    )
                self.chunks.append((clips_path, audio_path, text_path))

    def load_metadata(self) -> pl.DataFrame:
        parts: list[pl.DataFrame] = []
        for chunk_idx, (clips_path, _audio_path, _text_path) in enumerate(self.chunks):
            part = pl.read_parquet(
                clips_path,
                columns=[
                    "source_id",
                    "clip_num",
                    "audio_token_offset",
                    "audio_token_length",
                    "text_token_offset",
                    "text_token_length",
                ],
            ).with_columns(
                pl.lit(chunk_idx).alias("_chunk_idx"),
                (pl.col("audio_token_offset") // np.dtype(np.int32).itemsize).cast(pl.Int64).alias("_audio_token_start"),
                (pl.col("text_token_offset") // np.dtype(np.int32).itemsize).cast(pl.Int64).alias("_text_token_start"),
            )
            parts.append(part)
        if not parts:
            return pl.DataFrame(
                {
                    "source_id": pl.Series([], dtype=pl.String),
                    "clip_num": pl.Series([], dtype=pl.Int64),
                    "audio_token_offset": pl.Series([], dtype=pl.Int64),
                    "audio_token_length": pl.Series([], dtype=pl.Int32),
                    "text_token_offset": pl.Series([], dtype=pl.Int64),
                    "text_token_length": pl.Series([], dtype=pl.Int32),
                    "_chunk_idx": pl.Series([], dtype=pl.Int32),
                    "_audio_token_start": pl.Series([], dtype=pl.Int64),
                    "_text_token_start": pl.Series([], dtype=pl.Int64),
                }
            )
        return pl.concat(parts, how="vertical")

    def _ensure_fd_budget(self) -> None:
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit in (-1, resource.RLIM_INFINITY):
            return

        required_fds = 2 * len(self.chunks)
        reserve_fds = 32
        try:
            current_open = len(os.listdir("/proc/self/fd"))
        except OSError:
            current_open = None

        projected = required_fds + reserve_fds
        if current_open is not None:
            projected += current_open

        if projected > soft_limit:
            current_part = f"current_open={current_open}, " if current_open is not None else ""
            raise RuntimeError(
                "Insufficient file descriptor budget for v2 structured cache: "
                f"{current_part}required_memmaps={required_fds}, reserve={reserve_fds}, "
                f"soft_limit={soft_limit}. Split planning by partition or raise ulimit -n."
            )

    def prepare(self, sorted_df: pl.DataFrame) -> PreparedInterleaveCache:
        self._ensure_fd_budget()
        audio_mmaps = [np.memmap(audio_path, dtype=np.int32, mode="r") for _clips_path, audio_path, _text_path in self.chunks]
        text_mmaps = [np.memmap(text_path, dtype=np.int32, mode="r") for _clips_path, _audio_path, text_path in self.chunks]

        chunk_indices = sorted_df["_chunk_idx"].to_numpy()
        audio_starts = sorted_df["_audio_token_start"].to_numpy()
        audio_lengths = sorted_df["audio_token_length"].to_numpy()
        text_starts = sorted_df["_text_token_start"].to_numpy()
        text_lengths = sorted_df["text_token_length"].to_numpy()

        return PreparedInterleaveCache(
            _MemmapTokenAccessor(audio_mmaps, chunk_indices, audio_starts, audio_lengths),
            _MemmapTokenAccessor(text_mmaps, chunk_indices, text_starts, text_lengths),
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TR_KEY = "transcribe"

# Histogram bins / labels for sequence-length distributions
_HIST_BINS = [0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 8192, 999_999]
_HIST_LABELS = [
    "0-99", "100-199", "200-499", "500-999", "1K-2K", "2K-3K",
    "3K-4K", "4K-5K", "5K-6K", "6K-7K", "7K-8K", "8K-8192", ">8192",
]
_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def load_token_ids(tokenizer_path: str) -> tuple[int, int, int, int, int, int]:
    """Load BOS, EOS, stt_continue, stt_transcribe, tts_continue IDs and vocab_size."""
    from audio_tokenization.utils.token_mapping import get_structure_tokens

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    assert bos_id is not None, "BOS token must be defined in tokenizer"
    assert eos_id is not None, "EOS token must be defined in tokenizer"

    st = get_structure_tokens(
        tokenizer_path,
        required=["stt_continue", "stt_transcribe", "tts_continue"],
    )

    vocab_size = len(tokenizer)
    return bos_id, eos_id, st["stt_continue"], st["stt_transcribe"], st["tts_continue"], vocab_size


def find_consecutive_runs(sorted_clip_nums: list[int]) -> list[list[int]]:
    """Split sorted clip numbers into runs of consecutive integers.

    >>> find_consecutive_runs([3, 4, 5, 8, 9, 15])
    [[3, 4, 5], [8, 9], [15]]
    """
    if not sorted_clip_nums:
        return []
    runs: list[list[int]] = []
    current: list[int] = [sorted_clip_nums[0]]
    for prev, cur in zip(sorted_clip_nums, sorted_clip_nums[1:]):
        if cur == prev + 1:
            current.append(cur)
        else:
            runs.append(current)
            current = [cur]
    runs.append(current)
    return runs


# ---------------------------------------------------------------------------
# Vectorized run detection
# ---------------------------------------------------------------------------


def _detect_runs(df: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Sort by (source_id, clip_num) and detect consecutive-clip runs.

    A new run starts whenever the source changes **or** clip_num is not the
    previous clip_num + 1.

    **Invariant**: this function assumes clip_num is dense (0, 1, 2, ...)
    within each source_id in the SHAR before tokenization. Gaps in clip_num
    after tokenization indicate filtered clips (min_duration, min_rms_db)
    and are treated as real discontinuities — the run breaks and isolated
    clips become transcribe sequences.

    Sparse or timestamp-based clip_num (e.g. start_ms) will cause excessive
    fragmentation. Ensure SHAR patching assigns dense clip_num before
    tokenization.

    Returns (sorted_df, run_starts, run_lengths) where *run_starts* and
    *run_lengths* are 1-D int arrays indexing into the sorted dataframe.
    """
    df = df.sort(["source_id", "clip_num"])
    is_new_run = (
        (df["source_id"] != df["source_id"].shift(1))
        | (df["clip_num"] != (df["clip_num"].shift(1) + 1))
    ).fill_null(True)  # first row is always a run start
    starts = np.where(is_new_run.to_numpy())[0]
    lengths = np.diff(np.append(starts, len(df)))
    return df, starts, lengths


# ---------------------------------------------------------------------------
# Megatron .idx writer
# ---------------------------------------------------------------------------


def _write_idx_file(
    idx_path: str,
    dtype: type,
    sequence_lengths: np.ndarray,
    document_indices: np.ndarray,
) -> None:
    """Write a Megatron .idx file from numpy arrays."""
    itemsize = DType.size(dtype)
    pointers = np.zeros(len(sequence_lengths), dtype=np.int64)
    if len(sequence_lengths) > 0:
        np.cumsum(sequence_lengths.astype(np.int64) * itemsize, out=pointers)
        # shift right: pointers[i] = cumsum[i-1], pointers[0] = 0
        pointers = np.roll(pointers, 1)
        pointers[0] = 0

    with open(idx_path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", DType.code_from_dtype(dtype)))
        f.write(struct.pack("<Q", len(sequence_lengths)))
        f.write(struct.pack("<Q", len(document_indices)))
        f.write(sequence_lengths.astype(np.int32).tobytes(order="C"))
        f.write(pointers.tobytes(order="C"))
        f.write(document_indices.astype(np.int64).tobytes(order="C"))


# ---------------------------------------------------------------------------
# Shard merging
# ---------------------------------------------------------------------------


def _merge_shards(
    worker_results: list[dict[str, dict]],
    all_keys: list[str],
    output_dir: Path,
    dtype: type,
    tmp_dir: Path,
) -> dict[str, dict[str, int]]:
    """Concatenate shard .bin + sidecar .npy files into final output."""
    counters: dict[str, dict[str, int]] = {k: {"seqs": 0, "tokens": 0} for k in all_keys}
    copy_buf = 64 * 1024 * 1024  # 64 MB

    for key in all_keys:
        # Collect shard info in worker order
        shard_prefixes: list[str] = []
        for wr in worker_results:
            info = wr[key]
            counters[key]["seqs"] += info["seqs"]
            counters[key]["tokens"] += info["tokens"]
            if info["seqs"] > 0:
                shard_prefixes.append(info["shard_prefix"])

        if not shard_prefixes:
            # No data for this key — skip instead of writing empty files
            continue

        # Concatenate .bin shards
        bin_path = get_bin_path(str(output_dir / key))
        with open(bin_path, "wb") as out_f:
            for sp in shard_prefixes:
                src = get_bin_path(sp)
                with open(src, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, copy_buf)

        # Load and merge sidecar .npy files
        all_seqlens: list[np.ndarray] = []
        all_docidx: list[np.ndarray] = []
        seq_offset = 0
        for i, sp in enumerate(shard_prefixes):
            sl = np.load(f"{sp}_seqlens.npy")
            di = np.load(f"{sp}_docidx.npy")
            all_seqlens.append(sl)
            if i == 0:
                all_docidx.append(di)
            else:
                # Drop leading 0, apply offset
                all_docidx.append(di[1:] + seq_offset)
            seq_offset += len(sl)

        merged_seqlens = np.concatenate(all_seqlens)
        merged_docidx = np.concatenate(all_docidx)

        _write_idx_file(
            get_idx_path(str(output_dir / key)),
            dtype,
            merged_seqlens,
            merged_docidx,
        )

    return counters


# ---------------------------------------------------------------------------
# Worker partitioning
# ---------------------------------------------------------------------------


def _partition_runs(
    run_lengths: np.ndarray, num_workers: int,
) -> list[tuple[int, int]]:
    """Partition runs into balanced chunks by cumulative clip count.

    Returns a list of (start_run, end_run) tuples — one per worker.
    """
    n_runs = len(run_lengths)
    total_clips = int(run_lengths.sum())
    clips_per_worker = total_clips // num_workers
    cum_clips = np.cumsum(run_lengths)

    boundaries = [0]
    for w in range(1, num_workers):
        target = w * clips_per_worker
        idx = int(np.searchsorted(cum_clips, target))
        boundaries.append(min(idx, n_runs))
    boundaries.append(n_runs)

    return [
        (boundaries[w], boundaries[w + 1])
        for w in range(num_workers)
        if boundaries[w] < boundaries[w + 1]
    ]


# ---------------------------------------------------------------------------
# Distribution formatting
# ---------------------------------------------------------------------------


def format_distribution(arr: np.ndarray, indent: str = "    ") -> list[str]:
    """Return lines with percentiles + histogram for a sequence-length array."""
    lines: list[str] = []
    lines.append(f"{indent}Mean:   {arr.mean():.1f}")
    lines.append(f"{indent}Median: {np.median(arr):.0f}")
    lines.append(f"{indent}Std:    {arr.std():.1f}")
    lines.append(f"{indent}Min:    {arr.min()}")
    lines.append(f"{indent}Max:    {arr.max()}")
    lines.append("")
    for p in _PERCENTILES:
        lines.append(f"{indent}P{p:02d}: {np.percentile(arr, p):.0f}")
    lines.append("")
    counts, _ = np.histogram(arr, bins=_HIST_BINS)
    lines.append(f"{indent}{'Bin':>12s}  {'Count':>12s}  {'Pct':>6s}  Cumulative")
    lines.append(f"{indent}{'-' * 12}  {'-' * 12}  {'-' * 6}  {'-' * 10}")
    cumul = 0
    for i, label in enumerate(_HIST_LABELS):
        cumul += counts[i]
        pct = counts[i] / len(arr) * 100
        cpct = cumul / len(arr) * 100
        bar = "#" * int(pct / 2)
        lines.append(
            f"{indent}{label:>12s}  {counts[i]:>12,}  {pct:>5.1f}%  {cpct:>5.1f}%  {bar}"
        )
    return lines


# ---------------------------------------------------------------------------
# Parquet loading & Arrow preparation helpers
# ---------------------------------------------------------------------------


def _load_cache_layout(parquet_dir: Path) -> dict[str, object] | None:
    layout_path = parquet_dir / "_CACHE_LAYOUT.json"
    if not layout_path.exists():
        return None
    with open(layout_path) as f:
        return json.load(f)


def list_interleave_cache_partitions(cache_dir: Path) -> list[Path]:
    """Return leaf cache directories to plan/build independently."""
    layout = _load_cache_layout(cache_dir)
    if not layout:
        return [cache_dir]

    if layout.get("version") != "v2":
        return [cache_dir]

    if any(p.is_dir() and p.name.startswith("rank_") for p in cache_dir.iterdir()):
        return [cache_dir]

    partition_dirs = sorted(
        p for p in cache_dir.iterdir()
        if p.is_dir()
        and (p / "_CACHE_LAYOUT.json").exists()
        and any(child.is_dir() and child.name.startswith("rank_") for child in p.iterdir())
    )
    if partition_dirs:
        return partition_dirs
    return [cache_dir]


def summarize_partition_stats(
    partition_stats: list[dict[str, int | str]],
    top_k: int = 5,
) -> dict[str, object]:
    """Summarize per-partition build stats for logs/metadata."""
    ordered = sorted(
        partition_stats,
        key=lambda s: (
            int(s["clips"]),
            int(s["runs"]),
            int(s.get("audio_tokens", 0)) + int(s.get("text_tokens", 0)),
        ),
        reverse=True,
    )
    return {
        "num_partitions": len(partition_stats),
        "total_clips": int(sum(int(s["clips"]) for s in partition_stats)),
        "total_runs": int(sum(int(s["runs"]) for s in partition_stats)),
        "total_sources": int(sum(int(s["sources"]) for s in partition_stats)),
        "total_audio_tokens": int(sum(int(s.get("audio_tokens", 0)) for s in partition_stats)),
        "total_text_tokens": int(sum(int(s.get("text_tokens", 0)) for s in partition_stats)),
        "top_partitions": ordered[:top_k],
    }


def print_partition_stats(partition_stats: list[dict[str, int | str]], top_k: int = 5) -> dict[str, object]:
    """Print a concise partition summary and return the structured payload."""
    summary = summarize_partition_stats(partition_stats, top_k=top_k)
    if not partition_stats:
        print("\nPartition summary: no partitions processed")
        return summary

    print(
        f"\nPartition summary: {summary['num_partitions']} partitions, "
        f"{summary['total_clips']:,} clips, {summary['total_runs']:,} runs, "
        f"{summary['total_audio_tokens'] + summary['total_text_tokens']:,} payload tokens"
    )
    print(f"Top {min(top_k, len(partition_stats))} partitions by clip count:")
    for stats in summary["top_partitions"]:
        total_tokens = int(stats.get("audio_tokens", 0)) + int(stats.get("text_tokens", 0))
        print(
            f"  {stats['name']}: clips={int(stats['clips']):,}, runs={int(stats['runs']):,}, "
            f"sources={int(stats['sources']):,}, tokens={total_tokens:,}, workers={int(stats['workers']):,}"
        )
    return summary


def load_interleave_cache(parquet_dir: Path) -> tuple[pl.DataFrame, object]:
    """Load interleave cache metadata and return the matching cache reader."""
    layout = _load_cache_layout(parquet_dir)
    if layout:
        version = layout.get("version")
        if version == "v2":
            if not any(p.is_dir() and p.name.startswith("rank_") for p in parquet_dir.iterdir()):
                raise RuntimeError(
                    f"{parquet_dir} is a partitioned v2 cache root. "
                    "Pass a leaf partition directory or use list_interleave_cache_partitions()."
                )
            reader = _V2InterleaveCacheReader(parquet_dir, layout)
            df = reader.load_metadata()
            print(f"\nFound {len(reader.chunks)} v2 cache chunks in {parquet_dir}")
            print("Cache layout: v2 (metadata parquet + token bins)")
            return df, reader
        raise RuntimeError(
            f"Unsupported interleave cache layout version at {parquet_dir / '_CACHE_LAYOUT.json'}: {version!r}"
        )

    reader = _V1InterleaveCacheReader(parquet_dir)
    print(f"\nFound {len(reader.parquet_files)} parquet files in {parquet_dir}")
    print("Cache layout: v1 (nested token parquet)")
    t0 = time.time()
    df = reader.load_metadata()
    elapsed = time.time() - t0
    print(f"Loaded {len(df):,} clips in {elapsed:.1f}s")
    return df, reader


def prepare_length_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """Return a metadata-only dataframe with token lengths for dry runs/planning."""
    if "audio_tokens" in df.columns and "text_tokens" in df.columns:
        return df.with_columns(
            df["audio_tokens"].list.len().cast(pl.UInt64).alias("_alen"),
            df["text_tokens"].list.len().cast(pl.UInt64).alias("_tlen"),
        ).select(["source_id", "clip_num", "_alen", "_tlen"])

    if "audio_token_length" in df.columns and "text_token_length" in df.columns:
        return df.select(
            "source_id",
            "clip_num",
            pl.col("audio_token_length").cast(pl.UInt64).alias("_alen"),
            pl.col("text_token_length").cast(pl.UInt64).alias("_tlen"),
        )

    raise ValueError("Unsupported interleave metadata schema: token lengths not available")


def prepare_interleave_cache_and_runs(
    df: pl.DataFrame,
    reader,
) -> tuple[PreparedInterleaveCache, np.ndarray, np.ndarray, int, int]:
    """Prepare a sorted cache view and run boundaries from interleave metadata."""
    print("\nDetecting consecutive runs ...")
    sorted_df, run_starts, run_lengths = _detect_runs(df)
    n_runs = len(run_starts)
    print(f"  {n_runs:,} runs")

    n_clips = len(sorted_df)
    n_sources = sorted_df["source_id"].n_unique()
    cache = reader.prepare(sorted_df)
    del sorted_df
    return cache, run_starts, run_lengths, n_clips, n_sources


def load_parquets(parquet_dir: Path) -> pl.DataFrame:
    """Load token parquets from a cache directory.

    Prefer the original tokenization cache naming scheme
    ``rank_*_chunk_*.parquet``. If none are present, fall back to any
    ``*.parquet`` files in the directory so repartitioned caches such as
    per-language ``part_*.parquet`` layouts remain compatible.
    """
    parquet_files = sorted(
        p
        for p in parquet_dir.glob("rank_*_chunk_*.parquet")
        if not p.name.endswith(".tmp")
    )
    if not parquet_files:
        parquet_files = sorted(
            p
            for p in parquet_dir.glob("*.parquet")
            if not p.name.endswith(".tmp")
        )
    print(f"\nFound {len(parquet_files)} parquet files in {parquet_dir}")

    t0 = time.time()
    df = pl.read_parquet(
        parquet_files,
        columns=["source_id", "clip_num", "audio_tokens", "text_tokens"],
    )
    elapsed = time.time() - t0
    print(f"Loaded {len(df):,} clips in {elapsed:.1f}s")
    return df


def prepare_arrow_and_runs(
    df: pl.DataFrame,
) -> tuple[pa.Array, pa.Array, np.ndarray, np.ndarray, int, int]:
    """Detect runs, extract Arrow arrays, free DataFrame.

    Returns (audio_arrow, text_arrow, run_starts, run_lengths, n_clips, n_sources).
    """
    print("\nDetecting consecutive runs ...")
    df, run_starts, run_lengths = _detect_runs(df)
    n_runs = len(run_starts)
    print(f"  {n_runs:,} runs")

    n_clips = len(df)
    n_sources = df["source_id"].n_unique()

    _audio_raw = df["audio_tokens"].to_arrow()
    _text_raw = df["text_tokens"].to_arrow()
    audio_arrow = _audio_raw.combine_chunks() if isinstance(_audio_raw, pa.ChunkedArray) else _audio_raw
    text_arrow = _text_raw.combine_chunks() if isinstance(_text_raw, pa.ChunkedArray) else _text_raw
    del _audio_raw, _text_raw

    # Free the polars DataFrame — token data lives in the Arrow arrays
    del df

    return audio_arrow, text_arrow, run_starts, run_lengths, n_clips, n_sources


# ---------------------------------------------------------------------------
# Transcribe-ratio adjustment
# ---------------------------------------------------------------------------


def compute_ratio_adjustment(
    il_per_run: np.ndarray,
    tr_per_run: np.ndarray,
    run_lengths: np.ndarray,
    target_ratio: float,
    seed: int = 42,
) -> set[int]:
    """Select multi-clip runs to convert to transcribe-only.

    When the natural transcribe ratio is below *target_ratio*, randomly
    convert multi-clip runs (each clip becomes an individual transcribe
    sequence) until the ratio is met.  If the ratio is already at or
    above the target, return an empty set (never downsample transcribe).

    Parameters
    ----------
    il_per_run : array of int
        Number of interleaved sequences produced per run.
    tr_per_run : array of int
        Number of transcribe sequences produced per run.
    run_lengths : array of int
        Number of clips in each run.
    target_ratio : float
        Desired minimum fraction of transcribe sequences (e.g. 0.1 = 10%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    set[int]
        Run indices to convert. Empty if ratio already met.
    """
    natural_il = int(il_per_run.sum())
    natural_tr = int(tr_per_run.sum())
    total = natural_il + natural_tr
    if total == 0:
        return set()

    current_ratio = natural_tr / total
    if current_ratio >= target_ratio:
        return set()

    # Candidates: multi-clip runs that produce interleaved sequences
    candidates = np.where((run_lengths >= 2) & (il_per_run > 0))[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    il = natural_il
    tr = natural_tr
    convert_set: set[int] = set()

    for r_idx in candidates:
        r = int(r_idx)
        # Converting this run: lose its interleaved seqs, gain one
        # transcribe seq per clip (replacing whatever it already produces)
        il -= int(il_per_run[r])
        tr += int(run_lengths[r]) - int(tr_per_run[r])
        convert_set.add(r)

        new_total = il + tr
        if new_total > 0 and tr / new_total >= target_ratio:
            break

    return convert_set
