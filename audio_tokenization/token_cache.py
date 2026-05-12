"""Asset-level audio-token cache shared by tokenization and materialization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from audio_tokenization.contracts.artifacts import next_chunk_id, prune_orphan_bin_files
from audio_tokenization.utils.io import atomic_replace_files, atomic_write_json, fsync_file


MANIFEST_FILENAME = "_MANIFEST.json"
AUDIO_TOKEN_CACHE_FORMAT = "audio_token_cache_v1"

_INDEX_SCHEMA = pa.schema([
    ("audio_id", pa.string()),
    ("token_file", pa.string()),
    ("token_offset", pa.int64()),
    ("token_count", pa.int32()),
    ("duration_sec", pa.float64()),
])


def audio_tokenizer_fingerprint(tokenizer_path: str | Path) -> str:
    """Fingerprint the tokenizer fields that define audio token IDs."""
    path = Path(tokenizer_path)
    mapping_path = path / "audio_token_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"audio token cache requires {mapping_path} to validate tokenizer identity"
        )
    # The audio cache stores already-offset token IDs, so this mapping is the
    # minimum semantic identity needed to prevent consuming a cache with the
    # wrong audio marker IDs or offset.
    digest = hashlib.sha256()
    digest.update(mapping_path.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(mapping_path.read_bytes())
    return digest.hexdigest()


def build_audio_token_cache_manifest(
    *,
    tokenizer_path: str | Path,
    vocab_size: int | None,
    token_dtype: str = "int32",
) -> dict[str, Any]:
    path = Path(tokenizer_path)
    return {
        "format": AUDIO_TOKEN_CACHE_FORMAT,
        "token_dtype": str(token_dtype),
        "token_offset_unit": "bytes",
        "tokenizer_path": str(path),
        "tokenizer_fingerprint": audio_tokenizer_fingerprint(path),
        "tokenizer_fingerprint_files": ["audio_token_mapping.json"],
        "vocab_size": None if vocab_size is None else int(vocab_size),
    }


def write_audio_token_cache_manifest(
    cache_dir: str | Path,
    *,
    tokenizer_path: str | Path,
    vocab_size: int | None,
    token_dtype: str = "int32",
) -> None:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        root / MANIFEST_FILENAME,
        build_audio_token_cache_manifest(
            tokenizer_path=tokenizer_path,
            vocab_size=vocab_size,
            token_dtype=token_dtype,
        ),
    )


def read_audio_token_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(cache_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Audio token cache manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Audio token cache manifest must be a JSON object: {manifest_path}")
    return payload


def validate_audio_token_cache_manifest(
    cache_dir: str | Path,
    *,
    tokenizer_path: str | Path,
    token_dtype: str = "int32",
) -> dict[str, Any]:
    manifest = read_audio_token_cache_manifest(cache_dir)
    if manifest.get("format") != AUDIO_TOKEN_CACHE_FORMAT:
        raise ValueError(
            f"Audio token cache manifest has unsupported format "
            f"{manifest.get('format')!r}; expected {AUDIO_TOKEN_CACHE_FORMAT!r}"
        )
    if manifest.get("token_dtype") != token_dtype:
        raise ValueError(
            f"Audio token cache token dtype mismatch: "
            f"{manifest.get('token_dtype')!r} != {token_dtype!r}"
        )
    expected = audio_tokenizer_fingerprint(tokenizer_path)
    actual = manifest.get("tokenizer_fingerprint")
    if actual != expected:
        raise ValueError(
            "Audio token cache tokenizer fingerprint mismatch: "
            f"{actual!r} != {expected!r}"
        )
    return manifest


@dataclass(frozen=True)
class AudioTokenSpan:
    audio_id: str
    token_file: str
    token_offset: int
    token_count: int
    duration_sec: float | None = None


class AudioTokenCache:
    def __init__(self, root: Path, spans: dict[str, AudioTokenSpan]):
        self.root = root
        self._spans = spans
        self._memmaps: dict[str, np.memmap] = {}

    @cached_property
    def audio_ids(self) -> set[str]:
        return set(self._spans)

    def span(self, audio_id: str) -> AudioTokenSpan:
        try:
            return self._spans[audio_id]
        except KeyError as exc:
            raise KeyError(f"audio_id {audio_id!r} is missing from audio token cache") from exc

    def read(self, audio_id: str) -> np.ndarray:
        span = self.span(audio_id)
        mm = self._memmap(span.token_file)
        start = span.token_offset // np.dtype(np.int32).itemsize
        end = start + span.token_count
        return np.asarray(mm[start:end], dtype=np.int32)

    def _memmap(self, token_file: str) -> np.memmap:
        mm = self._memmaps.get(token_file)
        if mm is None:
            mm = np.memmap(self.root / token_file, dtype=np.int32, mode="r")
            self._memmaps[token_file] = mm
        return mm


class AudioTokenCacheWriter:
    """Write ``audio_id -> token span`` rows plus flat int32 token payloads."""

    def __init__(self, output_dir: str | Path, *, rank: int, chunk_id: int = 0):
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.rank_dir = self.output_dir / f"rank_{self.rank:04d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        prune_orphan_bin_files(
            self.rank_dir,
            commit_glob="audio_index.*.parquet",
            payload_globs=("audio_tokens.*.bin",),
            label="audio token cache payload with no index",
        )
        self.chunk_id = max(
            int(chunk_id),
            next_chunk_id(self.rank_dir, commit_glob="audio_index.*.parquet"),
        )
        self._rows: list[dict[str, Any]] = []
        self._seen_audio_ids: set[str] = set()
        self._token_offset = 0
        self._opened = False
        self._token_tmp_path: Path | None = None
        self._index_tmp_path: Path | None = None
        self._token_final_path: Path | None = None
        self._index_final_path: Path | None = None
        self._fh = None

    @property
    def num_rows(self) -> int:
        return len(self._rows)

    def add(
        self,
        *,
        audio_id: str,
        tokens: list[int] | np.ndarray,
        duration_sec: float | None,
    ) -> None:
        if not audio_id:
            raise ValueError("audio_id must be non-empty")
        if audio_id in self._seen_audio_ids:
            raise ValueError(f"Duplicate audio_id within chunk: {audio_id!r}")
        if not self._opened:
            self._open()
        assert self._fh is not None
        assert self._token_final_path is not None

        arr = np.ascontiguousarray(np.asarray(tokens, dtype=np.int32))
        if arr.ndim != 1:
            raise ValueError(f"audio tokens for {audio_id!r} must be a 1-D sequence")
        if arr.size == 0:
            raise ValueError(f"audio tokens for {audio_id!r} are empty")
        arr.tofile(self._fh)
        self._rows.append({
            "audio_id": str(audio_id),
            "token_file": str(self._token_final_path.relative_to(self.output_dir)),
            "token_offset": self._token_offset,
            "token_count": int(arr.shape[0]),
            "duration_sec": None if duration_sec is None else float(duration_sec),
        })
        self._seen_audio_ids.add(audio_id)
        self._token_offset += arr.nbytes

    def finalize(self) -> int:
        if not self._opened:
            return self.chunk_id
        assert self._fh is not None
        assert self._token_tmp_path is not None
        assert self._index_tmp_path is not None
        assert self._token_final_path is not None
        assert self._index_final_path is not None

        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()

        table = pa.Table.from_pylist(self._rows, schema=_INDEX_SCHEMA)
        pq.write_table(table, self._index_tmp_path, compression="zstd")
        fsync_file(self._index_tmp_path)
        atomic_replace_files(
            [
                (self._token_tmp_path, self._token_final_path),
                (self._index_tmp_path, self._index_final_path),
            ],
            fsync_sources=False,
        )

        finalized = self.chunk_id
        self.chunk_id += 1
        self._rows = []
        self._seen_audio_ids.clear()
        self._token_offset = 0
        self._opened = False
        self._fh = None
        self._token_tmp_path = None
        self._index_tmp_path = None
        self._token_final_path = None
        self._index_final_path = None
        return finalized

    def get_state(self) -> int:
        return self.chunk_id

    def _open(self) -> None:
        stem = f"{self.chunk_id:06d}"
        self._token_tmp_path = self.rank_dir / f"audio_tokens.{stem}.bin.tmp"
        self._index_tmp_path = self.rank_dir / f"audio_index.{stem}.parquet.tmp"
        self._token_final_path = self.rank_dir / f"audio_tokens.{stem}.bin"
        self._index_final_path = self.rank_dir / f"audio_index.{stem}.parquet"
        self._fh = open(self._token_tmp_path, "wb")
        self._opened = True


def load_audio_token_cache(cache_dir: str | Path) -> AudioTokenCache:
    root = Path(cache_dir)
    index_paths = sorted(root.glob("rank_*/audio_index.*.parquet"))
    if not index_paths:
        raise FileNotFoundError(f"No audio token cache index files found under {root}")

    spans: dict[str, AudioTokenSpan] = {}
    for index_path in index_paths:
        table = pq.read_table(index_path, schema=_INDEX_SCHEMA)
        if table.num_rows == 0:
            continue

        audio_ids = table["audio_id"].combine_chunks().to_numpy(zero_copy_only=False)
        token_files = table["token_file"].combine_chunks().to_numpy(zero_copy_only=False)
        token_offsets = table["token_offset"].combine_chunks().to_numpy(zero_copy_only=False)
        token_counts = table["token_count"].combine_chunks().to_numpy(zero_copy_only=False)
        durations = table["duration_sec"].combine_chunks().to_numpy(zero_copy_only=False)

        duplicate = _first_duplicate(audio_ids, existing=spans)
        if duplicate is not None:
            raise ValueError(f"Duplicate audio_id in audio token cache: {duplicate!r}")

        for token_file, required_bytes in sorted(
            _required_token_file_sizes(
                token_files=token_files,
                token_offsets=token_offsets,
                token_counts=token_counts,
            ).items()
        ):
            token_path = root / token_file
            if not token_path.is_file():
                raise FileNotFoundError(
                    f"Audio token cache index {index_path} points to missing token file {token_path}"
                )
            actual_bytes = token_path.stat().st_size
            if actual_bytes < required_bytes:
                raise ValueError(
                    f"Audio token cache file {token_path} is shorter than its "
                    f"audio-token index requires: {actual_bytes} < {required_bytes} bytes"
                )

        spans.update(
            _build_spans_from_columns(
                audio_ids=audio_ids,
                token_files=token_files,
                token_offsets=token_offsets,
                token_counts=token_counts,
                durations=durations,
            )
        )
    return AudioTokenCache(root, spans)


def _first_duplicate(audio_ids, *, existing: dict[str, AudioTokenSpan]) -> str | None:
    seen = set(existing)
    for audio_id in audio_ids:
        if audio_id in seen:
            return str(audio_id)
        seen.add(audio_id)
    return None


def _build_spans_from_columns(
    *,
    audio_ids,
    token_files,
    token_offsets,
    token_counts,
    durations,
) -> dict[str, AudioTokenSpan]:
    return {
        str(audio_id): AudioTokenSpan(
            audio_id=str(audio_id),
            token_file=str(token_file),
            token_offset=int(token_offset),
            token_count=int(token_count),
            duration_sec=_coerce_optional_float(duration),
        )
        for audio_id, token_file, token_offset, token_count, duration in zip(
            audio_ids,
            token_files,
            token_offsets,
            token_counts,
            durations,
            strict=True,
        )
    }


def _required_token_file_sizes(*, token_files, token_offsets, token_counts) -> dict[str, int]:
    itemsize = np.dtype(np.int32).itemsize
    sizes: dict[str, int] = {}
    for token_file, token_offset, token_count in zip(
        token_files,
        token_offsets,
        token_counts,
        strict=True,
    ):
        key = str(token_file)
        end = int(token_offset) + int(token_count) * itemsize
        sizes[key] = max(sizes.get(key, 0), end)
    return sizes


def _coerce_optional_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value):
        return None
    return value
