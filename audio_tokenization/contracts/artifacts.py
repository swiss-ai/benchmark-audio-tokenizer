"""Stable artifact names shared across pipeline layers, plus filesystem
helpers for the v2 structured-cache layout.

A v2 chunk is a triplet ``(clips.{stem}.parquet, audio_tokens.{stem}.bin,
text_tokens.{stem}.bin)`` under a rank-local directory. The helpers below
are the single source of truth for stem extraction, completeness checks,
and orphan detection — so writer cleanup, reader enumeration, and
chunk-id assignment cannot drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


SHAR_INDEX_FILENAME = "shar_index.json"
SUCCESS_MARKER_FILE = "_SUCCESS"
INTERLEAVE_CACHE_LAYOUT_FILENAME = "_CACHE_LAYOUT.json"
INTERLEAVE_CACHE_LAYOUT_V2 = "v2"
INTERLEAVE_CACHE_SCHEMA_VERSION = 3
INTERLEAVE_CACHE_OUTPUT_STEM = f"interleave_cache_v{INTERLEAVE_CACHE_SCHEMA_VERSION}"


@dataclass(frozen=True)
class V2CacheChunk:
    """A complete v2 structured-cache chunk under a rank directory."""

    stem: str  # zero-padded chunk id, e.g. "000007"
    rank_dir: Path
    clips_path: Path
    audio_path: Path
    text_path: Path

    @property
    def stem_int(self) -> int:
        return int(self.stem)


def _stem_for(clips_path: Path) -> str:
    """Extract the chunk stem from a ``clips.{stem}.parquet`` filename."""
    return clips_path.name.split(".")[1]


def _sibling_bins(rank_dir: Path, stem: str) -> tuple[Path, Path]:
    return (
        rank_dir / f"audio_tokens.{stem}.bin",
        rank_dir / f"text_tokens.{stem}.bin",
    )


def validate_v2_chunks_complete(rank_dir: Path, *, error_label: str = "v2 structured cache") -> list[V2CacheChunk]:
    """Return all chunks; raise if any ``clips.*`` lacks its sibling bins."""
    chunks: list[V2CacheChunk] = []
    for clips_path in sorted(rank_dir.glob("clips.*.parquet")):
        stem = _stem_for(clips_path)
        audio_path, text_path = _sibling_bins(rank_dir, stem)
        if not audio_path.exists() or not text_path.exists():
            raise RuntimeError(
                f"Incomplete {error_label} chunk under {rank_dir}: "
                f"{clips_path.name} missing token bins"
            )
        chunks.append(
            V2CacheChunk(
                stem=stem,
                rank_dir=rank_dir,
                clips_path=clips_path,
                audio_path=audio_path,
                text_path=text_path,
            )
        )
    return chunks


def next_chunk_id(rank_dir: Path, *, commit_glob: str = "clips.*.parquet") -> int:
    """Monotonic chunk id for the next write into *rank_dir*."""
    if not rank_dir.exists():
        return 0
    stems: list[int] = []
    for clips_path in rank_dir.glob(commit_glob):
        try:
            stems.append(int(_stem_for(clips_path)))
        except (IndexError, ValueError):
            continue
    return max(stems) + 1 if stems else 0


def prune_orphan_bin_files(
    rank_dir: Path,
    *,
    commit_glob: str = "clips.*.parquet",
    payload_globs: tuple[str, ...] = ("audio_tokens.*.bin", "text_tokens.*.bin"),
    logger: logging.Logger | None = None,
    label: str = "structured cache payload with no commit marker",
) -> list[Path]:
    """Delete token payloads whose commit marker is missing.

    Such files come from a writer crash between the bin commit and the
    clips-parquet commit; without the parquet they cannot be associated
    with any chunk. Returns the deleted paths.
    """
    committed_stems = {_stem_for(p) for p in rank_dir.glob(commit_glob)}
    removed: list[Path] = []
    payload_paths = [
        path
        for payload_glob in payload_globs
        for path in rank_dir.glob(payload_glob)
    ]
    for bin_path in payload_paths:
        try:
            stem = bin_path.name.split(".")[1]
        except IndexError:
            continue
        if stem in committed_stems:
            continue
        if logger is not None:
            logger.warning("Removing orphan %s: %s", label, bin_path.name)
        bin_path.unlink()
        removed.append(bin_path)
    return removed
