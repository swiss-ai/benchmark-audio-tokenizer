"""Small durable IO helpers shared across pipeline stages."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Iterable

from audio_tokenization.contracts.artifacts import SUCCESS_MARKER_FILE


def fsync_file(path: str | Path) -> None:
    """Flush an existing file path to stable storage."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: str | Path) -> None:
    """Flush a directory entry after an atomic rename."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> None:
    """Write JSON via tmp + fsync + atomic rename.

    Lustre/client write-back can leave zero-byte or stale files after node
    failure unless both data and directory metadata are flushed before/after
    the rename.
    """
    final_path = Path(path)
    tmp_path = final_path.with_suffix(f"{final_path.suffix}.tmp.{os.getpid()}")
    try:
        data = json.dumps(payload, indent=indent, sort_keys=sort_keys, default=str)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        fsync_dir(final_path.parent)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def atomic_replace_files(
    replacements: Iterable[tuple[str | Path, str | Path]],
    *,
    fsync_sources: bool = True,
) -> None:
    """Publish multiple temp files with a shared durable replace sequence.

    This is not a true multi-file transaction, but all chunk writers should use
    the same fsync + replace + directory-fsync pattern instead of open-coding it.
    """
    pairs = [(Path(src), Path(dst)) for src, dst in replacements]
    if fsync_sources:
        for src, _ in pairs:
            fsync_file(src)
    for src, dst in pairs:
        os.replace(src, dst)
    for parent in {dst.parent for _, dst in pairs}:
        fsync_dir(parent)


def cleanup_tmp_files(
    directory: str | Path,
    pattern: str = "*.tmp",
    *,
    logger: Any | None = None,
    label: str = "stale temp file",
) -> list[Path]:
    """Remove temp files matching ``pattern`` and return the removed paths."""
    root = Path(directory)
    if not root.exists():
        return []
    removed: list[Path] = []
    for tmp in sorted(root.glob(pattern)):
        if logger is not None:
            logger.warning("Removing %s: %s", label, tmp.name)
        tmp.unlink()
        removed.append(tmp)
    return removed


class AtomicStreamingWrite:
    """Streaming temp-file writer that commits with an atomic rename on close."""

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "wb",
        compression: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        self.mode = mode
        self.compression = compression
        self._file = None
        self._closed = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = _open_streaming_path(
            self.tmp_path,
            mode=self.mode,
            compression=self.compression,
        )
        return self._file

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.commit()
        else:
            self.abort()
        return False

    def commit(self) -> None:
        """Close, fsync, and atomically publish the temp file."""
        if self._closed:
            return
        self.close_temp()
        os.replace(self.tmp_path, self.path)
        fsync_dir(self.path.parent)
        self._closed = True

    def close_temp(self) -> None:
        """Close and fsync the temp file without publishing it."""
        self._close_file()
        fsync_file(self.tmp_path)

    def mark_committed(self) -> None:
        """Mark the stream closed after an external atomic replace."""
        self._closed = True

    def abort(self) -> None:
        """Close and remove the temp file without publishing it."""
        if self._closed:
            return
        self._close_file()
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self._closed = True

    def _close_file(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()


def atomic_streaming_write(
    path: str | Path,
    *,
    mode: str = "wb",
    compression: str | None = None,
) -> AtomicStreamingWrite:
    """Return a streaming writer that atomically commits on context exit.

    Use this for large line-oriented sidecars where whole-payload helpers would
    keep unnecessary data in memory. The temp path is ``<final>.tmp`` so stale
    chunk cleanup can sweep it with the same rank-local ``*.tmp`` pattern.
    """
    return AtomicStreamingWrite(path, mode=mode, compression=compression)


def write_success_marker(directory: str | Path) -> None:
    """Atomically write the canonical ``_SUCCESS`` marker into *directory*.

    Single primitive shared by stage-level (``run_stage``) and partition-level
    (per-worker, per-chunk, per-product cache) completion sites.
    """
    with atomic_streaming_write(Path(directory) / SUCCESS_MARKER_FILE, mode="w") as f:
        f.write("ok\n")


def _open_streaming_path(path: Path, *, mode: str, compression: str | None):
    if compression is None:
        return open(path, mode)
    if compression == "gz":
        import gzip

        return gzip.open(path, mode)
    if compression == "zst":
        import zstandard

        return zstandard.open(path, mode)
    raise ValueError(f"Unsupported compression: {compression!r}")


def open_compressed(path: str | Path, mode: str = "rb"):
    """Open plain, gzip, or zstandard-compressed files."""
    p = Path(path)
    if p.suffix == ".gz":
        import gzip

        return gzip.open(p, mode)
    if p.suffix == ".zst":
        import zstandard

        if "b" in mode:
            raw = zstandard.open(p, mode)
            if "r" in mode:
                return io.BufferedReader(raw)
            return raw
        return zstandard.open(p, mode)
    return open(p, mode)
