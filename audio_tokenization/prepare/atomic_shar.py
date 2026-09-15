"""Publish a structurally validated SHAR partition with one directory rename."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


# JSON in the existing index/planning schemas. Lhotse in_dir discovery ignores
# .idx sidecars; ordinary .json files here would be interpreted as SHAR fields.
PARTITION_INDEX_FILE = "shar_index.idx"
PARTITION_MANIFEST_FILE = "_shar_work_manifest.idx"


def _partition_fields(directory: Path, fields: Iterable[str]) -> dict[str, dict[str, Path]]:
    """Group standard SHAR filenames by field and shard, ignoring index sidecars."""
    grouped: dict[str, dict[str, Path]] = {name: {} for name in ("cuts", *fields)}
    suffixes = (".jsonl.gz", ".jsonl", ".tar.gz", ".tar")
    for path in directory.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Unexpected non-file SHAR output: {path}")
        suffix = next((s for s in suffixes if path.name.endswith(s)), None)
        if suffix is None:
            if path.name.endswith(".idx"):
                continue
            raise ValueError(f"Unexpected SHAR output: {path}")
        stem = path.name[:-len(suffix)]
        field, separator, shard = stem.partition(".")
        if field not in grouped or (separator and not shard.isdigit()):
            raise ValueError(f"Unexpected SHAR field/shard: {path}")
        if field == "cuts" and suffix not in (".jsonl", ".jsonl.gz"):
            raise ValueError(f"Cuts must be JSONL manifests: {path}")
        if shard in grouped[field]:
            raise ValueError(f"Duplicate SHAR field/shard: {path}")
        grouped[field][shard] = path
    return grouped


def _validate_partition(directory: Path, fields: Iterable[str]) -> None:
    from audio_tokenization.prepare.validate_shar import _validate_structural_shard
    from audio_tokenization.pipelines.lhotse.planning import write_shar_work_manifest
    from audio_tokenization.utils.io import atomic_write_json

    grouped = _partition_fields(directory, fields)
    cuts = grouped["cuts"]
    for field, shards in grouped.items():
        if set(shards) != set(cuts):
            raise ValueError(f"SHAR field {field!r} has different shards from cuts in {directory}")
    validated_stats = {}
    for shard, cuts_path in sorted(cuts.items()):
        stats = {}
        _validate_structural_shard(
            shard_name=cuts_path.name,
            slice_fields={field: [str(paths[shard])] for field, paths in grouped.items()},
            planning_stats=stats,
        )
        validated_stats[str(cuts_path.resolve())] = stats
    if cuts:
        atomic_write_json(directory / PARTITION_INDEX_FILE, {
            "version": 1,
            "fields": {field: [paths[shard].name for shard in sorted(cuts)]
                       for field, paths in grouped.items()},
        })
        write_shar_work_manifest(
            directory, index_name=PARTITION_INDEX_FILE,
            manifest_filename=PARTITION_MANIFEST_FILE, validated_stats=validated_stats,
        )


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_shar_partition(output_dir: Path, *, fields: Iterable[str]) -> Iterator[Path]:
    """Yield a private sibling, then validate, sync and publish the closed output.

    The caller must close all writers before leaving this context. The final
    directory may be absent or empty (the existing prepare setup creates it).
    Existing directory permissions are preserved. New destinations retain the
    private staging mode. Nonempty destinations are never replaced; the stage
    owns overwrite policy.
    Worker stats and success markers are written only after this returns.
    """
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or (output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    )):
        raise FileExistsError(f"Refusing to replace nonempty SHAR partition: {output_dir}")
    output_mode = stat.S_IMODE(output_dir.stat().st_mode) if output_dir.exists() else None
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        yield staged
        _validate_partition(staged, tuple(fields))
        for path in staged.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        if output_mode is not None:
            staged.chmod(output_mode)
        _fsync_directory(staged)
        os.replace(staged, output_dir)
        _fsync_directory(output_dir.parent)
    finally:
        # Only this invocation's temporary output is ours to clean up. A
        # crashed sibling invocation may leave another staging dir for review.
        if staged.exists():
            shutil.rmtree(staged)
