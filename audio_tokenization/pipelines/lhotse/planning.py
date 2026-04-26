"""Rank-independent SHAR work manifests and rank-specific tokenize assignment.

The tokenizer runtime should not infer production layout from today's GPU
count. This module separates:

- a SHAR work manifest: stable work units derived from SHAR metadata only
- a tokenize assignment: deterministic mapping from work units to launch ranks

No audio is decoded here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.contracts.artifacts import SHAR_INDEX_FILENAME
from audio_tokenization.utils.io import atomic_write_json, open_compressed

from .data import _resolve_index_paths, resolve_shar_dirs


SHAR_WORK_MANIFEST_FILE = "_shar_work_manifest.json"
TOKENIZE_ASSIGNMENT_FILE = "_tokenize_assignment.json"
PLANNING_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TokenizeFilter:
    """Subset of tokenization filters that can be estimated from cut metadata.

    This intentionally excludes runtime-only transformations such as peak
    normalization. The planner must not decode audio or instantiate tokenizer
    state; it only uses fields already present in SHAR cut manifests.
    """

    min_duration: float | None = None
    max_duration: float | None = None
    min_sample_rate: int | None = None
    min_rms_db: float | None = None

    @classmethod
    def from_spec(cls, spec: TokenizeSpec) -> "TokenizeFilter":
        filt = spec.filter
        return cls(
            min_duration=_optional_float(filt.min_duration),
            max_duration=_optional_float(filt.max_duration),
            min_sample_rate=_optional_int(filt.min_sample_rate),
            min_rms_db=_optional_float(filt.min_rms_db),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "min_sample_rate": self.min_sample_rate,
            "min_rms_db": self.min_rms_db,
        }


@dataclass(frozen=True)
class SharWorkUnit:
    """One indivisible SHAR scheduling unit.

    A work unit corresponds to one cut shard plus the same-position companion
    SHAR fields from ``shar_index.json`` (recordings, supervisions, custom
    fields, etc.). Keeping all fields aligned is what lets a rank pass the
    unit straight to ``CutSet.from_shar(fields=...)`` without rebuilding a
    partial index.
    """

    work_unit_id: str
    shar_dir: str
    shard_index: int
    fields: dict[str, list[str]]
    cut_count: int
    duration_sec: float
    min_duration_sec: float | None
    max_duration_sec: float | None
    rms_db_count: int
    sample_rate_count: int
    source_id_count: int
    clip_num_count: int
    clip_start_count: int
    clip_duration_count: int
    min_rms_db: float | None
    max_rms_db: float | None
    min_sample_rate: int | None
    max_sample_rate: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "work_unit_id": self.work_unit_id,
            "shar_dir": self.shar_dir,
            "shard_index": self.shard_index,
            "fields": self.fields,
            "cut_count": self.cut_count,
            "duration_sec": self.duration_sec,
            "min_duration_sec": self.min_duration_sec,
            "max_duration_sec": self.max_duration_sec,
            "rms_db_count": self.rms_db_count,
            "sample_rate_count": self.sample_rate_count,
            "source_id_count": self.source_id_count,
            "clip_num_count": self.clip_num_count,
            "clip_start_count": self.clip_start_count,
            "clip_duration_count": self.clip_duration_count,
            "min_rms_db": self.min_rms_db,
            "max_rms_db": self.max_rms_db,
            "min_sample_rate": self.min_sample_rate,
            "max_sample_rate": self.max_sample_rate,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SharWorkUnit":
        return cls(
            work_unit_id=str(payload["work_unit_id"]),
            shar_dir=str(payload["shar_dir"]),
            shard_index=int(payload["shard_index"]),
            fields={str(k): list(v) for k, v in dict(payload["fields"]).items()},
            cut_count=int(payload["cut_count"]),
            duration_sec=float(payload["duration_sec"]),
            min_duration_sec=_optional_float(payload.get("min_duration_sec")),
            max_duration_sec=_optional_float(payload.get("max_duration_sec")),
            rms_db_count=int(payload["rms_db_count"]),
            sample_rate_count=int(payload["sample_rate_count"]),
            source_id_count=int(payload["source_id_count"]),
            clip_num_count=int(payload["clip_num_count"]),
            clip_start_count=int(payload["clip_start_count"]),
            clip_duration_count=int(payload["clip_duration_count"]),
            min_rms_db=_optional_float(payload.get("min_rms_db")),
            max_rms_db=_optional_float(payload.get("max_rms_db")),
            min_sample_rate=_optional_int(payload.get("min_sample_rate")),
            max_sample_rate=_optional_int(payload.get("max_sample_rate")),
        )


@dataclass(frozen=True)
class SharWorkManifest:
    """Rank-independent description of a prepared SHAR dataset.

    Durable manifests describe raw work. Tokenization filters are runtime
    policy: they validate metadata coverage, but do not alter assignment cost.
    """

    input_shar_dirs: list[str]
    shar_index_filename: str
    work_units: list[SharWorkUnit]

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": PLANNING_SCHEMA_VERSION,
            "input_shar_dirs": self.input_shar_dirs,
            "shar_index_filename": self.shar_index_filename,
            "work_units": [
                {
                    "work_unit_id": u.work_unit_id,
                    "fields": u.fields,
                    "cut_count": u.cut_count,
                    "duration_sec": u.duration_sec,
                    "rms_db_count": u.rms_db_count,
                    "sample_rate_count": u.sample_rate_count,
                    "source_id_count": u.source_id_count,
                    "clip_num_count": u.clip_num_count,
                    "clip_start_count": u.clip_start_count,
                    "clip_duration_count": u.clip_duration_count,
                }
                for u in self.work_units
            ],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": PLANNING_SCHEMA_VERSION,
            "kind": "shar_work_manifest",
            "input_shar_dirs": self.input_shar_dirs,
            "shar_index_filename": self.shar_index_filename,
            "fingerprint": self.fingerprint,
            "total_work_units": len(self.work_units),
            "total_cut_count": sum(u.cut_count for u in self.work_units),
            "total_duration_sec": sum(u.duration_sec for u in self.work_units),
            "rms_db_count": sum(u.rms_db_count for u in self.work_units),
            "sample_rate_count": sum(u.sample_rate_count for u in self.work_units),
            "source_id_count": sum(u.source_id_count for u in self.work_units),
            "clip_num_count": sum(u.clip_num_count for u in self.work_units),
            "clip_start_count": sum(u.clip_start_count for u in self.work_units),
            "clip_duration_count": sum(u.clip_duration_count for u in self.work_units),
            "work_units": [u.to_json() for u in self.work_units],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SharWorkManifest":
        if payload.get("schema_version") != PLANNING_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SHAR work manifest schema: {payload.get('schema_version')!r}"
            )
        return cls(
            input_shar_dirs=list(payload["input_shar_dirs"]),
            shar_index_filename=str(payload["shar_index_filename"]),
            work_units=[
                SharWorkUnit.from_json(item)
                for item in list(payload["work_units"])
            ],
        )


@dataclass(frozen=True)
class RankAssignment:
    """Launch-specific work assigned to one rank."""

    rank: int
    active: bool
    work_unit_ids: list[str]
    fields: dict[str, list[str]]
    cut_count: int
    duration_sec: float

    def to_json(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "active": self.active,
            "work_unit_ids": self.work_unit_ids,
            "fields": self.fields,
            "cut_count": self.cut_count,
            "duration_sec": self.duration_sec,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "RankAssignment":
        return cls(
            rank=int(payload["rank"]),
            active=bool(payload["active"]),
            work_unit_ids=list(payload["work_unit_ids"]),
            fields={str(k): list(v) for k, v in dict(payload["fields"]).items()},
            cut_count=int(payload["cut_count"]),
            duration_sec=float(payload["duration_sec"]),
        )


@dataclass(frozen=True)
class TokenizeAssignment:
    """Deterministic mapping from SHAR work units to ranks for one launch.

    This object is intentionally not a dataset artifact: it depends on
    ``world_size``. It is written under the tokenization
    output directory so resume/debug can inspect exactly what each rank owned.
    """

    world_size: int
    active_ranks: int
    manifest_fingerprint: str
    assignments: list[RankAssignment]

    def assignment_for_rank(self, rank: int) -> RankAssignment:
        for assignment in self.assignments:
            if assignment.rank == rank:
                return assignment
        raise ValueError(f"No tokenize assignment for rank {rank}")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": PLANNING_SCHEMA_VERSION,
            "kind": "tokenize_assignment",
            "world_size": self.world_size,
            "active_ranks": self.active_ranks,
            "manifest_fingerprint": self.manifest_fingerprint,
            "assignments": [a.to_json() for a in self.assignments],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TokenizeAssignment":
        return cls(
            world_size=int(payload["world_size"]),
            active_ranks=int(payload["active_ranks"]),
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
            assignments=[
                RankAssignment.from_json(item)
                for item in list(payload["assignments"])
            ],
        )


@dataclass
class _Bin:
    rank: int
    work_units: list[str]
    duration: float
    cut_count: int
    fields: dict[str, list[str]]


def build_shar_work_manifest(
    shar_dir: str | list[str],
    *,
    index_name: str = SHAR_INDEX_FILENAME,
    tokenize_filter: TokenizeFilter | None = None,
    require_interleave_ids: bool = False,
) -> SharWorkManifest:
    """Build a SHAR work manifest by scanning cut manifests only.

    The manifest is filter-independent: estimated fields intentionally use
    unfiltered durations for deterministic, cheap rank assignment. The
    optional filter is only used to validate required metadata coverage.
    """
    raw_manifest = _build_shar_work_manifest_unfiltered(
        shar_dir,
        index_name=index_name,
    )
    _assert_manifest_coverage(
        raw_manifest,
        tokenize_filter=tokenize_filter,
        require_interleave_ids=require_interleave_ids,
    )
    return raw_manifest


def _build_shar_work_manifest_unfiltered(
    shar_dir: str | list[str],
    *,
    index_name: str,
) -> SharWorkManifest:
    resolved_dirs = resolve_shar_dirs(shar_dir, index_name=index_name)
    planned_units: list[tuple[Path, int, dict[str, list[str]], Path]] = []

    for sd in resolved_dirs:
        shar_path = Path(sd)
        index_path = shar_path / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"SHAR index not found: {index_path}")
        with open(index_path) as f:
            fields = json.load(f).get("fields", {})
        if "cuts" not in fields:
            raise ValueError(f"Shar index missing required 'cuts' field: {index_path}")
        resolved_fields = _resolve_index_paths(shar_path, fields)
        _validate_parallel_fields(index_path, resolved_fields)

        cut_paths = resolved_fields["cuts"]
        for shard_index, cut_path in enumerate(cut_paths):
            unit_fields = {
                field: [paths[shard_index]]
                for field, paths in resolved_fields.items()
            }
            planned_units.append((shar_path, shard_index, unit_fields, Path(cut_path)))

    stats_by_unit = _scan_cut_manifests(
        [cut_path for _shar_path, _shard_index, _unit_fields, cut_path in planned_units],
    )
    work_units = [
        SharWorkUnit(
            work_unit_id=_work_unit_id(shar_path, cut_path, shard_index),
            shar_dir=str(shar_path),
            shard_index=shard_index,
            fields=unit_fields,
            **stats,
        )
        for (shar_path, shard_index, unit_fields, cut_path), stats in zip(planned_units, stats_by_unit)
    ]
    work_units.sort(key=lambda u: u.work_unit_id)
    return SharWorkManifest(
        input_shar_dirs=resolved_dirs,
        shar_index_filename=index_name,
        work_units=work_units,
    )


def write_shar_work_manifest(
    output_dir: str | Path,
    shar_dir: str | list[str] | None = None,
    *,
    index_name: str = SHAR_INDEX_FILENAME,
) -> SharWorkManifest:
    """Build and write the durable, filter-independent SHAR work manifest."""
    output_dir = Path(output_dir)
    manifest = build_shar_work_manifest(
        str(output_dir) if shar_dir is None else shar_dir,
        index_name=index_name,
        tokenize_filter=None,
    )
    atomic_write_json(output_dir / SHAR_WORK_MANIFEST_FILE, manifest.to_json())
    return manifest


def read_shar_work_manifest(path: str | Path) -> SharWorkManifest:
    """Read a durable SHAR work manifest from a file path or SHAR directory."""
    path = Path(path)
    manifest_path = path / SHAR_WORK_MANIFEST_FILE if path.is_dir() else path
    payload = json.loads(manifest_path.read_text())
    return SharWorkManifest.from_json(payload)


def load_or_build_shar_work_manifest(
    shar_dir: str | list[str],
    *,
    index_name: str = SHAR_INDEX_FILENAME,
    tokenize_filter: TokenizeFilter | None = None,
    require_interleave_ids: bool = False,
) -> tuple[SharWorkManifest, str]:
    """Load a durable SHAR manifest when available, otherwise build one.

    Returns ``(manifest, source)`` where source is ``"manifest"`` or
    ``"scan"``. Tokenize filters are runtime policy and are not folded into
    rank assignment by default; assignment uses the durable unfiltered manifest
    durations, while filters still trigger fail-fast metadata coverage checks.
    """
    raw_manifest = _load_existing_manifest(shar_dir, index_name=index_name)
    source = "manifest"
    if raw_manifest is None:
        return (
            build_shar_work_manifest(
                shar_dir,
                index_name=index_name,
                tokenize_filter=tokenize_filter,
                require_interleave_ids=require_interleave_ids,
            ),
            "scan",
        )

    _assert_manifest_coverage(
        raw_manifest,
        tokenize_filter=tokenize_filter,
        require_interleave_ids=require_interleave_ids,
    )
    return raw_manifest, source


def build_tokenize_assignment(
    manifest: SharWorkManifest,
    *,
    world_size: int,
) -> TokenizeAssignment:
    """Assign SHAR work units to launch ranks by estimated audio duration.

    The algorithm is greedy bin packing over whole SHAR work units. Whole-unit
    ownership preserves Lhotse SHAR alignment and rank-local checkpoint/output
    ownership; duration-based cost avoids the old failure mode where equal shard
    counts produced badly imbalanced audio hours.
    """
    if world_size <= 0:
        raise ValueError("world_size must be > 0")

    active_ranks = min(world_size, len(manifest.work_units))
    bins = [
        _Bin(rank=rank, work_units=[], duration=0.0, cut_count=0, fields={})
        for rank in range(active_ranks)
    ]

    ordered_units = sorted(
        manifest.work_units,
        key=lambda u: (-u.duration_sec, -u.cut_count, u.work_unit_id),
    )
    for unit in ordered_units:
        if not bins:
            break
        target = min(bins, key=lambda b: (b.duration, b.cut_count, b.rank))
        target.work_units.append(unit.work_unit_id)
        target.duration += unit.duration_sec
        target.cut_count += unit.cut_count
        for field, paths in unit.fields.items():
            target.fields.setdefault(field, []).extend(paths)

    assignments: list[RankAssignment] = []
    for rank in range(world_size):
        if rank < active_ranks:
            b = bins[rank]
            fields = {field: sorted(paths) for field, paths in b.fields.items()}
            assignments.append(
                RankAssignment(
                    rank=rank,
                    active=bool(b.work_units),
                    work_unit_ids=list(b.work_units),
                    fields=fields,
                    cut_count=int(b.cut_count),
                    duration_sec=float(b.duration),
                )
            )
        else:
            assignments.append(
                RankAssignment(
                    rank=rank,
                    active=False,
                    work_unit_ids=[],
                    fields={},
                    cut_count=0,
                    duration_sec=0.0,
                )
            )

    return TokenizeAssignment(
        world_size=world_size,
        active_ranks=sum(1 for a in assignments if a.active),
        manifest_fingerprint=manifest.fingerprint,
        assignments=assignments,
    )


def write_tokenize_plan_artifacts(
    output_dir: str | Path,
    *,
    manifest: SharWorkManifest,
    assignment: TokenizeAssignment,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / SHAR_WORK_MANIFEST_FILE, manifest.to_json())
    atomic_write_json(output_dir / TOKENIZE_ASSIGNMENT_FILE, assignment.to_json())


def read_tokenize_assignment(path: str | Path) -> TokenizeAssignment:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != PLANNING_SCHEMA_VERSION:
        raise ValueError(f"Unsupported tokenize assignment schema: {payload.get('schema_version')!r}")
    return TokenizeAssignment.from_json(payload)


def _load_existing_manifest(
    shar_dir: str | list[str],
    *,
    index_name: str,
) -> SharWorkManifest | None:
    """Best-effort durable manifest discovery.

    Globs are intentionally not interpreted here because a glob can expand to
    mixed old/new SHAR roots. The caller falls back to an explicit scan in that
    case, which is safer than silently trusting a partial manifest set.
    """

    raw_dirs = shar_dir if isinstance(shar_dir, list) else [shar_dir]
    candidates: list[Path] = []
    for item in raw_dirs:
        item_str = str(item)
        if any(ch in item_str for ch in "*?["):
            logger.debug("Skipping durable manifest discovery: glob char in %r", item_str)
            return None
        candidate = Path(item_str) / SHAR_WORK_MANIFEST_FILE
        if candidate.is_file():
            candidates.append(candidate)

    if not candidates:
        logger.debug("No durable manifest found under %s; falling back to scan", raw_dirs)
        return None

    manifests = [read_shar_work_manifest(path) for path in candidates]
    if len(manifests) == 1:
        manifest = manifests[0]
        if manifest.shar_index_filename != index_name:
            logger.debug(
                "Rejecting manifest %s: shar_index_filename=%r != requested %r",
                candidates[0], manifest.shar_index_filename, index_name,
            )
            return None
        return manifest

    work_units: list[SharWorkUnit] = []
    input_dirs: list[str] = []
    for manifest, path in zip(manifests, candidates):
        if manifest.shar_index_filename != index_name:
            logger.debug(
                "Rejecting partitioned manifest %s: shar_index_filename=%r != requested %r",
                path, manifest.shar_index_filename, index_name,
            )
            return None
        work_units.extend(manifest.work_units)
        input_dirs.extend(manifest.input_shar_dirs)

    return SharWorkManifest(
        input_shar_dirs=sorted(dict.fromkeys(input_dirs)),
        shar_index_filename=index_name,
        work_units=sorted(work_units, key=lambda u: u.work_unit_id),
    )


def _validate_parallel_fields(index_path: Path, fields: Mapping[str, list[str]]) -> None:
    expected = len(fields["cuts"])
    for field, paths in fields.items():
        if len(paths) != expected:
            raise ValueError(
                f"SHAR index field {field!r} in {index_path} has {len(paths)} "
                f"shards, expected {expected} to match 'cuts'."
            )


def _assert_manifest_coverage(
    manifest: SharWorkManifest,
    *,
    tokenize_filter: TokenizeFilter | None,
    require_interleave_ids: bool = False,
) -> None:
    total = sum(unit.cut_count for unit in manifest.work_units)
    required: list[tuple[str, int, str]] = []
    if tokenize_filter is not None and tokenize_filter.min_rms_db is not None:
        required.append(("rms_db_count", sum(unit.rms_db_count for unit in manifest.work_units), "min_rms_db"))
    if tokenize_filter is not None and tokenize_filter.min_sample_rate is not None:
        required.append(("sample_rate_count", sum(unit.sample_rate_count for unit in manifest.work_units), "min_sample_rate"))
    if require_interleave_ids:
        required.extend(
            [
                ("source_id_count", sum(unit.source_id_count for unit in manifest.work_units), "interleaved audio-text"),
                ("clip_num_count", sum(unit.clip_num_count for unit in manifest.work_units), "interleaved audio-text"),
            ]
        )

    missing = [
        f"{field}={count}/{total} required by {reason}"
        for field, count, reason in required
        if count != total
    ]
    if missing:
        raise ValueError(
            "SHAR manifest metadata coverage is incomplete. "
            "Reconvert the SHAR with the canonical conversion pipeline. "
            f"Missing coverage: {', '.join(missing)}"
        )


def _scan_cut_manifests(
    paths: list[Path],
) -> list[dict[str, Any]]:
    if not paths:
        return []
    workers = _resolve_scan_workers(len(paths))
    if workers <= 1:
        return [_scan_cut_manifest(path) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_scan_cut_manifest_for_pool, paths))


def _scan_cut_manifest_for_pool(path: Path) -> dict[str, Any]:
    return _scan_cut_manifest(path)


def _resolve_scan_workers(num_paths: int) -> int:
    if num_paths <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(32, cpu_count, num_paths))


def _scan_cut_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Cut manifest shard not found: {path}")

    cut_count = 0
    duration_sec = 0.0
    min_duration_sec: float | None = None
    max_duration_sec: float | None = None
    rms_db_count = 0
    sample_rate_count = 0
    source_id_count = 0
    clip_num_count = 0
    clip_start_count = 0
    clip_duration_count = 0
    min_rms_db: float | None = None
    max_rms_db: float | None = None
    min_sample_rate: int | None = None
    max_sample_rate: int | None = None

    with open_compressed(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            cut = json.loads(line)
            cut_count += 1
            duration = _optional_float(cut.get("duration")) or 0.0
            duration_sec += duration
            min_duration_sec = duration if min_duration_sec is None else min(min_duration_sec, duration)
            max_duration_sec = duration if max_duration_sec is None else max(max_duration_sec, duration)

            sample_rate = _cut_sample_rate(cut)
            if sample_rate is not None:
                sample_rate_count += 1
                min_sample_rate = sample_rate if min_sample_rate is None else min(min_sample_rate, sample_rate)
                max_sample_rate = sample_rate if max_sample_rate is None else max(max_sample_rate, sample_rate)

            rms_db = _cut_rms_db(cut)
            if rms_db is not None:
                rms_db_count += 1
                min_rms_db = rms_db if min_rms_db is None else min(min_rms_db, rms_db)
                max_rms_db = rms_db if max_rms_db is None else max(max_rms_db, rms_db)

            interleave = _cut_interleave_metadata(cut)
            if interleave is not None:
                if interleave.get("source_id") is not None:
                    source_id_count += 1
                if interleave.get("clip_num") is not None:
                    clip_num_count += 1
                if interleave.get("clip_start") is not None:
                    clip_start_count += 1
                if interleave.get("clip_duration") is not None:
                    clip_duration_count += 1

    return {
        "cut_count": cut_count,
        "duration_sec": duration_sec,
        "min_duration_sec": min_duration_sec,
        "max_duration_sec": max_duration_sec,
        "rms_db_count": rms_db_count,
        "sample_rate_count": sample_rate_count,
        "source_id_count": source_id_count,
        "clip_num_count": clip_num_count,
        "clip_start_count": clip_start_count,
        "clip_duration_count": clip_duration_count,
        "min_rms_db": min_rms_db,
        "max_rms_db": max_rms_db,
        "min_sample_rate": min_sample_rate,
        "max_sample_rate": max_sample_rate,
    }


def _cut_sample_rate(cut: Mapping[str, Any]) -> int | None:
    for source in (cut, cut.get("recording"), cut.get("custom")):
        if not isinstance(source, Mapping):
            continue
        sample_rate = _optional_int(source.get("sampling_rate"))
        if sample_rate is not None:
            return sample_rate
    return None


def _cut_rms_db(cut: Mapping[str, Any]) -> float | None:
    custom = cut.get("custom")
    if not isinstance(custom, Mapping):
        return None
    return _optional_float(custom.get("rms_db"))


def _cut_interleave_metadata(cut: Mapping[str, Any]) -> Mapping[str, Any] | None:
    custom = cut.get("custom")
    if not isinstance(custom, Mapping):
        return None
    interleave = custom.get("interleave")
    if not isinstance(interleave, Mapping):
        return None
    return interleave


def _work_unit_id(shar_dir: Path, cut_path: Path, shard_index: int) -> str:
    raw = f"{shar_dir.resolve()}::{cut_path.name}::{shard_index}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
