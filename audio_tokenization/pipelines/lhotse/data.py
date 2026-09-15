"""Lhotse data loading: load prepared Shar for tokenization.

Data preparation (HF/WDS -> Shar) is handled by standalone scripts:
    - audio_tokenization.prepare.prepare_hf_to_shar
    - audio_tokenization.prepare.prepare_wds_to_shar

This module only loads pre-built Shar and applies runtime filters.
"""

import glob
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.contracts.artifacts import SHAR_INDEX_FILENAME

logger = logging.getLogger(__name__)

_GLOB_CHARS = "*?["


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_shar_dirs(shar_dir, *, index_name: str = SHAR_INDEX_FILENAME) -> list[str]:
    """Resolve SHAR inputs to concrete leaf directories.

    Accepts a single directory, a list of directories, glob patterns such as
    ``/path/to/root/node_*``, and partitioned roots containing child directories
    with ``shar_index.json``. Returned paths are sorted and de-duplicated for
    stable audit manifests.
    """
    if not shar_dir:
        raise ValueError("Lhotse tokenization requires 'shar_dir' with prepared Shar data.")

    raw_dirs = shar_dir if isinstance(shar_dir, (list, tuple)) else [shar_dir]
    resolved: list[str] = []

    for item in raw_dirs:
        item_str = str(item)
        if any(ch in item_str for ch in _GLOB_CHARS):
            matches = sorted(Path(p) for p in glob.glob(item_str))
            dirs = [p for p in matches if p.is_dir()]
            if not dirs:
                raise FileNotFoundError(f"No Shar directories match pattern: {item_str}")
            for d in dirs:
                resolved.extend(_expand_partitioned_root(d, index_name=index_name))
            continue

        resolved.extend(_expand_partitioned_root(Path(item_str), index_name=index_name))

    return sorted({str(Path(path).resolve()) for path in resolved})


def _expand_partitioned_root(shar_path: Path, *, index_name: str) -> list[str]:
    """Expand a root with node_*/shar_index.json children to leaf dirs."""
    if not shar_path.is_dir():
        return [str(shar_path)]

    if (shar_path / index_name).is_file() or _shar_exists(str(shar_path)):
        return [str(shar_path)]

    child_dirs = sorted(
        child
        for child in shar_path.iterdir()
        if child.is_dir() and (child / index_name).is_file()
    )
    if child_dirs:
        return [str(child) for child in child_dirs]

    return [str(shar_path)]


def _resolve_index_paths(shar_root: Path, fields: dict[str, list]) -> dict[str, list]:
    """Resolve relative paths in shar_index fields against *shar_root*.

    Absolute index entries are rejected to keep SHAR fully relocatable.
    """
    resolved: dict[str, list] = {}
    for field, paths in fields.items():
        out = []
        for p in paths:
            pp = Path(p)
            if pp.is_absolute():
                raise ValueError(
                    f"Absolute path in shar index is not allowed: {pp}. "
                    f"Rebuild {shar_root / SHAR_INDEX_FILENAME} with relative paths."
                )
            pp = (shar_root / pp).resolve()
            out.append(str(pp))
        resolved[field] = out
    return resolved


def _read_shar_index_fields(index_path: Path) -> dict[str, list[str]]:
    """Resolve and validate the paired fields shared by planning and loading."""
    with index_path.open() as stream:
        fields = json.load(stream).get("fields", {})
    if "cuts" not in fields:
        raise ValueError(f"Shar index missing required 'cuts' field: {index_path}")
    fields = _resolve_index_paths(index_path.parent, fields)
    _validate_parallel_fields(index_path, fields)
    return fields


def _validate_parallel_fields(index_path: Path, fields: Mapping[str, list[str]]) -> None:
    if "cuts" not in fields:
        raise ValueError(f"Shar index missing required 'cuts' field: {index_path}")
    expected = len(fields["cuts"])
    for field, paths in fields.items():
        if len(paths) != expected:
            raise ValueError(
                f"SHAR index field {field!r} in {index_path} has {len(paths)} "
                f"shards, expected {expected} to match 'cuts'."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cutset(
    spec: TokenizeSpec,
    *,
    input_shar_dirs: Sequence[str],
    planned_shar_fields: dict[str, list[str]] | None,
    rank: int,
    world_size: int,
    stats=None,
):
    """Load prepared Shar into a CutSet and apply post-load filters."""
    _set_resampling_backend(rank, spec.resampling_backend)

    cuts = _load_shar_cutset(
        input_shar_dirs=input_shar_dirs,
        planned_shar_fields=planned_shar_fields,
        index_name=spec.output.shar_index_filename,
        rank=rank,
        world_size=world_size,
    )

    # Drop low sample-rate audio before resampling (e.g., 8kHz -> 24kHz = garbage).
    min_sr = spec.filter.min_sample_rate
    if min_sr is not None:
        min_sr = int(min_sr)
        cuts = cuts.filter(
            lambda cut: getattr(cut, "sampling_rate", None) is not None
            and cut.sampling_rate >= min_sr
        )

    # Lazy safety-net resample (no-op when SR already matches).
    target_sr = spec.tokenizer.sampling_rate
    if target_sr:
        cuts = cuts.resample(int(target_sr))

    min_dur = spec.filter.min_duration
    max_dur = spec.filter.max_duration
    if min_dur is not None or max_dur is not None:
        def _dur_filter(cut) -> bool:
            d = cut.duration
            if min_dur is not None and d < min_dur:
                return False
            if max_dur is not None and d > max_dur:
                return False
            return True

        cuts = cuts.filter(_dur_filter)

    # In audio_text mode, drop cuts without text supervision.
    if spec.mode == "audio_text":
        def _has_text(cut):
            if not cut.supervisions or not cut.supervisions[0].text:
                if stats is not None:
                    stats.no_text_skipped += 1
                return False
            return True

        cuts = cuts.filter(_has_text)

    # Drop quiet audio using precomputed rms_db stored in cut.custom during
    # SHAR preparation. Missing rms_db is a conversion error: rebuild the SHAR
    # instead of silently changing the tokenization filter semantics.
    min_rms_db = spec.filter.min_rms_db
    if min_rms_db is not None:
        _min_rms = float(min_rms_db)

        def _rms_filter(cut):
            val = (cut.custom or {}).get("rms_db")
            if val is None:
                raise ValueError(
                    f"Cut {cut.id!r} is missing cut.custom['rms_db'] while "
                    "min_rms_db filtering is enabled. Reconvert this SHAR so "
                    "RMS metadata is written during conversion."
                )
            if val is not None and val < _min_rms:
                if stats is not None:
                    stats.rms_skipped += 1
                return False
            return True

        cuts = cuts.filter(_rms_filter)

    return cuts


# ---------------------------------------------------------------------------
# Shar loading
# ---------------------------------------------------------------------------


def _load_shar_cutset(
    *,
    input_shar_dirs: Sequence[str],
    planned_shar_fields: dict[str, list[str]] | None,
    index_name: str,
    rank: int,
    world_size: int = 1,
):
    """Load a CutSet from one or more prepared Shar directories.

    ``shar_dir`` may be a single path (str) or a list of paths.  When
    multiple directories are given their shar indexes are merged so that
    ``CutSet.from_shar`` sees one unified pool of shards.

    Multi-rank launches must pass ``planned_shar_fields`` from
    ``stages/tokenize.py``. That plan assigns whole SHAR work units by
    estimated duration before this loader is called.
    """
    if planned_shar_fields is not None:
        planned_fields = {k: list(v) for k, v in planned_shar_fields.items()}
        logger.info(
            "[rank %s] Loading planned SHAR assignment: %s cut shard(s)",
            rank,
            len(planned_fields.get("cuts", [])),
        )
        return _cutset_from_paired_fields(planned_fields, rank=rank)

    shar_dirs = resolve_shar_dirs(input_shar_dirs, index_name=index_name)
    merged_fields: dict[str, list[str]] = {}

    for sd in shar_dirs:
        shar_path = Path(sd)
        if not shar_path.is_dir():
            raise FileNotFoundError(f"Shar directory does not exist: {sd}")

        index_path = shar_path / index_name
        if index_path.is_file():
            fields = _read_shar_index_fields(index_path)
            logger.info(f"[rank {rank}] Loading Shar index from {index_path}")
        elif _shar_exists(sd):
            raise FileNotFoundError(
                f"Shar directory {sd} has manifests but no {index_name}. "
                "Build the index first."
            )
        else:
            raise FileNotFoundError(
                f"No Shar manifests found in {sd}. "
                "Run prepare_hf_to_shar or prepare_wds_to_shar first."
            )

        for field, paths in fields.items():
            merged_fields.setdefault(field, []).extend(paths)

    _validate_parallel_fields(Path("merged SHAR inputs"), merged_fields)
    # Match the planned path's lexical cuts order by permuting whole units.
    order = sorted(range(len(merged_fields["cuts"])), key=lambda i: merged_fields["cuts"][i])
    merged_fields = {field: [paths[i] for i in order] for field, paths in merged_fields.items()}
    total_shards = len(merged_fields.get("cuts", []))
    logger.info(
        f"[rank {rank}] Merged {len(shar_dirs)} shar dir(s): "
        f"{total_shards} cut shards"
    )

    if world_size > 1:
        raise RuntimeError(
            "Multi-rank tokenization requires planned_shar_fields from the "
            "tokenize stage; refusing to load the full SHAR on every rank."
        )

    return _cutset_from_paired_fields(merged_fields, rank=rank)


def _cutset_from_paired_fields(fields: dict[str, list[str]], *, rank: int):
    from lhotse import CutSet

    _validate_parallel_fields(Path("SHAR reader fields"), fields)
    # Intentionally keep split_for_dataloading disabled here.
    # This pipeline assigns whole SHAR shards to ranks via planned fields so
    # checkpoint ownership and output ownership are both rank-local. Switching
    # to Lhotse's worker/node striding would blur that ownership boundary and
    # make recovery/output layout harder to reason about.
    cuts = CutSet.from_shar(fields=fields, split_for_dataloading=False, shuffle_shards=True)
    logger.info(
        "[rank %s] SHAR reader_mode=%s split_for_dataloading=False shuffle_shards=True",
        rank, "indexed" if cuts.is_indexed else "streaming",
    )
    return cuts


def _shar_exists(shar_dir: str) -> bool:
    p = Path(shar_dir)
    if not p.is_dir():
        return False
    return any(p.glob("cuts*.jsonl.gz"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_resampling_backend(rank: int, backend: str | None) -> None:
    from lhotse.audio.resampling_backend import set_current_resampling_backend

    if not backend:
        return
    set_current_resampling_backend(str(backend))
    logger.info("[rank %s] Using %s resampling backend", rank, backend)
