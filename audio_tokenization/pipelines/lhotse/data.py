"""Lhotse data loading: load prepared Shar for tokenization.

Data preparation (HF/WDS -> Shar) is handled by standalone scripts:
    - audio_tokenization.utils.prepare_data.prepare_hf_to_shar
    - audio_tokenization.utils.prepare_data.prepare_wds_to_shar

This module only loads pre-built Shar and applies runtime filters.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

SHAR_INDEX_FILENAME = "shar_index.json"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_index_paths(shar_root: Path, fields: Dict[str, list]) -> Dict[str, list]:
    """Resolve relative paths in shar_index fields against *shar_root*.

    Absolute index entries are rejected to keep SHAR fully relocatable.
    """
    resolved: Dict[str, list] = {}
    for field, paths in fields.items():
        out = []
        for p in paths:
            pp = Path(p)
            if pp.is_absolute():
                raise ValueError(
                    f"Absolute path in shar index is not allowed: {pp}. "
                    f"Rebuild {shar_root / SHAR_INDEX_FILENAME} with relative paths."
                )
            pp = shar_root / pp
            out.append(str(pp))
        resolved[field] = out
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cutset(cfg: Dict[str, Any], rank: int, world_size: int, stats=None):
    """Load prepared Shar into a CutSet and apply post-load filters."""
    _set_resampling_backend(rank)

    cuts = _load_shar_cutset(cfg, rank, world_size)

    # Drop low sample-rate audio before resampling (e.g., 8kHz -> 24kHz = garbage).
    min_sr = cfg.get("min_sample_rate")
    if min_sr is not None:
        min_sr = int(min_sr)
        cuts = cuts.filter(
            lambda cut: getattr(cut, "sampling_rate", None) is not None
            and cut.sampling_rate >= min_sr
        )

    # Lazy safety-net resample (no-op when SR already matches).
    target_sr = cfg.get("target_sample_rate")
    if target_sr:
        cuts = cuts.resample(int(target_sr))

    min_dur = cfg.get("min_duration")
    max_dur = cfg.get("max_duration")
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
    mode = cfg.get("mode", "audio_only")
    if mode == "audio_text":
        def _has_text(cut):
            if not cut.supervisions or not cut.supervisions[0].text:
                if stats is not None:
                    stats.no_text_skipped += 1
                return False
            return True

        cuts = cuts.filter(_has_text)

    # Drop quiet audio using precomputed rms_db stored in cut.custom during
    # Shar preparation.  Old Shar without the field passes through unfiltered.
    min_rms_db = cfg.get("min_rms_db")
    if min_rms_db is not None:
        _min_rms = float(min_rms_db)

        def _rms_filter(cut):
            val = (cut.custom or {}).get("rms_db")
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


def _load_shar_cutset(cfg, rank, world_size=1):
    """Load a CutSet from one or more prepared Shar directories.

    ``shar_dir`` may be a single path (str) or a list of paths.  When
    multiple directories are given their shar indexes are merged so that
    ``CutSet.from_shar`` sees one unified pool of shards.

    When ``world_size > 1``, shards are split across DDP ranks via
    round-robin assignment so each rank loads only its subset.  This
    avoids the O(world_size) overhead of Lhotse's strided distribution.
    """
    from lhotse import CutSet

    shar_dir = cfg.get("shar_dir")
    if not shar_dir:
        raise ValueError("Lhotse tokenization requires 'shar_dir' with prepared Shar data.")

    # Normalise to a list so single-dir and multi-dir use the same code path.
    shar_dirs = shar_dir if isinstance(shar_dir, (list, tuple)) else [shar_dir]

    index_name = cfg.get("shar_index_filename", SHAR_INDEX_FILENAME)
    merged_fields: dict[str, list[str]] = {}

    for sd in shar_dirs:
        shar_path = Path(sd)
        if not shar_path.is_dir():
            raise FileNotFoundError(f"Shar directory does not exist: {sd}")

        index_path = shar_path / index_name
        if index_path.is_file():
            with open(index_path) as f:
                fields = json.load(f).get("fields", {})
            if "cuts" not in fields:
                raise ValueError(f"Shar index missing required 'cuts' field: {index_path}")
            fields = _resolve_index_paths(shar_path, fields)
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

    # Sort for determinism.
    merged_fields = {k: sorted(v) for k, v in merged_fields.items()}
    total_shards = len(merged_fields.get("cuts", []))
    logger.info(
        f"[rank {rank}] Merged {len(shar_dirs)} shar dir(s): "
        f"{total_shards} cut shards"
    )

    # Split shards across DDP ranks (round-robin) so each rank's sampler
    # only iterates its own subset — eliminates O(world_size) overhead.
    if world_size > 1:
        for field in merged_fields:
            merged_fields[field] = merged_fields[field][rank::world_size]
        logger.info(
            f"[rank {rank}] Shard split: "
            f"{len(merged_fields['cuts'])}/{total_shards} shards"
        )

    # Intentionally keep split_for_dataloading disabled here.
    # This pipeline assigns whole SHAR shards to ranks explicitly above so
    # checkpoint ownership and output ownership are both rank-local. Switching
    # to Lhotse's worker/node striding would blur that ownership boundary and
    # make recovery/output layout harder to reason about.
    return CutSet.from_shar(fields=merged_fields, split_for_dataloading=False, shuffle_shards=True)


def _shar_exists(shar_dir: str) -> bool:
    p = Path(shar_dir)
    if not p.is_dir():
        return False
    return any(p.glob("cuts*.jsonl.gz"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_resampling_backend(rank: int) -> None:
    from lhotse.audio.resampling_backend import set_current_resampling_backend

    set_current_resampling_backend("soxr")
    logger.info(f"[rank {rank}] Using soxr resampling backend")
