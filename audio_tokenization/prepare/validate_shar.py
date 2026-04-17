#!/usr/bin/env python3
"""Validate a Lhotse SHAR directory by reading every shard end-to-end.

Three independent checks per shard, any of which trips a
:class:`SharValidationError`:

1. Manifest line count == tar member count for every non-cuts field. Catches
   dropped/added cuts that left the recording tars untouched (lockstep break).
2. ``LazySharIterator`` consumes every shard without raising (catches
   corrupt cuts JSONL, missing tar entries, etc.).
3. Cuts yielded by the iterator == manifest line count. Catches the case
   where Lhotse silently skips cut↔tar id mismatches (the dev lhotse at
   ``shar/readers/lazy.py:294-300`` warns + skips instead of asserting, so
   relying on iteration alone is insufficient).

Used both as a library (`validate_shar_directory`) and a CLI:

    python -m audio_tokenization.prepare.validate_shar \\
        --shar-dir /path/to/shar [--verbose]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.shar.readers.tar import TarIterator


logger = logging.getLogger(__name__)

SHAR_INDEX_FILE = "shar_index.json"


class SharValidationError(RuntimeError):
    """Raised when a SHAR directory cannot be read back end-to-end."""

    def __init__(
        self,
        *,
        shard_name: str,
        cuts_path: Path,
        last_good_cut_id: str | None,
        cuts_consumed: int,
        original: BaseException,
    ) -> None:
        self.shard_name = shard_name
        self.cuts_path = cuts_path
        self.last_good_cut_id = last_good_cut_id
        self.cuts_consumed = cuts_consumed
        self.original = original
        super().__init__(
            f"SHAR shard {shard_name!r} failed validation after "
            f"{cuts_consumed} cuts (last good id: {last_good_cut_id!r}). "
            f"cuts_path={cuts_path}. Underlying error: "
            f"{type(original).__name__}: {original}"
        )


def _load_shar_index(shar_dir: Path, index_filename: str) -> dict[str, list[str]]:
    index_path = shar_dir / index_filename
    if not index_path.is_file():
        raise FileNotFoundError(
            f"No {index_filename} in {shar_dir}; nothing to validate."
        )
    payload = json.loads(index_path.read_text())
    fields = payload.get("fields")
    if not isinstance(fields, dict) or "cuts" not in fields:
        raise RuntimeError(
            f"Invalid {index_filename} at {index_path}: missing 'fields.cuts'."
        )
    return fields


def _shard_slices(
    shar_dir: Path, fields: dict[str, list[str]]
) -> list[tuple[str, dict[str, list[str]]]]:
    num_shards = len(fields["cuts"])
    for name, paths in fields.items():
        if len(paths) != num_shards:
            raise RuntimeError(
                f"shar_index field {name!r} has {len(paths)} shards but 'cuts' "
                f"has {num_shards}; index is inconsistent."
            )
    slices: list[tuple[str, dict[str, list[str]]]] = []
    for shard_idx in range(num_shards):
        per_shard = {
            name: [str(shar_dir / paths[shard_idx])] for name, paths in fields.items()
        }
        slices.append((fields["cuts"][shard_idx], per_shard))
    return slices


def _count_jsonl_entries(path: Path) -> int:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as f:
        return sum(1 for line in f if line.strip())


def _count_tar_pairs(path: Path) -> int:
    """Count cut-equivalent pairs in a Lhotse SHAR tar.

    Lhotse SHAR tars store each cut as a (data, metadata) pair of members,
    so the per-cut count is half the raw tar member count. Using
    ``TarIterator`` here also fails fast on uneven member counts (the same
    invariant Lhotse enforces on read).
    """
    return sum(1 for _ in TarIterator(str(path)))


def _raise(
    *,
    shard_name: str,
    cuts_path: Path,
    last_good_cut_id: str | None,
    cuts_consumed: int,
    message: str,
) -> None:
    err = RuntimeError(message)
    raise SharValidationError(
        shard_name=shard_name,
        cuts_path=cuts_path,
        last_good_cut_id=last_good_cut_id,
        cuts_consumed=cuts_consumed,
        original=err,
    ) from err


def validate_shar_directory(
    shar_dir: Path,
    *,
    verbose: bool = False,
    index_filename: str = SHAR_INDEX_FILE,
) -> dict[str, int]:
    """Read every shard in *shar_dir* via Lhotse and return per-shard cut counts.

    Raises :class:`SharValidationError` on the first shard that fails any of
    the three checks documented at module level.
    """
    shar_dir = Path(shar_dir)
    fields = _load_shar_index(shar_dir, index_filename)
    counts: dict[str, int] = {}

    for shard_name, slice_fields in _shard_slices(shar_dir, fields):
        cuts_path = Path(slice_fields["cuts"][0])
        expected = _count_jsonl_entries(cuts_path)

        # Check 1: each non-cuts field's per-shard count must equal `expected`.
        # SharWriter supports binary fields written as .tar (recording, custom
        # arrays) and metadata fields written as .jsonl.gz (e.g. captions); the
        # counter dispatches on extension so jsonl sidecars don't get sent
        # through tarfile.open.
        for field_name, paths in slice_fields.items():
            if field_name == "cuts":
                continue
            field_path = Path(paths[0])
            if field_path.name.endswith((".jsonl", ".jsonl.gz")):
                field_count = _count_jsonl_entries(field_path)
                kind = "jsonl"
            else:
                field_count = _count_tar_pairs(field_path)
                kind = "tar"
            if field_count != expected:
                _raise(
                    shard_name=shard_name,
                    cuts_path=cuts_path,
                    last_good_cut_id=None,
                    cuts_consumed=0,
                    message=(
                        f"manifest has {expected} cuts but {field_name} {kind} "
                        f"({field_path.name}) has {field_count} entries — "
                        f"cuts and fields must stay in lockstep."
                    ),
                )

        # Check 2 + 3: iteration must complete and yield exactly `expected`.
        last_good_cut_id: str | None = None
        consumed = 0
        try:
            for cut in LazySharIterator(fields=slice_fields):
                last_good_cut_id = cut.id
                consumed += 1
        except BaseException as e:
            raise SharValidationError(
                shard_name=shard_name,
                cuts_path=cuts_path,
                last_good_cut_id=last_good_cut_id,
                cuts_consumed=consumed,
                original=e,
            ) from e

        if consumed != expected:
            _raise(
                shard_name=shard_name,
                cuts_path=cuts_path,
                last_good_cut_id=last_good_cut_id,
                cuts_consumed=consumed,
                message=(
                    f"Lhotse yielded {consumed} cuts but manifest has "
                    f"{expected} entries — the reader is silently skipping "
                    f"cut↔tar id mismatches. SHAR is corrupt."
                ),
            )

        counts[shard_name] = consumed
        if verbose:
            logger.info("validated %s: %d cuts", shard_name, consumed)

    return counts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate a Lhotse SHAR directory by reading every shard."
    )
    p.add_argument("--shar-dir", type=Path, required=True)
    p.add_argument(
        "--index-filename",
        default=SHAR_INDEX_FILE,
        help=(
            "Name of the SHAR index file inside --shar-dir "
            f"(default: {SHAR_INDEX_FILE}). Match this to "
            "--shar_index_filename if the SHAR was prepared with a non-default value."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Log each validated shard")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()

    try:
        counts = validate_shar_directory(
            args.shar_dir,
            verbose=args.verbose,
            index_filename=args.index_filename,
        )
    except SharValidationError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)

    total = sum(counts.values())
    print(f"OK: {total:,} cuts across {len(counts)} shards in {args.shar_dir}")
    for shard_name, n in counts.items():
        print(f"  {shard_name}: {n:,} cuts")


if __name__ == "__main__":
    main()
