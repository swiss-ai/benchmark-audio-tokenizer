#!/usr/bin/env python3
"""Validate a Lhotse SHAR directory with one fixed production contract.

Every shard gets a full structural pass — exhaustive, deterministic, cheap
enough to be the prepare gate:

1. ``cuts.*.jsonl.gz`` deserializes line-by-line into real Lhotse ``Cut``
   objects (full schema check, not just ``id``).
2. Every tar-backed field stays in exact lockstep with the cuts manifest:
   same count, same order, same per-cut stem. The tar's per-cut JSON
   metadata is deserialized into a real Lhotse manifest object (Recording,
   Features, Array, …) — only the metadata blob is read; audio payload
   bytes are skipped.
3. Every jsonl-backed sidecar parses row-by-row as JSON and stays in exact
   lockstep with ``cuts`` via ``cut_id``.

That is the entire contract callers gate on (e.g. ``mark_partition_success``
in ``runtime.py`` writes ``_SUCCESS`` only after this returns). Audio
*payload decode* — i.e. ``cut.load_audio()`` — is intentionally NOT part of
this gate; that's a separate consumer-side smoke tool.

Under the cut.id immutability invariant (postprocess never rewrites cut.id,
only cut.custom) the silent-skip / ID-mismatch class of bugs cannot arise;
the lockstep checks are still load-bearing for producer-side corruption
(crashed workers, partial writes, off-by-N index merges).

Used both as a library (`validate_shar_directory`) and a CLI:

    python -m audio_tokenization.prepare.validate_shar \\
        --shar-dir /path/to/shar [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import tarfile
from itertools import zip_longest
from pathlib import Path

from lhotse.serialization import decode_json_line, deserialize_item
from lhotse.shar.utils import fill_shar_placeholder

from audio_tokenization.contracts.artifacts import SHAR_INDEX_FILENAME
from audio_tokenization.prepare.runtime import resolve_num_workers
from audio_tokenization.utils.io import open_compressed


logger = logging.getLogger(__name__)


class _StructuralReadError(RuntimeError):
    """Raised by inner readers (cuts.jsonl / tar pair iterators) when they
    detect a structural defect they have no shard context to wrap. The outer
    boundary in ``_validate_structural_shard`` catches this and re-raises as
    :class:`SharValidationError` with the missing context populated.
    """


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


def _member_id(raw_path: str) -> str:
    """Return the logical SHAR item id from a tar/jsonl member path.

    Mirrors Lhotse's reader behavior: strip exactly the final extension while
    preserving all parent-directory components. This is intentionally string-
    based rather than ``Path.stem`` because valid cut ids may themselves
    contain ``/`` or URL-like prefixes, and ``Path`` normalization would drop
    information the reader still treats as part of the id.
    """
    return raw_path.rsplit(".", 1)[0] if "." in raw_path else raw_path


def _iter_jsonl_rows(path: Path):
    """Yield ``(line_no, parsed)`` per non-empty line of a jsonl/jsonl.gz file.

    Wraps `json.JSONDecodeError` into `_StructuralReadError` with line context.
    The shared scaffolding for `_iter_cut_ids` and `_iter_jsonl_sidecar_ids`.
    """
    with open_compressed(path, "rt") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, decode_json_line(line)
            except json.JSONDecodeError as e:
                raise _StructuralReadError(
                    f"{path} line {line_no}: cannot parse JSON "
                    f"({type(e).__name__}: {e})"
                ) from e


def _iter_cut_ids(path: Path):
    """Yield ``cut.id`` per line, after deserializing the full Cut.

    Each line must parse as JSON AND deserialize via Lhotse's manifest
    dispatch (``deserialize_item``). Construction failures — missing
    required fields, unknown ``type``, wrong shape — are raised as
    ``_StructuralReadError`` and wrapped with shard context by the caller.
    Returns just the id (the only thing the lockstep check needs); the cut
    object is dropped after validation.
    """
    for line_no, payload in _iter_jsonl_rows(path):
        try:
            cut = deserialize_item(payload)
        except (TypeError, ValueError, KeyError, AssertionError) as e:
            raise _StructuralReadError(
                f"{path} line {line_no}: cannot deserialize cut manifest "
                f"({type(e).__name__}: {e})"
            ) from e
        cut_id = getattr(cut, "id", None)
        if not isinstance(cut_id, str) or not cut_id:
            raise _StructuralReadError(
                f"{path} line {line_no}: deserialized cut has no usable 'id'."
            )
        yield cut_id


def _iter_jsonl_sidecar_ids(path: Path):
    """Yield ``cut_id`` from a jsonl/jsonl.gz SHAR sidecar.

    This mirrors the reader contract in ``LazyJsonlIterator`` +
    ``_jsonl_tar_adaptor``:

    - every non-empty line must parse as JSON
    - every row must be a mapping
    - every row must carry a non-empty ``cut_id`` string

    The field payload itself is intentionally opaque here. The reader accepts
    either a real payload under the field key or a placeholder row with the
    field absent, so the validator only enforces the structural contract it
    actually relies on: JSON readability and lockstep ``cut_id`` ordering.
    """
    for line_no, item in _iter_jsonl_rows(path):
        if not isinstance(item, dict):
            raise _StructuralReadError(
                f"{path} line {line_no}: jsonl sidecar row must be an object, "
                f"got {type(item).__name__}."
            )
        cut_id = item.get("cut_id")
        if not isinstance(cut_id, str) or not cut_id:
            raise _StructuralReadError(
                f"{path} line {line_no}: jsonl sidecar row has no usable "
                f"'cut_id'."
            )
        yield cut_id


def _iter_tar_pair_stems(path: Path):
    """Yield one cut-equivalent stem per (data, metadata) tar member pair.

    Uses Lhotse's metadata-only SHAR tar iterator, which reads ONLY metadata
    bytes, then deserializes the JSON into a real Lhotse manifest object
    (Recording / Features / Array / …). Malformed metadata, unrecognised
    manifest types, stem mismatches within a pair, or uneven member counts
    raise ``_StructuralReadError``.

    Lhotse encodes "optional field omitted for this cut" as a paired
    ``.nodata`` data member + ``.nometa`` metadata member (both empty);
    ``parse_tarinfo_metadata`` returns ``(None, path)`` for both, so they're
    detected by path suffix here, not by metadata presence.

    Tar mode is dispatched on extension: ``.tar.gz`` → ``r:gz`` (the only
    compressed shape this repo emits, via ``merge_shar.py`` with
    ``kind == "tar.gz"``), plain ``.tar`` → ``r:`` (skips a
    compression-probe header read).
    """
    try:
        from lhotse.shar import iterate_tarfile_pairwise_metadata
    except ImportError as exc:
        raise RuntimeError(
            "validate_shar requires Lhotse with "
            "lhotse.shar.iterate_tarfile_pairwise_metadata; source "
            "scripts/utils/source_lhotse_runtime.sh or set LHOTSE_DIR."
        ) from exc

    lower_name = path.name.lower()
    mode = "r:gz" if lower_name.endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(path, mode=mode) as tar:
        try:
            pairs = iterate_tarfile_pairwise_metadata(tar)
            for (left_meta, left_path), (right_meta, right_path) in pairs:
                if _member_id(left_path) != _member_id(right_path):
                    raise _StructuralReadError(
                        f"tar pair stem mismatch in {path.name}: "
                        f"{left_path} vs {right_path}"
                    )

                if left_path.endswith(".nodata") and right_path.endswith(".nometa"):
                    yield _member_id(left_path)
                    continue

                # Real pair: data first (None bytes — payload is not read),
                # metadata second (JSON bytes). Anything else is a producer
                # bug we want to surface, not silently pick a side.
                left_is_meta = left_meta is not None
                right_is_meta = right_meta is not None
                if left_is_meta == right_is_meta:
                    raise _StructuralReadError(
                        f"tar pair in {path.name} has both/neither metadata "
                        f"members: {left_path} / {right_path}"
                    )
                if left_is_meta or not right_is_meta:
                    raise _StructuralReadError(
                        f"tar pair in {path.name} is not ordered as "
                        f"data-then-metadata: {left_path} / {right_path}"
                    )

                meta_blob = right_meta
                meta_path = right_path
                try:
                    manifest = deserialize_item(decode_json_line(meta_blob.decode("utf-8")))
                    # Reapply the same placeholder-fill invariant the real
                    # reader enforces, without reading full payload bytes —
                    # we only need to prove the manifest shape is compatible
                    # with SHAR placeholder filling (Recording has exactly one
                    # source, array suffix matches a supported backend, etc.).
                    placeholder_data = None if left_path.endswith(".nodata") else b""
                    fill_shar_placeholder(
                        manifest=manifest,
                        data=placeholder_data,
                        tarpath=left_path,
                    )
                except (
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    TypeError,
                    ValueError,
                    AssertionError,
                ) as e:
                    raise _StructuralReadError(
                        f"tar metadata in {path.name} entry {meta_path} "
                        f"failed to deserialize ({type(e).__name__}: {e})"
                    ) from e
                except RuntimeError as e:
                    raise _StructuralReadError(
                        f"tar metadata in {path.name} entry {meta_path} "
                        f"failed SHAR placeholder validation "
                        f"({type(e).__name__}: {e})"
                    ) from e

                yield _member_id(left_path)
        except _StructuralReadError:
            raise
        except RuntimeError as exc:
            # Upstream raises a plain RuntimeError("Uneven number of files
            # in the tarfile...") when the archive has odd member counts.
            # Wrap so _validate_structural_shard sees one canonical
            # structural-error type instead of bypassing its except list.
            raise _StructuralReadError(
                f"tar archive {path.name} has malformed structure "
                f"(expected pairs of data + metadata members): {exc}"
            ) from exc


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


def _validate_field_against_cuts(
    *,
    shard_name: str,
    cuts_path: Path,
    cut_ids: list[str],
    field_name: str,
    field_path: Path,
    payload_kind: str,  # "tar" or "jsonl" — used for error messages only
    payload_iter,
) -> None:
    """Walk ``cut_ids`` and ``payload_iter`` in lockstep.

    Both iterators must yield the SAME stems/ids in the SAME order. Any
    length mismatch or per-position mismatch raises ``SharValidationError``
    with shard context and the failing position.
    """
    last_good_cut_id: str | None = None
    cuts_consumed = 0
    missing = object()
    payload_label = "tar stem" if payload_kind == "tar" else "sidecar cut_id"

    for expected_id, payload_id in zip_longest(
        cut_ids, payload_iter, fillvalue=missing,
    ):
        if expected_id is missing or payload_id is missing:
            _raise(
                shard_name=shard_name,
                cuts_path=cuts_path,
                last_good_cut_id=last_good_cut_id,
                cuts_consumed=cuts_consumed,
                message=(
                    f"manifest has {len(cut_ids)} cuts but {field_name} "
                    f"{payload_kind} ({field_path.name}) has "
                    f"{cuts_consumed + (payload_id is not missing)} entries "
                    f"before lockstep broke — cuts and fields must stay in "
                    f"lockstep."
                ),
            )
        if payload_id != expected_id:
            _raise(
                shard_name=shard_name,
                cuts_path=cuts_path,
                last_good_cut_id=last_good_cut_id,
                cuts_consumed=cuts_consumed,
                message=(
                    f"{field_name} {payload_kind} ({field_path.name}) is out "
                    f"of lockstep: manifest cut id {expected_id!r} != "
                    f"{payload_label} {payload_id!r}."
                ),
            )
        last_good_cut_id = expected_id
        cuts_consumed += 1


def _validate_structural_shard(
    *,
    shard_name: str,
    slice_fields: dict[str, list[str]],
) -> int:
    cuts_path = Path(slice_fields["cuts"][0])
    try:
        # Materialize once; without this each non-cuts field would
        # re-deserialize every cut from cuts.jsonl.gz, scaling the heaviest
        # per-shard cost (Cut.from_dict via deserialize_item) by M fields.
        cut_ids = list(_iter_cut_ids(cuts_path))
        for field_name, paths in slice_fields.items():
            if field_name == "cuts":
                continue
            field_path = Path(paths[0])
            if field_path.name.endswith((".jsonl", ".jsonl.gz")):
                payload_iter = _iter_jsonl_sidecar_ids(field_path)
                payload_kind = "jsonl"
            else:
                payload_iter = _iter_tar_pair_stems(field_path)
                payload_kind = "tar"
            _validate_field_against_cuts(
                shard_name=shard_name,
                cuts_path=cuts_path,
                cut_ids=cut_ids,
                field_name=field_name,
                field_path=field_path,
                payload_kind=payload_kind,
                payload_iter=payload_iter,
            )
        return len(cut_ids)
    except SharValidationError:
        # Already carries shard context; pass through unchanged.
        raise
    except (
        json.JSONDecodeError,
        tarfile.ReadError,
        EOFError,
        OSError,
        _StructuralReadError,
    ) as e:
        # Any structural read/parse failure is a SHAR validation failure;
        # wrap so callers (and the CLI) see one canonical error type with
        # shard context preserved.
        raise SharValidationError(
            shard_name=shard_name,
            cuts_path=cuts_path,
            last_good_cut_id=None,
            cuts_consumed=0,
            original=e,
        ) from e


def _worker_error_result(
    *,
    shard_name: str,
    slice_fields: dict[str, list[str]],
    error: BaseException,
) -> dict[str, object]:
    cuts_paths = slice_fields.get("cuts") or [""]
    if isinstance(error, SharValidationError):
        return {
            "ok": False,
            "shard_name": error.shard_name,
            "cuts_path": str(error.cuts_path),
            "last_good_cut_id": error.last_good_cut_id,
            "cuts_consumed": error.cuts_consumed,
            "error_type": type(error.original).__name__,
            "message": str(error.original),
        }
    return {
        "ok": False,
        "shard_name": shard_name,
        "cuts_path": str(cuts_paths[0]),
        "last_good_cut_id": None,
        "cuts_consumed": 0,
        "error_type": type(error).__name__,
        "message": f"{type(error).__name__}: {error}",
    }


def _validate_shard_worker(args: tuple[str, dict[str, list[str]]]) -> dict[str, object]:
    shard_name, slice_fields = args
    try:
        count = _validate_structural_shard(
            shard_name=shard_name,
            slice_fields=slice_fields,
        )
    except Exception as e:
        return _worker_error_result(
            shard_name=shard_name,
            slice_fields=slice_fields,
            error=e,
        )
    return {
        "ok": True,
        "shard_name": shard_name,
        "count": count,
    }


def _raise_worker_validation_error(result: dict[str, object]) -> None:
    # Lossy by design: the worker's real exception type and traceback don't
    # survive the multiprocessing pickle boundary, so we reconstruct a
    # RuntimeError carrying the rendered ``error_type: message`` string. The
    # message + shard context is the operator-facing signal; the original
    # traceback lives in the worker's stderr if deeper post-mortem is needed.
    message = str(result.get("message") or "SHAR worker validation failed")
    error_type = str(result.get("error_type") or "RuntimeError")
    err = RuntimeError(f"{error_type}: {message}")
    raise SharValidationError(
        shard_name=str(result.get("shard_name") or "<unknown>"),
        cuts_path=Path(str(result.get("cuts_path") or "")),
        last_good_cut_id=result.get("last_good_cut_id") or None,
        cuts_consumed=int(result.get("cuts_consumed") or 0),
        original=err,
    ) from err


def validate_shar_directory(
    shar_dir: Path,
    *,
    verbose: bool = False,
    index_filename: str = SHAR_INDEX_FILENAME,
    num_workers: int | None = None,
) -> dict[str, int]:
    """Validate *shar_dir* and return per-shard cut counts.

    The production contract is fixed: full structural validation on every
    shard. No sampling, no deep iteration. ``_SUCCESS`` after this gate
    means exactly that — structural correctness, not payload decodability.

    *num_workers* is the per-shard parallelism. ``None`` (default) uses
    ``SLURM_CPUS_PER_TASK`` if set, else ``os.cpu_count()``, capped at the
    shard count. Each worker holds one shard's cuts.jsonl + scans the tar
    headers; ~hundreds of MB peak per worker.
    """
    shar_dir = Path(shar_dir)
    fields = _load_shar_index(shar_dir, index_filename)
    slices = _shard_slices(shar_dir, fields)

    n_workers = resolve_num_workers(num_workers, num_inputs=len(slices))
    if verbose:
        logger.info("validating %d shards (structural, %d workers)", len(slices), n_workers)

    counts: dict[str, int] = {}

    def _record(shard_name: str, expected: int) -> None:
        counts[shard_name] = expected
        if verbose:
            logger.info("validated %s: %d cuts", shard_name, expected)

    if n_workers == 1:
        # Avoid pool overhead and keep tracebacks readable for single-shard
        # debugging.
        for shard_name, slice_fields in slices:
            _record(
                shard_name,
                _validate_structural_shard(
                    shard_name=shard_name,
                    slice_fields=slice_fields,
                ),
            )
    else:
        # Default context (fork on Linux) — cheaper than forkserver because
        # workers inherit the parent's lhotse imports via COW. Also works in
        # restricted sandboxes where forkserver can't open its AF_UNIX
        # socket. Workers never mutate global state, so fork is safe.
        # imap_unordered streams results as workers finish so the verbose
        # log shows live progress instead of dumping at the end.
        with multiprocessing.Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_validate_shard_worker, slices):
                if result.get("ok") is not True:
                    _raise_worker_validation_error(result)
                _record(str(result["shard_name"]), int(result["count"]))
    return counts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate a Lhotse SHAR directory with full structural checks.",
    )
    p.add_argument("--shar-dir", type=Path, required=True)
    p.add_argument(
        "--index-filename",
        default=SHAR_INDEX_FILENAME,
        help=(
            "Name of the SHAR index file inside --shar-dir "
            f"(default: {SHAR_INDEX_FILENAME}). Match this to "
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
