"""Columnar row metadata helpers for parquet and HF arrow prepare paths."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re

from audio_tokenization.prepare.cli import (
    add_language_arg,
    add_text_tokenizer_args,
)
from audio_tokenization.prepare.constants import _MISSING
from audio_tokenization.prepare.identity import resolve_input_source_and_clip_num


@dataclass(frozen=True, slots=True)
class ColumnarWorkerArgs:
    """Typed worker contract for columnar prepare families.

    Shared by parquet and HF arrow convert paths to avoid positional tuple
    drift across worker builders, tests, and the worker entrypoints.
    """

    worker_id: int
    input_paths: tuple[str, ...]
    shar_dir: str
    target_sr: int | None
    shard_size: int
    shar_format: str
    id_column: str | tuple[str, ...] | None
    id_prefix: str | None
    audio_column: str
    text_column: str | None
    # parquet-only: hf workers ignore this. Kept on the shared contract so
    # the same dataclass covers both families; an extracted shared worker
    # body would gate on family.
    duration_column: str | None
    language_column: str | None
    language: str | None
    custom_columns: tuple[str, ...] | None
    constant_custom: dict[str, object] | None
    derived_custom: dict[str, str] | None
    text_tokenize_custom_columns: tuple[str, ...] | None
    text_tokenizer_path: str | None
    resampling_backend: str | None
    input_clip_id_parser_name: str | None
    source_id_column: str | None
    clip_num_column: str | None
    clip_start_column: str | None
    clip_end_column: str | None
    clip_duration_column: str | None
    read_batch_size: int


def add_columnar_metadata_args(
    parser,
    *,
    id_column_default=None,
    text_column_default="text",
    duration_column_default=None,
):
    """Add shared CLI args for extracting metadata from columnar sources."""
    parser.add_argument("--id-column", type=str, nargs="*", default=id_column_default,
                        help="Column name(s) for row ID. Multiple columns are joined with '_'. "
                             "Dotted paths like 'audio.path' access nested struct fields. "
                             "Omit to auto-generate IDs from filename + row index.")
    parser.add_argument("--audio-column", type=str, default="audio",
                        help="Column name for audio struct (default: 'audio')")
    parser.add_argument("--text-column", type=str, default=text_column_default,
                        help=f"Column name for transcription text (default: {text_column_default!r})")
    parser.add_argument("--duration-column", type=str, default=duration_column_default,
                        help="Column name for duration in seconds (for filtering)")
    parser.add_argument("--source-id-column", type=str, default=None,
                        help="Column name for interleave source_id. "
                             "When clip timestamps are present, clip_num is only "
                             "used as a deterministic tie-breaker.")
    parser.add_argument("--clip-num-column", type=str, default=None,
                        help="Column name for interleave clip_num. "
                             "Use with --source-id-column to avoid parsing row IDs.")
    parser.add_argument("--clip-start-column", type=str, default=None,
                        help="Column name for segment start time in the source timeline")
    parser.add_argument("--clip-end-column", type=str, default=None,
                        help="Column name for segment end time in the source timeline")
    parser.add_argument("--clip-duration-column", type=str, default=None,
                        help="Column name for segment duration in the source timeline")
    parser.add_argument("--language-column", type=str, default=None,
                        help="Column name for per-row language code. "
                             "Takes precedence over --language when set.")
    parser.add_argument("--custom-columns", type=str, nargs="*", default=None,
                        help="Additional columns to store in cut.custom dict")
    add_language_arg(parser)
    add_text_tokenizer_args(parser, include_custom_columns=True)


def _require_id_field(row, col):
    value = _get_field(row, col)
    if value is _MISSING or value is None:
        raise ValueError(f"id_column {col!r} is missing or null in row")
    return value


def extract_row_metadata(
    row,
    *,
    id_column=None,
    id_prefix=None,
    text_column=None,
    language_column=None,
    language=None,
    custom_columns=None,
    constant_custom=None,
    derived_custom=None,
    fallback_id=None,
):
    """Extract metadata from a columnar row (parquet or arrow)."""
    if not id_column:
        row_id = fallback_id
    elif isinstance(id_column, (list, tuple)):
        row_id = "_".join(str(_require_id_field(row, c)) for c in id_column)
    else:
        row_id = str(_require_id_field(row, id_column))
    if id_prefix and row_id:
        row_id = f"{id_prefix}_{row_id}"

    text = None
    if text_column:
        value = _get_field(row, text_column)
        if value is not _MISSING and value is not None:
            text = value

    lang = None
    if language_column:
        value = _get_field(row, language_column)
        if value is not _MISSING and value is not None:
            lang = value
    if lang is None:
        lang = language

    custom = dict(constant_custom or {})
    if custom_columns:
        for col in custom_columns:
            val = _get_field(row, col)
            if val is not _MISSING and val is not None:
                custom[col] = val
    if derived_custom:
        for key, template in derived_custom.items():
            custom[key] = _render_custom_template(
                template,
                row=row,
                row_id=row_id,
                custom=custom,
            )

    return row_id, text, lang, custom


_CUSTOM_TEMPLATE_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _render_custom_template(template: str, *, row, row_id, custom: dict) -> str:
    def replace(match):
        key = match.group(1)
        if key in {"id", "row_id"}:
            return "" if row_id is None else str(row_id)
        if key in custom:
            return str(custom[key])
        value = _get_field(row, key)
        if value is _MISSING or value is None:
            raise ValueError(f"derived custom template references missing field {key!r}")
        return str(value)

    return _CUSTOM_TEMPLATE_RE.sub(replace, template)


def external_metadata_lookup_id(row_id: str | None, id_prefix: str | None) -> str | None:
    """Return the unprefixed row ID used by external metadata side tables."""
    if row_id is None or not id_prefix:
        return row_id
    prefix = f"{id_prefix}_"
    if row_id.startswith(prefix):
        return row_id[len(prefix):]
    return row_id


def _get_field(row, path: str):
    """Resolve a flat or dotted path against a row dict-like object."""
    if "." not in path:
        return row.get(path, _MISSING) if isinstance(row, dict) else _MISSING
    cur = row
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return _MISSING
        cur = cur[key]
    return cur


def _coerce_optional_float_field(row, path: str | None, *, field_name: str) -> float | None:
    if not path:
        return None
    value = _get_field(row, path)
    if value is _MISSING or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} {path!r} must be numeric or null")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} {path!r} must be finite")
    return float(value)


def _coerce_clip_num_field(row, path: str) -> int:
    value = _get_field(row, path)
    if value is _MISSING or value is None:
        raise ValueError(f"clip_num_column {path!r} is missing or null in row")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"clip_num_column {path!r} must be integer-like")
    if not math.isfinite(value):
        raise ValueError(f"clip_num_column {path!r} must be finite")
    clip_num = int(value)
    if clip_num != value:
        raise ValueError(f"clip_num_column {path!r} must be integer-like")
    if clip_num < 0:
        raise ValueError(f"clip_num_column {path!r} must be >= 0")
    return clip_num


def derive_timestamp_clip_num(
    *,
    row_id: object,
    clip_start: float,
    clip_duration: float | None = None,
) -> int:
    """Build a stable int64 tie-breaker for timestamp-ordered interleave rows."""
    payload = "\0".join(
        [
            str(row_id),
            float(clip_start).hex(),
            "" if clip_duration is None else float(clip_duration).hex(),
        ]
    ).encode("utf-8", errors="surrogatepass")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def extract_interleave_identity(
    row,
    *,
    row_id: object,
    chunk_idx: int = 0,
    source_id_column: str | None = None,
    clip_num_column: str | None = None,
    clip_start: float | None = None,
    clip_duration: float | None = None,
    input_clip_id_parser=None,
) -> tuple[str, int]:
    """Resolve interleave identity from explicit columns or row ID parsing."""
    if source_id_column or clip_num_column:
        if not source_id_column:
            raise ValueError(
                "clip_num_column requires source_id_column"
            )
        source_id = _get_field(row, source_id_column)
        if source_id is _MISSING or source_id is None:
            raise ValueError(
                f"source_id_column {source_id_column!r} is missing or null in row"
            )
        clip_num = (
            _coerce_clip_num_field(row, clip_num_column)
            if clip_num_column
            else derive_timestamp_clip_num(
                row_id=row_id,
                clip_start=_require_clip_start_for_derived_clip_num(
                    source_id_column=source_id_column,
                    clip_start=clip_start,
                ),
                clip_duration=clip_duration,
            )
        )
        return str(source_id), clip_num

    return resolve_input_source_and_clip_num(
        row_id,
        chunk_idx=chunk_idx,
        input_clip_id_parser=input_clip_id_parser,
    )


def _require_clip_start_for_derived_clip_num(
    *,
    source_id_column: str,
    clip_start: float | None,
) -> float:
    if clip_start is None:
        raise ValueError(
            "source_id_column without clip_num_column requires clip_start_column "
            f"metadata; cannot derive a stable timestamp tie-breaker for {source_id_column!r}"
        )
    return clip_start


def extract_clip_timestamps(
    row,
    *,
    clip_start_column: str | None = None,
    clip_end_column: str | None = None,
    clip_duration_column: str | None = None,
) -> tuple[float | None, float | None]:
    clip_start = _coerce_optional_float_field(
        row, clip_start_column, field_name="clip_start_column"
    )
    clip_end = _coerce_optional_float_field(
        row, clip_end_column, field_name="clip_end_column"
    )
    clip_duration = _coerce_optional_float_field(
        row, clip_duration_column, field_name="clip_duration_column"
    )
    if clip_duration is not None and clip_duration <= 0:
        raise ValueError(
            f"clip_duration_column {clip_duration_column!r} must be > 0 when set"
        )
    if clip_start is not None and clip_end is not None and clip_end < clip_start:
        raise ValueError(
            f"clip_end_column {clip_end_column!r} must be >= clip_start_column {clip_start_column!r}"
        )
    if clip_duration is None and clip_end is not None and clip_start is not None:
        clip_duration = clip_end - clip_start
    return clip_start, clip_duration


def _projected_columns(*cols) -> list[str]:
    """Plan a minimal projection for parquet reading."""
    requested: list[str] = []
    for spec in cols:
        if spec is None:
            continue
        items = spec if isinstance(spec, (list, tuple)) else [spec]
        for col in items:
            if col and col not in requested:
                requested.append(col)

    out = []
    requested_set = set(requested)
    for col in requested:
        parts = col.split(".")
        ancestors = {".".join(parts[:i]) for i in range(1, len(parts))}
        if not ancestors.intersection(requested_set):
            out.append(col)
    return out


def required_column_roots(*columns) -> set[str]:
    """Return required top-level roots for flat or dotted column specs."""
    roots: set[str] = set()
    for spec in columns:
        if spec is None:
            continue
        items = spec if isinstance(spec, (list, tuple)) else [spec]
        for col in items:
            if col:
                roots.add(str(col).split(".", 1)[0])
    return roots


def validate_columnar_schema_roots(
    *,
    available_roots,
    required_columns,
    optional_columns,
    source_path: str,
    source_kind: str,
    logger,
) -> None:
    """Validate required roots and log optional missing roots for a columnar source."""
    available = set(available_roots)
    required = required_column_roots(*required_columns)
    missing_required = sorted(required - available)
    if missing_required:
        raise RuntimeError(
            f"{source_kind} preflight failed: required column roots are missing from "
            f"{source_path}: {missing_required}. Available columns: {sorted(available)}"
        )

    optional = required_column_roots(*optional_columns)
    missing_optional = sorted(optional - available)
    if missing_optional:
        logger.info(
            "%s preflight: optional column roots missing from %s and will be treated as absent: %s",
            source_kind,
            source_path,
            missing_optional,
        )
