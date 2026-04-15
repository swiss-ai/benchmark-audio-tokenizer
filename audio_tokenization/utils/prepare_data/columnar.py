"""Columnar row metadata helpers for parquet and HF arrow prepare paths."""

from __future__ import annotations

from audio_tokenization.utils.prepare_data.cli import (
    add_language_arg,
    add_text_tokenizer_args,
)
from audio_tokenization.utils.prepare_data.constants import _MISSING


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
    text_column=None,
    language_column=None,
    language=None,
    custom_columns=None,
    fallback_id=None,
):
    """Extract metadata from a columnar row (parquet or arrow)."""
    if not id_column:
        row_id = fallback_id
    elif isinstance(id_column, (list, tuple)):
        row_id = "_".join(str(_require_id_field(row, c)) for c in id_column)
    else:
        row_id = str(_require_id_field(row, id_column))

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

    custom = {}
    if custom_columns:
        for col in custom_columns:
            val = _get_field(row, col)
            if val is not _MISSING and val is not None:
                custom[col] = val

    return row_id, text, lang, custom


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
