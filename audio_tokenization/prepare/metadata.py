"""Metadata loading and override helpers for prepare scripts."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from audio_tokenization.prepare.constants import MetadataEntry
from audio_tokenization.utils.io import open_compressed


def _strip_compression_suffix(path: Path) -> str:
    """Return the format suffix, ignoring .gz/.zst compression."""
    if path.suffix in (".gz", ".zst"):
        return Path(path.stem).suffix
    return path.suffix


def load_external_metadata(
    path: str,
    custom_fields: tuple[str, ...] | None = None,
    *,
    id_field: str = "id",
    text_field: str = "text",
) -> dict[str, MetadataEntry]:
    """Load transcript metadata from an external file."""
    p = Path(path)
    fmt = _strip_compression_suffix(p)
    result: dict[str, MetadataEntry] = {}

    if fmt == ".tsv":
        import csv

        with open_compressed(p, "rt") as f:
            first_line = ""
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                first_line = line
                break
            if first_line:
                header = first_line.split("\t")
                if id_field in header:
                    reader = csv.DictReader(f, fieldnames=header, delimiter="\t")
                    for row in reader:
                        custom = {k: row[k] for k in (custom_fields or ()) if k in row}
                        result[str(row[id_field])] = (row.get(text_field), custom)
                else:
                    parts = first_line.split("\t", 1)
                    if len(parts) == 2:
                        result[parts[0]] = (parts[1], {})
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            result[parts[0]] = (parts[1], {})

    elif fmt == ".jsonl":
        import orjson

        skipped = 0
        with open_compressed(p, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except (orjson.JSONDecodeError, ValueError):
                    skipped += 1
                    continue
                custom = {k: obj[k] for k in (custom_fields or ()) if k in obj}
                result[str(obj[id_field])] = (obj.get(text_field), custom)
        if skipped:
            logging.getLogger(__name__).warning(
                "Skipped %d malformed lines in %s", skipped, p
            )

    elif fmt == ".csv":
        import csv

        with open_compressed(p, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                custom = {k: row[k] for k in (custom_fields or ()) if k in row}
                result[str(row[id_field])] = (row.get(text_field), custom)

    elif fmt == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(p)
        schema_names = set(parquet_file.schema_arrow.names)
        if id_field not in schema_names:
            raise ValueError(
                f"External metadata {p} is missing required id field {id_field!r}"
            )

        selected = [id_field]
        if text_field in schema_names:
            selected.append(text_field)
        selected.extend(
            field
            for field in (custom_fields or ())
            if field in schema_names and field not in selected
        )

        for batch in parquet_file.iter_batches(columns=selected, batch_size=65536):
            for row in batch.to_pylist():
                row_id = row[id_field]
                if row_id is None:
                    continue
                custom = {
                    k: row[k]
                    for k in (custom_fields or ())
                    if k in row and row[k] is not None
                }
                result[str(row_id)] = (row.get(text_field), custom)

    else:
        raise ValueError(
            f"Unsupported external metadata format: {p.name} "
            "(expected .tsv, .csv, .jsonl, or .parquet, optionally with .gz/.zst compression)"
        )

    logging.getLogger(__name__).info(
        "Loaded %d entries from external metadata: %s", len(result), p
    )
    return result


def lookup_external_metadata(
    metadata: Mapping[str, MetadataEntry],
    sample_id: str,
    *,
    stats: Counter | None = None,
    allow_extensions: Iterable[str] = (),
) -> MetadataEntry:
    """Resolve a sample from an external metadata map."""
    basename = sample_id.rsplit("/", 1)[-1] if "/" in sample_id else sample_id
    candidates = [sample_id, basename] if basename != sample_id else [sample_id]

    for key in candidates:
        if key in metadata:
            return metadata[key]

    for key in candidates:
        for ext in allow_extensions:
            candidate = key + ext
            if candidate in metadata:
                return metadata[candidate]

    if stats is not None:
        stats["external_meta_miss"] += 1
    return None, {}


def resolve_sample_text_and_custom(
    sample_id: str,
    *,
    default_text: str | None = None,
    default_custom: Mapping[str, object] | None = None,
    external_metadata: Mapping[str, MetadataEntry] | None = None,
    stats: Counter | None = None,
    allow_extensions: Iterable[str] = (),
) -> MetadataEntry:
    """Resolve text/custom for a sample, allowing external metadata overrides."""
    text = default_text
    custom = dict(default_custom or {})
    if not external_metadata:
        return text, custom

    ext_text, ext_custom = lookup_external_metadata(
        external_metadata,
        sample_id,
        stats=stats,
        allow_extensions=allow_extensions,
    )
    if ext_text is not None:
        text = ext_text
    if ext_custom:
        custom.update(ext_custom)
    return text, custom


def normalize_optional_path(path: str | Path | None) -> str | None:
    """Normalize an optional path for stable manifest payloads."""
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())
