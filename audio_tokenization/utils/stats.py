"""Small aggregation helpers for rank/worker JSON stats."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

StatPath = str | Sequence[str]
FieldMap = Sequence[tuple[str, StatPath]]


def load_json_records(
    paths: Iterable[str | Path],
    *,
    required_key: str | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Load JSON object records from paths, skipping unreadable files."""
    records: list[dict[str, Any]] = []
    for path in sorted(Path(p) for p in paths):
        try:
            record = json.loads(path.read_text())
        except Exception:
            if logger is not None:
                logger.debug("Failed to read %s", path, exc_info=True)
            continue
        if not isinstance(record, dict):
            continue
        if required_key is not None and required_key not in record:
            continue
        records.append(record)
    return records


def get_stat(record: Mapping[str, Any], path: StatPath, default: Any = 0) -> Any:
    """Read a flat or nested value from a stats record."""
    if isinstance(path, str):
        return record.get(path, default)

    current: Any = record
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key, default)
    return current


def sum_mapped_fields(
    records: Iterable[Mapping[str, Any]],
    field_map: FieldMap,
) -> dict[str, Any]:
    """Sum numeric stats according to ``(output_key, input_path)`` pairs."""
    totals = {out_key: 0 for out_key, _ in field_map}
    for record in records:
        for out_key, input_path in field_map:
            totals[out_key] += get_stat(record, input_path, 0) or 0
    return totals


def max_field(
    records: Iterable[Mapping[str, Any]],
    path: StatPath,
    default: float = 0.0,
) -> float:
    """Return the maximum numeric value at ``path`` across records."""
    value = default
    for record in records:
        value = max(value, float(get_stat(record, path, default) or default))
    return value


def sum_counter_fields(
    records: Iterable[Mapping[str, Any]],
    *paths: StatPath,
) -> dict[str, int]:
    """Merge Counter-like mapping fields from records."""
    counter: Counter[str] = Counter()
    for record in records:
        for path in paths:
            value = get_stat(record, path, {})
            if isinstance(value, Mapping):
                counter.update(value)
    return dict(counter)
