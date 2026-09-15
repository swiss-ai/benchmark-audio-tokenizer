"""Preflight checks for self-contained SFT audio packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from audio_tokenization.sft.materialize import (
    coerce_messages,
    ordered_audio_ids,
    render_structured_audio_attachments,
    select_conversation_columns,
)


@dataclass(frozen=True)
class SftPackagePreflightReport:
    conversation_files: int
    conversation_rows: int
    conversation_row_groups: int
    unique_audio_ids: int
    media_rows: int | None = None


def validate_sft_package(
    *,
    conversations_dir: str | Path,
    conversations_glob: str,
    messages_column: str,
    audio_ids_column: str,
    audio_placeholder: str,
    media_dir: str | Path | None = None,
    media_glob: str = "*.parquet",
    media_id_column: str | None = None,
    media_duration_column: str | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> SftPackagePreflightReport:
    """Validate conversation/audio alignment before expensive SFT work starts.

    This checks the package contract, not the audio payload itself: conversation
    rows must place every referenced audio item in the message sequence, and
    referenced audio IDs must exist in the media parquet metadata and survive
    tokenize duration filters when media metadata is available.
    """
    conversation_report, referenced_audio_ids = _scan_conversations(
        Path(conversations_dir),
        conversations_glob=conversations_glob,
        messages_column=messages_column,
        audio_ids_column=audio_ids_column,
        audio_placeholder=audio_placeholder,
    )
    unique_refs = set(referenced_audio_ids)
    if media_dir is None:
        return conversation_report

    media = _load_media_audio_index(
        Path(media_dir),
        media_glob=media_glob,
        media_id_column=media_id_column,
        media_duration_column=media_duration_column,
    )
    _reject_missing_media_refs(unique_refs, media)
    _reject_duration_filtered_refs(
        unique_refs,
        media,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    return SftPackagePreflightReport(
        conversation_files=conversation_report.conversation_files,
        conversation_rows=conversation_report.conversation_rows,
        conversation_row_groups=conversation_report.conversation_row_groups,
        unique_audio_ids=conversation_report.unique_audio_ids,
        media_rows=len(media),
    )


def _scan_conversations(
    conversations_dir: Path,
    *,
    conversations_glob: str,
    messages_column: str,
    audio_ids_column: str,
    audio_placeholder: str,
) -> tuple[SftPackagePreflightReport, list[str]]:
    paths = sorted(conversations_dir.glob(conversations_glob))
    if not paths:
        raise FileNotFoundError(
            f"No SFT conversation parquet files matching {conversations_glob!r} "
            f"under {conversations_dir}"
        )

    rows = 0
    row_groups = 0
    audio_ids: list[str] = []
    for path in paths:
        pf = pq.ParquetFile(path)
        selected = select_conversation_columns(
            pf,
            path=path,
            columns=["sample_id", messages_column, audio_ids_column],
            required=("sample_id", messages_column),
        )
        row_groups += pf.num_row_groups
        for batch in pf.iter_batches(columns=selected):
            for row in batch.to_pylist():
                rows += 1
                sample_id = str(row["sample_id"])
                messages = coerce_messages(
                    row[messages_column],
                    sample_id=sample_id,
                    column=messages_column,
                )
                row_audio_ids = ordered_audio_ids(
                    row.get(audio_ids_column),
                    messages=messages,
                    sample_id=sample_id,
                )
                _validate_audio_placement(
                    sample_id=sample_id,
                    messages=messages,
                    audio_ids=row_audio_ids,
                    audio_placeholder=audio_placeholder,
                )
                audio_ids.extend(row_audio_ids)

    if rows <= 0:
        raise ValueError(f"SFT conversations under {conversations_dir} contain no rows")
    return (
        SftPackagePreflightReport(
            conversation_files=len(paths),
            conversation_rows=rows,
            conversation_row_groups=row_groups,
            unique_audio_ids=len(set(audio_ids)),
        ),
        audio_ids,
    )




def _validate_audio_placement(
    *,
    sample_id: str,
    messages: list[dict[str, Any]],
    audio_ids: list[str],
    audio_placeholder: str,
) -> None:
    rendered = render_structured_audio_attachments(
        messages,
        audio_placeholder=audio_placeholder,
        sample_id=sample_id,
    )
    placement_count = sum(
        str(message.get("content") or "").count(audio_placeholder)
        for message in rendered
    )
    if placement_count != len(audio_ids):
        raise ValueError(
            f"SFT sample {sample_id!r} has {placement_count} audio placements "
            f"but {len(audio_ids)} audio ids"
        )


@dataclass(frozen=True)
class _MediaRow:
    duration_sec: float | None


def _load_media_audio_index(
    media_dir: Path,
    *,
    media_glob: str,
    media_id_column: str | None,
    media_duration_column: str | None,
) -> dict[str, _MediaRow]:
    if not media_dir.is_dir():
        raise FileNotFoundError(f"SFT media parquet dir not found: {media_dir}")
    if not media_id_column:
        raise ValueError("SFT media preflight requires a media id column")
    if "." in media_id_column:
        raise ValueError("SFT media preflight requires a flat media id column")

    index_path = media_dir / "_index.parquet"
    if index_path.is_file():
        index_schema = pq.read_schema(index_path)
        if media_id_column in index_schema.names:
            duration_path = _index_duration_column(index_schema.names, media_duration_column)
            return _read_media_rows(
                [index_path],
                id_path=media_id_column,
                duration_path=duration_path,
            )

    paths = sorted(path for path in media_dir.glob(media_glob) if path.name != "_index.parquet")
    if not paths:
        raise FileNotFoundError(f"No SFT media parquet files matching {media_glob!r} under {media_dir}")
    return _read_media_rows(
        paths,
        id_path=media_id_column,
        duration_path=media_duration_column,
    )


def _index_duration_column(
    column_names: Iterable[str],
    media_duration_column: str | None,
) -> str | None:
    names = set(column_names)
    candidates = []
    if media_duration_column:
        candidates.extend([
            media_duration_column,
            _top_level(media_duration_column),
            media_duration_column.split(".")[-1],
        ])
    candidates.append("duration_sec")
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _read_media_rows(
    paths: list[Path],
    *,
    id_path: str,
    duration_path: str | None,
) -> dict[str, _MediaRow]:
    rows: dict[str, _MediaRow] = {}
    for path in paths:
        pf = pq.ParquetFile(path)
        columns = _media_read_columns(
            pf,
            path=path,
            id_path=id_path,
            duration_path=duration_path,
        )
        for batch in pf.iter_batches(columns=columns):
            for row in batch.to_pylist():
                audio_id = _nested_value(row, id_path)
                if audio_id is None or audio_id == "":
                    raise ValueError(f"{path} contains a media row with empty audio id")
                audio_id = str(audio_id)
                if audio_id in rows:
                    raise ValueError(f"Duplicate SFT media audio_id {audio_id!r}")
                duration = _nested_value(row, duration_path) if duration_path else None
                rows[audio_id] = _MediaRow(
                    duration_sec=None if duration is None else float(duration)
                )
    if not rows:
        raise ValueError(f"SFT media parquets contain no rows: {[str(p) for p in paths[:3]]}")
    return rows


def _media_read_columns(
    pf: pq.ParquetFile,
    *,
    path: Path,
    id_path: str,
    duration_path: str | None,
) -> list[str]:
    available = set(pf.schema_arrow.names)
    columns = [_top_level(id_path)]
    if columns[0] not in available:
        raise ValueError(f"{path} is missing SFT media id column {id_path!r}")
    if duration_path:
        duration_top = _top_level(duration_path)
        if duration_top not in available:
            raise ValueError(f"{path} is missing SFT media duration column {duration_path!r}")
        if duration_top not in columns:
            columns.append(duration_top)
    return columns


def _reject_missing_media_refs(
    audio_ids: set[str],
    media: dict[str, _MediaRow],
) -> None:
    missing = sorted(audio_ids - set(media))
    if missing:
        raise ValueError(
            "SFT conversations reference audio IDs missing from media parquet: "
            f"{len(missing)} missing. First missing audio_ids: {missing[:10]}"
        )


def _reject_duration_filtered_refs(
    audio_ids: set[str],
    media: dict[str, _MediaRow],
    *,
    min_duration: float | None,
    max_duration: float | None,
) -> None:
    filtered: list[tuple[str, float]] = []
    for audio_id in audio_ids:
        duration = media[audio_id].duration_sec
        if duration is None:
            continue
        if min_duration is not None and duration < min_duration:
            filtered.append((audio_id, duration))
        elif max_duration is not None and duration > max_duration:
            filtered.append((audio_id, duration))
    if filtered:
        first = sorted(filtered)[:10]
        raise ValueError(
            "SFT conversations reference audio IDs that tokenize duration filters "
            f"would drop: {len(filtered)} affected. "
            f"min_duration={min_duration}, max_duration={max_duration}. "
            f"First affected audio_ids: {first}"
        )


def _nested_value(row: dict[str, Any], path: str | None) -> Any:
    if path is None:
        return None
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _top_level(path: str) -> str:
    return path.split(".", 1)[0]
