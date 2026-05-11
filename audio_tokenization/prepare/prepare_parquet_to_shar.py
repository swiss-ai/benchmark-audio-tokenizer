"""Convert HF-style parquet shards (with audio bytes) to Lhotse Shar format.

Each row carries an ``audio`` struct ``{bytes: Binary, sampling_rate: Int64}``
plus an id, optional ``text``, and optional ``duration``. Workers are assigned
*whole parquet files* (not rows) to keep input I/O per worker contiguous.

Invocation goes through the Hydra stage adapter:
``python -m audio_tokenization run dataset=<name> stage=convert`` with a
``configs/pipeline/dataset/<name>.yaml`` that picks the parquet recipe.
"""

from collections import Counter
import logging
import time
from pathlib import Path

from audio_tokenization.config.schema import PrepareSpec
from audio_tokenization.prepare.audio_ops import (
    apply_audio_pipeline,
    build_recording_from_audio_bytes,
    extract_audio_bytes,
    write_cut_to_shar,
)
from audio_tokenization.prepare.columnar import (
    _get_field,
    ColumnarWorkerArgs,
    external_metadata_lookup_id,
    extract_clip_timestamps,
    extract_interleave_identity,
    extract_row_metadata,
    _projected_columns,
    validate_columnar_schema_roots,
)
from audio_tokenization.prepare.constants import _MISSING
from audio_tokenization.prepare.identity import set_interleave_metadata
from audio_tokenization.prepare.metadata import (
    load_external_metadata,
    resolve_sample_text_and_custom,
)
from audio_tokenization.prepare.runtime import (
    check_worker_reuse,
    distribute_round_robin,
    ensure_worker_assignment,
    init_worker_process,
    maybe_log_worker_progress,
    coerce_resolved_inputs,
    run_pool_and_finalize,
    validate_prepare_runtime,
    write_prepare_state_for_spec,
    write_worker_result,
)
from audio_tokenization.prepare.text_ops import (
    load_text_tokenizer,
    make_text_tokenize_fn,
)
from audio_tokenization.utils.clip_id_parsers import get_clip_id_parser
from audio_tokenization.prepare.streaming import iter_parquet_rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_ITEMS_KEY = "resolved_parquets"
_EXTERNAL_METADATA = None
_DEFAULT_READ_BATCH_SIZE = 256


def _coerce_chunk_float(chunk: dict, key: str) -> float:
    value = chunk.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"chunk field {key!r} must be numeric")
    value = float(value)
    if value < 0:
        raise ValueError(f"chunk field {key!r} must be >= 0")
    return value


def _coerce_chunk_num(chunk: dict, fallback: int) -> int:
    value = chunk.get("clip_num", fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("chunk field 'clip_num' must be integer-like")
    clip_num = int(value)
    if clip_num != value or clip_num < 0:
        raise ValueError("chunk field 'clip_num' must be a non-negative integer")
    return clip_num


def _stable_chunk_id(source_id: str, clip_num: int, clip_start: float, clip_duration: float) -> str:
    start_ms = round(float(clip_start) * 1000)
    duration_ms = round(float(clip_duration) * 1000)
    return f"{source_id}_{clip_num:06d}_{start_ms:012d}_{duration_ms:012d}"


def _convert_worker(args: ColumnarWorkerArgs):
    """Convert rows from assigned parquet files to Shar.

    Each worker writes to its own ``worker_XX/`` directory. Resume is complete
    only when ``worker_XX/_SUCCESS`` exists.
    """
    worker_id = args.worker_id
    parquet_paths = args.input_paths
    shar_dir = args.shar_dir
    target_sr = args.target_sr
    shard_size = args.shard_size
    shar_format = args.shar_format
    id_column = args.id_column
    id_prefix = args.id_prefix
    audio_column = args.audio_column
    text_column = args.text_column
    duration_column = args.duration_column
    language_column = args.language_column
    language = args.language
    custom_columns = args.custom_columns
    constant_custom = args.constant_custom
    derived_custom = args.derived_custom
    text_tokenize_custom_columns = args.text_tokenize_custom_columns
    text_tokenizer = load_text_tokenizer(args.text_tokenizer_path)
    resampling_backend = args.resampling_backend
    input_clip_id_parser_name = args.input_clip_id_parser_name
    source_id_column = args.source_id_column
    clip_num_column = args.clip_num_column
    clip_start_column = args.clip_start_column
    clip_end_column = args.clip_end_column
    clip_duration_column = args.clip_duration_column
    read_batch_size = args.read_batch_size
    chunks_column = args.chunks_column

    reused = check_worker_reuse(worker_id, shar_dir)
    if reused is not None:
        return reused
    init_worker_process(resampling_backend)

    from lhotse import SupervisionSegment, fastcopy
    from lhotse.shar import SharWriter

    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    t0 = time.time()
    written = skipped = errors = 0
    next_log_at = 1000
    total_duration_sec = 0.0
    runtime_counts = Counter()
    _tokenize_text = make_text_tokenize_fn(text_tokenizer, text_tokenize_custom_columns) if text_tokenizer is not None else None
    input_clip_id_parser = (
        get_clip_id_parser(input_clip_id_parser_name)
        if input_clip_id_parser_name
        else None
    )
    external_metadata = _EXTERNAL_METADATA

    with SharWriter(
        output_dir=str(worker_dir),
        fields={"recording": shar_format},
        shard_size=shard_size,
    ) as writer:
        for pq_path in parquet_paths:
            pq_name = Path(pq_path).name
            logger.info(f"Worker {worker_id}: reading {pq_name}")
            read_columns = _projected_columns(
                audio_column,
                text_column,
                duration_column,
                source_id_column,
                clip_num_column,
                clip_start_column,
                clip_end_column,
                clip_duration_column,
                chunks_column,
                id_column,
                language_column,
                custom_columns,
            )

            row_idx = 0
            for row in iter_parquet_rows(
                pq_path,
                columns=read_columns,
                batch_size=read_batch_size,
            ):
                fallback_id = f"{Path(pq_path).stem}_{row_idx}" if not id_column else None
                row_id, default_text, lang, custom = extract_row_metadata(
                    row,
                    id_column=id_column,
                    id_prefix=id_prefix,
                    text_column=text_column,
                    language_column=language_column,
                    language=language,
                    custom_columns=custom_columns,
                    constant_custom=constant_custom,
                    derived_custom=derived_custom,
                    fallback_id=fallback_id,
                )
                try:
                    duration = _get_field(row, duration_column) if duration_column else None
                    if duration is not _MISSING and duration is not None:
                        if not isinstance(duration, (int, float)):
                            raise TypeError(
                                f"duration_column {duration_column!r} must be numeric or null; "
                                f"got {type(duration).__name__}"
                            )
                        if duration <= 0:
                            skipped += 1
                            runtime_counts["skipped_non_positive_duration"] += 1
                            continue

                    audio_bytes = extract_audio_bytes(row[audio_column])
                    if not audio_bytes:
                        skipped += 1
                        runtime_counts["skipped_empty_audio"] += 1
                        continue

                    recording = build_recording_from_audio_bytes(
                        audio_bytes,
                        row_id,
                        runtime_counts=runtime_counts,
                    )

                    cut = recording.to_cut()

                    text, custom = resolve_sample_text_and_custom(
                        external_metadata_lookup_id(row_id, id_prefix),
                        default_text=default_text,
                        default_custom=custom,
                        external_metadata=external_metadata,
                        stats=runtime_counts,
                    )
                    if custom:
                        cut.custom = custom

                    if chunks_column:
                        chunks = _get_field(row, chunks_column)
                        if not isinstance(chunks, list) or not chunks:
                            skipped += 1
                            runtime_counts["skipped_empty_chunks"] += 1
                            continue
                        if source_id_column:
                            source_id = _get_field(row, source_id_column)
                            if source_id is _MISSING or source_id is None:
                                raise ValueError(
                                    f"source_id_column {source_id_column!r} is missing or null in row"
                                )
                            source_id = str(source_id)
                        else:
                            source_id = str(row_id)

                        for chunk_idx, chunk in enumerate(chunks):
                            if not isinstance(chunk, dict):
                                raise TypeError("chunk entries must be structs")
                            clip_start_val = _coerce_chunk_float(
                                chunk, "clip_start_sec"
                            )
                            clip_duration_val = _coerce_chunk_float(
                                chunk, "clip_duration_sec"
                            )
                            if clip_duration_val <= 0:
                                raise ValueError(
                                    "chunk field 'clip_duration_sec' must be > 0"
                                )
                            clip_num = _coerce_chunk_num(chunk, chunk_idx)
                            clip_id = str(
                                chunk.get("clip_id")
                                or _stable_chunk_id(
                                    source_id,
                                    clip_num,
                                    clip_start_val,
                                    clip_duration_val,
                                )
                            )
                            try:
                                subcut = cut.truncate(
                                    offset=clip_start_val,
                                    duration=clip_duration_val,
                                    preserve_id=False,
                                )
                            except TypeError:
                                subcut = cut.truncate(
                                    offset=clip_start_val,
                                    duration=clip_duration_val,
                                )

                            subcut = fastcopy(subcut, id=clip_id)
                            subcut.custom = dict(custom or {})
                            subcut.custom["source_recording_id"] = str(row_id)
                            subcut.custom["global_offset_sec"] = clip_start_val
                            chunk_lang = chunk.get("lang", lang)
                            if chunk_lang is not None:
                                subcut.custom["lang"] = chunk_lang
                            if text:
                                subcut.supervisions = [SupervisionSegment(
                                    id=subcut.id,
                                    recording_id=subcut.recording_id,
                                    start=0.0,
                                    duration=subcut.duration,
                                    text=text,
                                    language=chunk_lang,
                                )]

                            set_interleave_metadata(
                                subcut,
                                source_id,
                                clip_num,
                                clip_start=clip_start_val,
                                clip_duration=clip_duration_val,
                            )
                            subcut, skip, decoded_audio = apply_audio_pipeline(
                                subcut,
                                target_sr=target_sr,
                                tokenize_fn=_tokenize_text,
                                runtime_counts=runtime_counts,
                            )
                            if skip:
                                skipped += 1
                                continue
                            write_cut_to_shar(
                                writer,
                                subcut,
                                audio=decoded_audio,
                                runtime_counts=runtime_counts,
                            )
                            written += 1
                            total_duration_sec += subcut.duration
                            next_log_at = maybe_log_worker_progress(
                                logger=logger,
                                worker_id=worker_id,
                                written=written,
                                skipped=skipped,
                                errors=errors,
                                t0=t0,
                                next_log_at=next_log_at,
                            )
                        continue
                    if text:
                        cut.supervisions = [SupervisionSegment(
                            id=cut.id,
                            recording_id=cut.recording_id,
                            start=0.0,
                            duration=cut.duration,
                            text=text,
                            language=lang,
                        )]

                    cut, skip, decoded_audio = apply_audio_pipeline(
                        cut,
                        target_sr=target_sr,
                        tokenize_fn=_tokenize_text,
                        runtime_counts=runtime_counts,
                    )
                    if skip:
                        skipped += 1
                        continue

                    clip_start_val, clip_duration_val = extract_clip_timestamps(
                        row,
                        clip_start_column=clip_start_column,
                        clip_end_column=clip_end_column,
                        clip_duration_column=clip_duration_column,
                    )
                    source_id, clip_num = extract_interleave_identity(
                        row,
                        row_id=row_id,
                        # Parquet rows are already prepared clips. ``row_idx`` is
                        # only a fallback ID counter, not an output chunk index.
                        chunk_idx=0,
                        source_id_column=source_id_column,
                        clip_num_column=clip_num_column,
                        clip_start=clip_start_val,
                        clip_duration=clip_duration_val,
                        input_clip_id_parser=input_clip_id_parser,
                    )
                    set_interleave_metadata(
                        cut,
                        source_id,
                        clip_num,
                        clip_start=clip_start_val,
                        clip_duration=clip_duration_val,
                    )
                    write_cut_to_shar(
                        writer,
                        cut,
                        audio=decoded_audio,
                        runtime_counts=runtime_counts,
                    )
                    written += 1
                    total_duration_sec += cut.duration
                    next_log_at = maybe_log_worker_progress(
                        logger=logger,
                        worker_id=worker_id,
                        written=written,
                        skipped=skipped,
                        errors=errors,
                        t0=t0,
                        next_log_at=next_log_at,
                    )

                except Exception as e:
                    errors += 1
                    runtime_counts["processing_errors"] += 1
                    if errors <= 5:
                        logger.warning(f"Worker {worker_id} error on {row_id}: {e}")
                finally:
                    row_idx += 1

    return write_worker_result(
        worker_id=worker_id, worker_dir=worker_dir,
        written=written, skipped=skipped, errors=errors,
        total_duration_sec=total_duration_sec,
        runtime_counts=runtime_counts, t0=t0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve(spec: PrepareSpec) -> tuple[list[str], dict]:
    """Resolve parquet input shards for this prepare family."""
    i = spec.input
    parquet_dir = Path(i.parquet_dir)
    resolved = sorted(str(p) for p in parquet_dir.glob(i.parquet_glob))
    if not resolved:
        raise FileNotFoundError(f"No files match {parquet_dir / i.parquet_glob}")
    return resolved, {
        "family": spec.family,
        "parquet_dir": str(parquet_dir),
        "parquet_glob": i.parquet_glob,
        "resolved_inputs": resolved,
    }


def preflight(
    spec: PrepareSpec,
    *,
    runtime_validator=validate_prepare_runtime,
) -> None:
    """Validate generic parquet prepare prerequisites."""
    i, o = spec.input, spec.output
    parquet_dir = Path(i.parquet_dir)
    if not parquet_dir.is_dir():
        raise NotADirectoryError(f"Parquet input dir not found: {parquet_dir}")
    runtime_validator(
        resampling_backend=o.resampling_backend,
        require_ffmpeg=True,
        text_tokenizer_path=o.text_tokenizer,
    )


def _preflight_prepare(spec: PrepareSpec, resolved: list[str]) -> None:
    import pyarrow.parquet as pq

    m = spec.metadata
    sample_path = resolved[0]
    validate_columnar_schema_roots(
        available_roots=pq.ParquetFile(sample_path).schema_arrow.names,
        required_columns=(
            m.audio_column,
            m.id_column,
            getattr(m, "source_id_column", None),
            getattr(m, "clip_num_column", None),
            getattr(m, "clip_start_column", None),
            getattr(m, "clip_end_column", None),
            getattr(m, "clip_duration_column", None),
            getattr(m, "chunks_column", None),
        ),
        optional_columns=(
            m.text_column,
            m.duration_column,
            m.language_column,
            m.custom_columns,
        ),
        source_path=sample_path,
        source_kind="Parquet",
        logger=logger,
    )

    preflight(spec, runtime_validator=validate_prepare_runtime)


def run(spec: PrepareSpec, *, resolved_inputs: list[str] | None = None):
    """Execute parquet prepare for a typed PrepareSpec."""
    o, m = spec.output, spec.metadata
    shar_dir = Path(o.shar_dir)

    resolved = coerce_resolved_inputs(spec, resolved_inputs)

    _preflight_prepare(spec, resolved)

    shar_dir.mkdir(parents=True, exist_ok=True)
    write_prepare_state_for_spec(spec)

    num_workers = ensure_worker_assignment(
        shar_dir, resolved, o.num_workers, _ITEMS_KEY, "parquet files",
    )

    logger.info(f"Found {len(resolved)} parquet files, using {num_workers} workers")
    logger.info(f"Output: {shar_dir}")

    worker_parquets = distribute_round_robin(resolved, num_workers)

    global _EXTERNAL_METADATA
    if m.external_metadata:
        _EXTERNAL_METADATA = load_external_metadata(
            m.external_metadata,
            tuple(m.custom_fields) if m.custom_fields else None,
            id_field=m.id_field,
            text_field=m.text_field,
        )
        mp_start_method = "fork"
        logger.info("Using fork start method for COW sharing of external metadata")
    else:
        _EXTERNAL_METADATA = None
        mp_start_method = o.mp_start_method

    worker_args = [
        ColumnarWorkerArgs(
            worker_id=wid,
            input_paths=tuple(parquets),
            shar_dir=str(shar_dir),
            target_sr=o.target_sr,
            shard_size=o.shard_size,
            shar_format=o.shar_format,
            id_column=tuple(m.id_column) if isinstance(m.id_column, list) else m.id_column,
            id_prefix=m.id_prefix,
            audio_column=m.audio_column,
            text_column=m.text_column,
            duration_column=m.duration_column,
            language_column=m.language_column,
            language=m.language,
            custom_columns=tuple(m.custom_columns) if m.custom_columns else None,
            constant_custom=dict(m.constant_custom),
            derived_custom=dict(m.derived_custom),
            text_tokenize_custom_columns=(
                tuple(m.text_tokenize_custom_columns)
                if m.text_tokenize_custom_columns
                else None
            ),
            text_tokenizer_path=o.text_tokenizer,
            resampling_backend=o.resampling_backend,
            input_clip_id_parser_name=m.input_clip_id_parser,
            source_id_column=m.source_id_column,
            clip_num_column=m.clip_num_column,
            clip_start_column=m.clip_start_column,
            clip_end_column=m.clip_end_column,
            clip_duration_column=m.clip_duration_column,
            read_batch_size=o.read_batch_size,
            chunks_column=m.chunks_column,
        )
        for wid, parquets in enumerate(worker_parquets)
        if parquets
    ]

    run_pool_and_finalize(
        _convert_worker,
        worker_args,
        shar_dir,
        num_workers,
        mp_start_method=mp_start_method,
    )


