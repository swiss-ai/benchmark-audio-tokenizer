"""Convert HF-style arrow shards (with audio bytes) to Lhotse Shar format.

Each row has an audio struct with raw bytes and optional text. Workers are
assigned *whole arrow files* (not rows) via round-robin.

Invocation goes through the Hydra stage adapter:
``python -m audio_tokenization run dataset=<name> stage=convert`` with a
``configs/pipeline/dataset/<name>.yaml`` that picks the HF arrow recipe.
"""

from collections import Counter
import logging
import time
from pathlib import Path

from audio_tokenization.prepare.audio_ops import (
    apply_audio_pipeline,
    build_recording_from_audio_bytes,
    extract_audio_bytes,
    write_cut_to_shar,
)
from audio_tokenization.prepare.cli import expand_path_patterns
from audio_tokenization.prepare.columnar import (
    ColumnarWorkerArgs,
    external_metadata_lookup_id,
    extract_clip_timestamps,
    extract_interleave_identity,
    extract_row_metadata,
    validate_columnar_schema_roots,
)
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
from audio_tokenization.prepare.streaming import iter_arrow_rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_ITEMS_KEY = "resolved_arrows"
_EXTERNAL_METADATA = None
_DEFAULT_READ_BATCH_SIZE = 256


def _convert_worker(args: ColumnarWorkerArgs):
    """Convert rows from assigned arrow files to Shar.

    Each worker writes to its own ``worker_XX/`` directory. Resume is complete
    only when ``worker_XX/_SUCCESS`` exists.
    """
    worker_id = args.worker_id
    arrow_paths = args.input_paths
    shar_dir = args.shar_dir
    target_sr = args.target_sr
    shard_size = args.shard_size
    shar_format = args.shar_format
    id_column = args.id_column
    id_prefix = args.id_prefix
    audio_column = args.audio_column
    text_column = args.text_column
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

    reused = check_worker_reuse(worker_id, shar_dir)
    if reused is not None:
        return reused
    init_worker_process(resampling_backend)

    from lhotse import SupervisionSegment
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
        for arrow_path in arrow_paths:
            arrow_name = Path(arrow_path).name
            arrow_stem = Path(arrow_path).stem
            logger.info(f"Worker {worker_id}: reading {arrow_name}")
            row_idx = 0
            for row in iter_arrow_rows(arrow_path, batch_size=read_batch_size):
                current_row_idx = row_idx
                fallback_id = f"{arrow_stem}_{current_row_idx}" if not id_column else None
                row_idx += 1
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
                        # HF/Arrow rows are already prepared clips. The row
                        # counter is only for fallback IDs, not chunk numbering.
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

    return write_worker_result(
        worker_id=worker_id, worker_dir=worker_dir,
        written=written, skipped=skipped, errors=errors,
        total_duration_sec=total_duration_sec,
        runtime_counts=runtime_counts, t0=t0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve(spec) -> tuple[list[str], dict]:
    """Resolve Arrow input shards for this prepare family."""
    i = spec.input
    if i.arrow_files:
        resolved = expand_path_patterns(i.arrow_files)
        input_summary = {
            "family": spec.family,
            "arrow_files": list(i.arrow_files),
            "resolved_inputs": resolved,
        }
    elif i.arrow_dir:
        arrow_dir = Path(i.arrow_dir)
        resolved = sorted(str(p) for p in arrow_dir.glob(i.arrow_glob))
        input_summary = {
            "family": spec.family,
            "arrow_dir": str(arrow_dir),
            "arrow_glob": i.arrow_glob,
            "resolved_inputs": resolved,
        }
    else:
        raise ValueError("Either arrow_files or arrow_dir must be provided")
    if not resolved:
        raise FileNotFoundError("No arrow files resolved for HF input")
    return resolved, input_summary


def preflight(
    spec,
    *,
    runtime_validator=validate_prepare_runtime,
) -> None:
    """Validate generic HF Arrow prepare prerequisites."""
    i, o = spec.input, spec.output
    if not i.arrow_files and i.arrow_dir is not None:
        arrow_dir = Path(i.arrow_dir)
        if not arrow_dir.is_dir():
            raise NotADirectoryError(f"Arrow input dir not found: {arrow_dir}")
    runtime_validator(
        resampling_backend=o.resampling_backend,
        require_ffmpeg=True,
        text_tokenizer_path=o.text_tokenizer,
    )


def _preflight_prepare(spec, resolved: list[str]) -> None:
    import pyarrow.ipc as ipc

    m = spec.metadata
    sample_path = resolved[0]
    with open(sample_path, "rb") as f:
        schema_names = ipc.open_stream(f).schema.names

    validate_columnar_schema_roots(
        available_roots=schema_names,
        required_columns=(
            m.audio_column,
            m.id_column,
            getattr(m, "source_id_column", None),
            getattr(m, "clip_num_column", None),
            getattr(m, "clip_start_column", None),
            getattr(m, "clip_end_column", None),
            getattr(m, "clip_duration_column", None),
        ),
        optional_columns=(
            m.text_column,
            m.language_column,
            m.custom_columns,
        ),
        source_path=sample_path,
        source_kind="Arrow",
        logger=logger,
    )

    preflight(spec, runtime_validator=validate_prepare_runtime)


def run(spec, *, resolved_inputs: list[str] | None = None):
    """Execute HF arrow prepare for a typed PrepareSpec."""
    i, o, m = spec.input, spec.output, spec.metadata
    shar_dir = Path(o.shar_dir)

    resolved = coerce_resolved_inputs(spec, resolved_inputs)

    _preflight_prepare(spec, resolved)

    shar_dir.mkdir(parents=True, exist_ok=True)
    write_prepare_state_for_spec(spec)

    num_workers = ensure_worker_assignment(
        shar_dir, resolved, o.num_workers, _ITEMS_KEY, "arrow files",
    )

    logger.info(f"Found {len(resolved)} arrow files, using {num_workers} workers")
    logger.info(f"Output: {shar_dir}")

    worker_arrows = distribute_round_robin(resolved, num_workers)
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
        mp_start_method = "forkserver"

    worker_args = [
        ColumnarWorkerArgs(
            worker_id=wid,
            input_paths=tuple(arrows),
            shar_dir=str(shar_dir),
            target_sr=o.target_sr,
            shard_size=o.shard_size,
            shar_format=o.shar_format,
            id_column=tuple(m.id_column) if isinstance(m.id_column, list) else m.id_column,
            id_prefix=m.id_prefix,
            audio_column=m.audio_column,
            text_column=m.text_column,
            duration_column=None,
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
        )
        for wid, arrows in enumerate(worker_arrows)
        if arrows
    ]

    run_pool_and_finalize(
        _convert_worker,
        worker_args,
        shar_dir,
        num_workers,
        mp_start_method=mp_start_method,
    )


