#!/usr/bin/env python3
"""Convert HF-style parquet shards (with audio bytes) to Lhotse Shar format.

Designed for the SPC-R (Speech Parliament Corpus) dataset which ships as ~130
HuggingFace parquet shards. Each row has:
- ``id``:       ``row{NNNNN}_seg{NNN}`` (source + segment)
- ``duration``: float (some rows have <=0 duration with 0 audio bytes)
- ``audio``:    struct ``{bytes: Binary, sampling_rate: Int64}`` (FLAC at 16kHz)
- ``text``:     str (transcription)

Workers are assigned *whole parquet files* (not rows). With 130 files / 20
workers, each worker handles ~6-7 files (~1001 rows each), giving good balance.

Usage (SPC-R):
    python -m audio_tokenization.prepare.prepare_parquet_to_shar \
        --parquet-dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/spc-r-segmented/train \
        --shar-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/spc_r_segmented_shar \
        --target-sr 24000 \
        --shard-size 2000 \
        --shar-format flac \
        --text-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --num-workers 20
"""

import argparse
from collections import Counter
import logging
import time
from pathlib import Path

from audio_tokenization.prepare.common import (
    _get_field,
    _projected_columns,
    PREPARE_STATE_FILE,
    add_audio_processing_args,
    add_columnar_metadata_args,
    add_external_metadata_args,
    add_input_clip_id_parser_arg,
    add_parallelism_args,
    add_shar_output_args,
    apply_audio_pipeline,
    build_recording_from_audio_bytes,
    check_worker_reuse,
    distribute_round_robin,
    ensure_worker_assignment,
    extract_row_metadata,
    init_worker_process,
    load_external_metadata,
    load_text_tokenizer,
    make_text_tokenize_fn,
    normalize_optional_path,
    resolve_sample_text_and_custom,
    resolve_input_source_and_clip_num,
    run_pool_and_finalize,
    set_universal_cut_id,
    validate_or_write_prepare_state,
    write_worker_result,
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


def _convert_worker(args_tuple):
    """Convert rows from assigned parquet files to Shar.

    Each worker writes to its own ``worker_XX/`` directory. Resume is complete
    only when ``worker_XX/_SUCCESS`` exists.
    """
    (
        worker_id,
        parquet_paths,
        shar_dir,
        target_sr,
        shard_size,
        shar_format,
        id_column,
        audio_column,
        text_column,
        duration_column,
        language_column,
        language,
        custom_columns,
        text_tokenize_custom_columns,
        text_tokenizer,
        resampling_backend,
        input_clip_id_parser_name,
        read_batch_size,
    ) = args_tuple

    reused = check_worker_reuse(worker_id, shar_dir)
    if reused is not None:
        return reused
    init_worker_process(resampling_backend)

    from lhotse import SupervisionSegment
    from lhotse.shar import SharWriter

    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    t0 = time.time()
    written = skipped = errors = 0
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
                    text_column=text_column,
                    language_column=language_column,
                    language=language,
                    custom_columns=custom_columns,
                    fallback_id=fallback_id,
                )
                try:
                    duration = _get_field(row, duration_column) if duration_column else None
                    if isinstance(duration, (int, float)) and duration <= 0:
                        skipped += 1
                        runtime_counts["skipped_non_positive_duration"] += 1
                        continue

                    audio_struct = row[audio_column]
                    audio_bytes = audio_struct["bytes"] if isinstance(audio_struct, dict) else None
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
                        row_id,
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

                    cut, skip = apply_audio_pipeline(
                        cut,
                        target_sr=target_sr,
                        tokenize_fn=_tokenize_text,
                        runtime_counts=runtime_counts,
                    )
                    if skip:
                        skipped += 1
                        continue

                    source_id, clip_num = resolve_input_source_and_clip_num(
                        row_id,
                        input_clip_id_parser=input_clip_id_parser,
                    )
                    set_universal_cut_id(
                        cut,
                        source_id,
                        clip_num,
                        clip_start=getattr(cut, "start", 0.0),
                    )
                    writer.write(cut)
                    written += 1
                    total_duration_sec += cut.duration

                    if written % 1000 == 0:
                        elapsed = time.time() - t0
                        logger.info(
                            f"Worker {worker_id}: {written} written, {skipped} skipped, "
                            f"{errors} errors ({written / elapsed:.1f} samples/s)"
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


def _validate_or_write_prepare_state(args) -> None:
    state_path = args.shar_dir / PREPARE_STATE_FILE
    expected = {
        "parquet_dir": str(Path(args.parquet_dir).resolve()),
        "text_tokenizer": normalize_optional_path(args.text_tokenizer),
        "input_clip_id_parser": args.input_clip_id_parser,
        "external_metadata": normalize_optional_path(args.external_metadata),
        "id_field": args.id_field,
        "text_field": args.text_field,
        "custom_fields": sorted(args.custom_fields) if args.custom_fields else None,
    }
    wrote = validate_or_write_prepare_state(
        state_path,
        expected=expected,
        invariant_keys=(
            "parquet_dir",
            "text_tokenizer",
            "input_clip_id_parser",
            "external_metadata",
            "id_field",
            "text_field",
            "custom_fields",
        ),
        guidance=(
            "Use the same --parquet-dir, --text-tokenizer, "
            "--input-clip-id-parser, --external-metadata, --id-field, "
            "--text-field, and --custom-fields to resume this output "
            "directory, or "
            f"remove {args.shar_dir} and restart from scratch."
        ),
    )
    if wrote:
        logger.info(f"Wrote prepare state: {state_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert HF parquet shards → Lhotse Shar (parallel)",
    )

    # Input
    parser.add_argument("--parquet-dir", type=Path, required=True,
                        help="Directory containing parquet files")
    parser.add_argument("--parquet-glob", type=str, default="*.parquet",
                        help="Glob pattern for parquet files (default: '*.parquet')")

    add_shar_output_args(parser)
    add_audio_processing_args(parser)
    add_columnar_metadata_args(
        parser,
        text_column_default="text",
        duration_column_default="duration",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=_DEFAULT_READ_BATCH_SIZE,
        help=(
            "Rows to materialize at once from each parquet shard. "
            "Lower values reduce worker RSS; default: 256."
        ),
    )
    add_external_metadata_args(parser, include_custom_fields=True)
    add_input_clip_id_parser_arg(parser)
    add_parallelism_args(parser, include_mp_start_method=True)

    args = parser.parse_args(argv)

    # Resolve parquet files
    resolved = sorted(str(p) for p in args.parquet_dir.glob(args.parquet_glob))
    if not resolved:
        raise FileNotFoundError(
            f"No files match {args.parquet_dir / args.parquet_glob}"
        )

    args.shar_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_prepare_state(args)

    num_workers = ensure_worker_assignment(
        args.shar_dir, resolved, args.num_workers, _ITEMS_KEY, "parquet files",
    )

    logger.info(f"Found {len(resolved)} parquet files, using {num_workers} workers")
    logger.info(f"Output: {args.shar_dir}")

    # Distribute parquet files across workers (round-robin)
    worker_parquets = distribute_round_robin(resolved, num_workers)

    # Load text tokenizer before forking (shared via COW across workers)
    text_tokenizer = load_text_tokenizer(args.text_tokenizer)
    global _EXTERNAL_METADATA
    if args.external_metadata:
        _EXTERNAL_METADATA = load_external_metadata(
            args.external_metadata,
            tuple(args.custom_fields) if args.custom_fields else None,
            id_field=args.id_field,
            text_field=args.text_field,
        )
        mp_start_method = "fork"
        logger.info("Using fork start method for COW sharing of external metadata")
    else:
        _EXTERNAL_METADATA = None
        mp_start_method = args.mp_start_method

    worker_args = [
        (
            wid,
            parquets,
            str(args.shar_dir),
            args.target_sr,
            args.shard_size,
            args.shar_format,
            args.id_column,
            args.audio_column,
            args.text_column,
            args.duration_column,
            args.language_column,
            args.language,
            args.custom_columns,
            args.text_tokenize_custom_columns,
            text_tokenizer,
            args.resampling_backend,
            args.input_clip_id_parser,
            args.read_batch_size,
        )
        for wid, parquets in enumerate(worker_parquets)
        if parquets
    ]

    run_pool_and_finalize(
        _convert_worker,
        worker_args,
        args.shar_dir,
        num_workers,
        mp_start_method=mp_start_method,
    )


if __name__ == "__main__":
    main()
