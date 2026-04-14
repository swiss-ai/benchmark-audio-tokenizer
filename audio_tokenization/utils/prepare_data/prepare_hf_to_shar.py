#!/usr/bin/env python3
"""Convert HF-style arrow files (with audio bytes) to Lhotse Shar format.

Designed for datasets stored as HuggingFace arrow shards (e.g. People's Speech
with 786 arrow files). Each row has an audio struct with raw bytes and optional
text transcription.

Workers are assigned *whole arrow files* (not rows) via round-robin. With 786
files / 128 workers, each worker handles ~6 files, giving good balance.

Usage (People's Speech):
    python -m audio_tokenization.utils.prepare_data.prepare_hf_to_shar \
        --arrow-dir /path/to/peoples_speech/arrow_files \
        --shar-dir /path/to/output_shar \
        --audio-column audio \
        --text-column text \
        --id-column id \
        --target-sr 24000 \
        --shard-size 2000 \
        --shar-format flac \
        --text-tokenizer /path/to/tokenizer.json \
        --num-workers 128
"""

import argparse
from collections import Counter
import logging
import time
from pathlib import Path

from audio_tokenization.utils.prepare_data.common import (
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
    resolve_sample_text_and_custom,
    resolve_input_source_and_clip_num,
    run_pool_and_finalize,
    set_universal_cut_id,
    write_worker_result,
)
from audio_tokenization.utils.clip_id_parsers import get_clip_id_parser
from audio_tokenization.utils.prepare_data.streaming import iter_arrow_rows

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


def _convert_worker(args_tuple):
    """Convert rows from assigned arrow files to Shar.

    Each worker writes to its own ``worker_XX/`` directory. Resume is complete
    only when ``worker_XX/_SUCCESS`` exists.
    """
    (
        worker_id,
        arrow_paths,
        shar_dir,
        target_sr,
        shard_size,
        shar_format,
        id_column,
        audio_column,
        text_column,
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
        for arrow_path in arrow_paths:
            arrow_name = Path(arrow_path).name
            arrow_stem = Path(arrow_path).stem
            logger.info(f"Worker {worker_id}: reading {arrow_name}")
            row_idx = 0
            for row in iter_arrow_rows(arrow_path, batch_size=read_batch_size):
                fallback_id = f"{arrow_stem}_{row_idx}" if not id_column else None
                row_idx += 1
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
                    audio_struct = row[audio_column]
                    audio_bytes = audio_struct.get("bytes") if isinstance(audio_struct, dict) else None
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

    return write_worker_result(
        worker_id=worker_id, worker_dir=worker_dir,
        written=written, skipped=skipped, errors=errors,
        total_duration_sec=total_duration_sec,
        runtime_counts=runtime_counts, t0=t0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert HF arrow shards → Lhotse Shar (parallel)",
    )

    # Input (use --arrow-files for explicit list, or --arrow-dir + --arrow-glob)
    parser.add_argument("--arrow-dir", type=Path, default=None,
                        help="Directory containing arrow files")
    parser.add_argument("--arrow-glob", type=str, default="*.arrow",
                        help="Glob pattern for arrow files (default: '*.arrow')")
    parser.add_argument("--arrow-files", nargs="+", default=None,
                        help="Explicit list of arrow file paths (overrides --arrow-dir)")

    add_shar_output_args(parser)
    add_audio_processing_args(parser, target_sr_default=None)
    add_columnar_metadata_args(
        parser,
        id_column_default="id",
        text_column_default=None,
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=_DEFAULT_READ_BATCH_SIZE,
        help=(
            "Rows to materialize at once from each Arrow shard. "
            "Lower values reduce worker RSS; default: 256."
        ),
    )
    add_external_metadata_args(parser, include_custom_fields=True)
    add_input_clip_id_parser_arg(parser)
    add_parallelism_args(parser)
    args = parser.parse_args(argv)

    # Resolve arrow files
    if args.arrow_files:
        resolved = sorted(args.arrow_files)
    elif args.arrow_dir:
        resolved = sorted(str(p) for p in args.arrow_dir.glob(args.arrow_glob))
    else:
        parser.error("Either --arrow-files or --arrow-dir is required")
    if not resolved:
        raise FileNotFoundError("No arrow files resolved")

    args.shar_dir.mkdir(parents=True, exist_ok=True)

    num_workers = ensure_worker_assignment(
        args.shar_dir, resolved, args.num_workers, _ITEMS_KEY, "arrow files",
    )

    logger.info(f"Found {len(resolved)} arrow files, using {num_workers} workers")
    logger.info(f"Output: {args.shar_dir}")

    # Distribute arrow files across workers (round-robin)
    worker_arrows = distribute_round_robin(resolved, num_workers)

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
        mp_start_method = "forkserver"

    worker_args = [
        (
            wid,
            arrows,
            str(args.shar_dir),
            args.target_sr,
            args.shard_size,
            args.shar_format,
            args.id_column,
            args.audio_column,
            args.text_column,
            args.language_column,
            args.language,
            args.custom_columns,
            args.text_tokenize_custom_columns,
            text_tokenizer,
            args.resampling_backend,
            args.input_clip_id_parser,
            args.read_batch_size,
        )
        for wid, arrows in enumerate(worker_arrows)
        if arrows
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
