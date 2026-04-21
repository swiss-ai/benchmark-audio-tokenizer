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

from audio_tokenization.config.schema import PrepareSpec
from audio_tokenization.prepare.audio_ops import (
    apply_audio_pipeline,
    build_recording_from_audio_bytes,
)
from audio_tokenization.prepare.cli import (
    add_audio_processing_args,
    add_external_metadata_args,
    add_parallelism_args,
    add_shar_output_args,
)
from audio_tokenization.prepare.columnar import (
    _get_field,
    add_columnar_metadata_args,
    extract_row_metadata,
    _projected_columns,
    validate_columnar_schema_roots,
)
from audio_tokenization.prepare.constants import _MISSING
from audio_tokenization.prepare.identity import (
    add_input_clip_id_parser_arg,
    resolve_input_source_and_clip_num,
    set_universal_cut_id,
)
from audio_tokenization.prepare.metadata import (
    load_external_metadata,
    resolve_sample_text_and_custom,
)
from audio_tokenization.prepare.runtime import (
    check_worker_reuse,
    distribute_round_robin,
    ensure_worker_assignment,
    init_worker_process,
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


def _preflight_prepare(spec: PrepareSpec, resolved: list[str]) -> None:
    import pyarrow.parquet as pq

    m = spec.metadata
    o = spec.output
    sample_path = resolved[0]
    validate_columnar_schema_roots(
        available_roots=pq.ParquetFile(sample_path).schema_arrow.names,
        required_columns=(m.audio_column, m.id_column),
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

    validate_prepare_runtime(
        resampling_backend=o.resampling_backend,
        require_ffmpeg=True,
        text_tokenizer_path=o.text_tokenizer,
    )


def build_parser() -> argparse.ArgumentParser:
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

    return parser


def run(spec: PrepareSpec):
    """Execute parquet prepare for a typed PrepareSpec.

    The legacy ``main(argv)`` path also feeds this; it parses argv, then
    routes the values through ``PrepareSpec`` so both Hydra and CLI runs
    converge on one execution path.
    """
    i, o, m = spec.input, spec.output, spec.metadata
    parquet_dir = Path(i.parquet_dir)
    shar_dir = Path(o.shar_dir)

    resolved = sorted(str(p) for p in parquet_dir.glob(i.parquet_glob))
    if not resolved:
        raise FileNotFoundError(f"No files match {parquet_dir / i.parquet_glob}")

    _preflight_prepare(spec, resolved)

    shar_dir.mkdir(parents=True, exist_ok=True)
    write_prepare_state_for_spec(spec)

    num_workers = ensure_worker_assignment(
        shar_dir, resolved, o.num_workers, _ITEMS_KEY, "parquet files",
    )

    logger.info(f"Found {len(resolved)} parquet files, using {num_workers} workers")
    logger.info(f"Output: {shar_dir}")

    worker_parquets = distribute_round_robin(resolved, num_workers)

    # Load text tokenizer before forking (shared via COW across workers).
    text_tokenizer = load_text_tokenizer(o.text_tokenizer)

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
        (
            wid,
            parquets,
            str(shar_dir),
            o.target_sr,
            o.shard_size,
            o.shar_format,
            m.id_column,
            m.audio_column,
            m.text_column,
            m.duration_column,
            m.language_column,
            m.language,
            m.custom_columns or None,
            m.text_tokenize_custom_columns or None,
            text_tokenizer,
            o.resampling_backend,
            m.input_clip_id_parser,
            o.read_batch_size,
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


def _args_to_spec(args) -> PrepareSpec:
    """Translate flat argparse Namespace → typed PrepareSpec.

    The legacy CLI is now a thin frontend: argv → Namespace (via
    build_parser) → PrepareSpec → run(spec). This keeps the per-family
    flat↔nested mapping co-located with the CLI it serves, and lets the
    schema do all validation regardless of caller.
    """
    return PrepareSpec.from_mapping({
        "family": "parquet",
        "input": {
            "parquet_dir": str(args.parquet_dir),
            "parquet_glob": args.parquet_glob,
        },
        "output": {
            "shar_dir": str(args.shar_dir),
            "shard_size": args.shard_size,
            "shar_format": args.shar_format,
            "target_sr": args.target_sr,
            "text_tokenizer": args.text_tokenizer,
            "num_workers": args.num_workers,
            "resampling_backend": args.resampling_backend,
            "mp_start_method": args.mp_start_method,
            "read_batch_size": args.read_batch_size,
        },
        "metadata": {
            "audio_column": args.audio_column,
            "text_column": args.text_column,
            "duration_column": args.duration_column,
            "id_column": args.id_column,
            "language_column": args.language_column,
            "language": args.language,
            "custom_columns": args.custom_columns or [],
            "text_tokenize_custom_columns": args.text_tokenize_custom_columns or [],
            "external_metadata": args.external_metadata,
            "custom_fields": args.custom_fields or [],
            "id_field": args.id_field,
            "text_field": args.text_field,
            "input_clip_id_parser": args.input_clip_id_parser,
        },
    })


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(_args_to_spec(args))


if __name__ == "__main__":
    main()
