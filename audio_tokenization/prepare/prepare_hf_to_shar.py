#!/usr/bin/env python3
"""Convert HF-style arrow files (with audio bytes) to Lhotse Shar format.

Designed for datasets stored as HuggingFace arrow shards (e.g. People's Speech
with 786 arrow files). Each row has an audio struct with raw bytes and optional
text transcription.

Workers are assigned *whole arrow files* (not rows) via round-robin. With 786
files / 128 workers, each worker handles ~6 files, giving good balance.

Usage (People's Speech):
    python -m audio_tokenization.prepare.prepare_hf_to_shar \
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
    add_columnar_metadata_args,
    extract_row_metadata,
    validate_columnar_schema_roots,
)
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


def _preflight_prepare(spec, resolved: list[str]) -> None:
    import pyarrow.ipc as ipc

    m, o = spec.metadata, spec.output
    sample_path = resolved[0]
    with open(sample_path, "rb") as f:
        schema_names = ipc.open_stream(f).schema.names

    validate_columnar_schema_roots(
        available_roots=schema_names,
        required_columns=(m.audio_column, m.id_column),
        optional_columns=(
            m.text_column,
            m.language_column,
            m.custom_columns,
        ),
        source_path=sample_path,
        source_kind="Arrow",
        logger=logger,
    )

    validate_prepare_runtime(
        resampling_backend=o.resampling_backend,
        require_ffmpeg=True,
        text_tokenizer_path=o.text_tokenizer,
    )


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def run(spec):
    """Execute HF arrow prepare for a typed PrepareSpec."""
    i, o, m = spec.input, spec.output, spec.metadata
    shar_dir = Path(o.shar_dir)

    if i.arrow_files:
        resolved = sorted(i.arrow_files)
    elif i.arrow_dir:
        resolved = sorted(str(p) for p in Path(i.arrow_dir).glob(i.arrow_glob))
    else:
        raise ValueError("prepare.input requires arrow_dir or arrow_files")
    if not resolved:
        raise FileNotFoundError("No arrow files resolved")

    _preflight_prepare(spec, resolved)

    shar_dir.mkdir(parents=True, exist_ok=True)
    write_prepare_state_for_spec(spec)

    num_workers = ensure_worker_assignment(
        shar_dir, resolved, o.num_workers, _ITEMS_KEY, "arrow files",
    )

    logger.info(f"Found {len(resolved)} arrow files, using {num_workers} workers")
    logger.info(f"Output: {shar_dir}")

    worker_arrows = distribute_round_robin(resolved, num_workers)
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
        mp_start_method = "forkserver"

    worker_args = [
        (
            wid,
            arrows,
            str(shar_dir),
            o.target_sr,
            o.shard_size,
            o.shar_format,
            m.id_column,
            m.audio_column,
            m.text_column,
            m.language_column,
            m.language,
            m.custom_columns or None,
            m.text_tokenize_custom_columns or None,
            text_tokenizer,
            o.resampling_backend,
            m.input_clip_id_parser,
            o.read_batch_size,
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


def _args_to_spec(args):
    from audio_tokenization.config.schema import PrepareSpec

    return PrepareSpec.from_mapping({
        "family": "hf",
        "input": {
            "arrow_dir": str(args.arrow_dir) if args.arrow_dir else None,
            "arrow_glob": args.arrow_glob,
            "arrow_files": args.arrow_files,
        },
        "output": {
            "shar_dir": str(args.shar_dir),
            "shard_size": args.shard_size,
            "shar_format": args.shar_format,
            "target_sr": args.target_sr,
            "text_tokenizer": args.text_tokenizer,
            "num_workers": args.num_workers,
            "resampling_backend": args.resampling_backend,
            "read_batch_size": args.read_batch_size,
        },
        "metadata": {
            "audio_column": args.audio_column,
            "text_column": args.text_column,
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
    return run(_args_to_spec(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
