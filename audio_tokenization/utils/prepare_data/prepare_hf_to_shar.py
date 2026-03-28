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
    check_worker_reuse,
    distribute_round_robin,
    ensure_worker_assignment,
    init_worker_process,
    load_text_tokenizer,
    make_text_tokenize_fn,
    rms_db,
    run_pool_and_finalize,
    to_mono,
    write_worker_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_ITEMS_KEY = "resolved_arrows"


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
        text_tokenizer,
        resampling_backend,
    ) = args_tuple

    reused = check_worker_reuse(worker_id, shar_dir)
    if reused is not None:
        return reused
    init_worker_process(resampling_backend)

    import pyarrow.ipc as ipc
    from lhotse import Recording, SupervisionSegment
    from lhotse.shar import SharWriter

    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    t0 = time.time()
    written = skipped = errors = 0
    total_duration_sec = 0.0
    runtime_counts = Counter()
    _tokenize_text = make_text_tokenize_fn(text_tokenizer) if text_tokenizer is not None else None

    with SharWriter(
        output_dir=str(worker_dir),
        fields={"recording": shar_format},
        shard_size=shard_size,
    ) as writer:
        for arrow_path in arrow_paths:
            arrow_name = Path(arrow_path).name
            logger.info(f"Worker {worker_id}: reading {arrow_name}")
            reader = ipc.open_stream(arrow_path)
            table = reader.read_all()

            id_col = table.column(id_column)
            audio_col = table.column(audio_column)
            text_col = table.column(text_column) if text_column else None

            for i in range(table.num_rows):
                row_id = id_col[i].as_py()
                try:
                    audio_struct = audio_col[i].as_py()
                    audio_bytes = audio_struct.get("bytes") if isinstance(audio_struct, dict) else None
                    if not audio_bytes:
                        skipped += 1
                        runtime_counts["skipped_empty_audio"] += 1
                        continue

                    recording = Recording.from_bytes(
                        data=audio_bytes, recording_id=str(row_id),
                    )
                    cut = recording.to_cut()

                    # Attach supervision with text
                    text = text_col[i].as_py() if text_col else None
                    if text:
                        cut.supervisions = [SupervisionSegment(
                            id=cut.id,
                            recording_id=cut.recording_id,
                            start=0.0,
                            duration=cut.duration,
                            text=text,
                        )]

                    # Resample if needed
                    if target_sr and cut.sampling_rate != target_sr:
                        cut = cut.resample(target_sr)
                        runtime_counts["resampled"] += 1

                    # Mono
                    cut = to_mono(cut, mono_downmix=True, stats=runtime_counts)

                    # Pre-tokenize text
                    if _tokenize_text is not None:
                        cut = _tokenize_text(cut)

                    # Precompute RMS (audio already decoded; avoids double load at tokenize time)
                    cut.custom = cut.custom or {}
                    cut.custom["rms_db"] = rms_db(cut)

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

    # Shar output
    parser.add_argument("--shar-dir", type=Path, required=True,
                        help="Output directory for Shar format")
    parser.add_argument("--shard-size", type=int, default=2000,
                        help="Samples per Shar shard (default: 2000)")
    parser.add_argument("--shar-format", type=str, default="flac",
                        choices=["flac", "wav", "mp3", "opus"],
                        help="Audio format in Shar (default: flac)")

    # Audio processing
    parser.add_argument("--target-sr", type=int, default=None,
                        help="Target sample rate (default: None, keep original)")
    parser.add_argument("--resampling-backend", type=str, default=None,
                        choices=["default", "sox"],
                        help="Lhotse resampling backend override (default: use "
                             "$LHOTSE_RESAMPLING_BACKEND or 'default')")

    # Column names
    parser.add_argument("--id-column", type=str, default="id",
                        help="Column name for row ID (default: 'id')")
    parser.add_argument("--audio-column", type=str, default="audio",
                        help="Column name for audio struct (default: 'audio')")
    parser.add_argument("--text-column", type=str, default=None,
                        help="Column name for transcription text (default: None)")

    # Text tokenizer
    parser.add_argument("--text-tokenizer", type=str, default=None,
                        help="Path to tokenizer.json for pre-tokenizing supervision text")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=20,
                        help="Number of parallel workers (default: 20)")
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
            text_tokenizer,
            args.resampling_backend,
        )
        for wid, arrows in enumerate(worker_arrows)
        if arrows
    ]

    run_pool_and_finalize(_convert_worker, worker_args, args.shar_dir, num_workers)


if __name__ == "__main__":
    main()
