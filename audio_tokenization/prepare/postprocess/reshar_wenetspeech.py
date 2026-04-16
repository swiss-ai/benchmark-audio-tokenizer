#!/usr/bin/env python3
"""Reshar WenetSpeech: upsample to 24kHz + pre-tokenize text.

Reads existing Lhotse Shar (16kHz wav), resamples to 24kHz, encodes as FLAC,
tokenizes supervision text, and writes a new Shar directory.

Usage:
    python -m audio_tokenization.prepare.postprocess.reshar_wenetspeech \
        --input-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/wenetspeech \
        --output-dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/wenetspeech_24k \
        --text-tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --target-sr 24000 \
        --shard-size 2000 \
        --num-workers 20
"""

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

from audio_tokenization.prepare.common import (
    build_shar_index,
    check_worker_reuse,
    distribute_round_robin,
    init_worker_process,
    load_text_tokenizer,
    make_text_tokenize_fn,
    mark_partition_success,
    write_worker_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _reshar_worker(args_tuple):
    (
        worker_id,
        shard_indices,
        input_dir,
        output_dir,
        target_sr,
        shard_size,
        shar_format,
        text_tokenizer,
        resampling_backend,
    ) = args_tuple

    reused = check_worker_reuse(worker_id, output_dir)
    if reused is not None:
        return reused
    init_worker_process(resampling_backend)

    from lhotse import CutSet
    from lhotse.shar import SharWriter

    worker_dir = Path(output_dir) / f"worker_{worker_id:02d}"
    input_path = Path(input_dir)
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
        for shard_idx in shard_indices:
            cuts_path = str(input_path / f"cuts.{shard_idx:06d}.jsonl.gz")
            rec_path = str(input_path / f"recording.{shard_idx:06d}.tar")
            try:
                cuts = CutSet.from_shar(
                    fields={"cuts": [cuts_path], "recording": [rec_path]},
                    split_for_dataloading=False,
                    shuffle_shards=False,
                )
            except Exception as e:
                logger.error(f"Worker {worker_id}: failed to load shard {shard_idx}: {e}")
                errors += 1
                continue

            for cut in cuts:
                try:
                    if target_sr and cut.sampling_rate != target_sr:
                        cut = cut.resample(target_sr)
                        runtime_counts["resampled"] += 1

                    if _tokenize_text is not None:
                        cut = _tokenize_text(cut)
                        runtime_counts["text_tokenized"] += 1

                    writer.write(cut)
                    written += 1
                    total_duration_sec += cut.duration
                except Exception as e:
                    errors += 1
                    runtime_counts["processing_errors"] += 1
                    if errors <= 10:
                        logger.warning(f"Worker {worker_id} error on {cut.id}: {e}")

            runtime_counts["shards_processed"] += 1
            if runtime_counts["shards_processed"] % 10 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"Worker {worker_id}: {runtime_counts['shards_processed']} shards, "
                    f"{written} written, {errors} errors ({elapsed:.0f}s)"
                )

    return write_worker_result(
        worker_id=worker_id, worker_dir=worker_dir,
        written=written, skipped=skipped, errors=errors,
        total_duration_sec=total_duration_sec,
        runtime_counts=runtime_counts, t0=t0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reshar WenetSpeech: upsample + text tokenize")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Input shar directory (with cuts.*.jsonl.gz + recording.*.tar)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output shar directory")
    parser.add_argument("--text-tokenizer", type=str, required=True,
                        help="Path to tokenizer.json for pre-tokenizing text")
    parser.add_argument("--target-sr", type=int, default=24000,
                        help="Target sample rate (default: 24000)")
    parser.add_argument("--resampling-backend", type=str, default=None,
                        choices=["default", "sox"],
                        help="Lhotse resampling backend override (default: use "
                             "$LHOTSE_RESAMPLING_BACKEND or 'default')")
    parser.add_argument("--shard-size", type=int, default=2000,
                        help="Samples per output shard (default: 2000)")
    parser.add_argument("--shar-format", type=str, default="flac",
                        choices=["flac", "wav", "mp3", "opus"],
                        help="Audio format in output shar (default: flac)")
    parser.add_argument("--num-workers", type=int, default=20,
                        help="Number of parallel workers (default: 20)")
    args = parser.parse_args(argv)

    # Discover input shards
    input_cuts = sorted(args.input_dir.glob("cuts.*.jsonl.gz"))
    input_recs = sorted(args.input_dir.glob("recording.*.tar"))
    if len(input_cuts) != len(input_recs):
        raise ValueError(f"Mismatch: {len(input_cuts)} cuts vs {len(input_recs)} recording tars")
    if not input_cuts:
        raise FileNotFoundError(f"No shards found in {args.input_dir}")

    num_shards = len(input_cuts)
    shard_indices = list(range(num_shards))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = min(args.num_workers, num_shards)
    logger.info(f"Resharding {num_shards} shards with {num_workers} workers")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Target SR: {args.target_sr}, format: {args.shar_format}")

    # Load text tokenizer before forking (shared via COW)
    text_tokenizer = load_text_tokenizer(args.text_tokenizer)

    worker_shards = distribute_round_robin(shard_indices, num_workers)
    worker_args = [
        (
            wid,
            shards,
            str(args.input_dir),
            str(args.output_dir),
            args.target_sr,
            args.shard_size,
            args.shar_format,
            text_tokenizer,
            args.resampling_backend,
        )
        for wid, shards in enumerate(worker_shards)
        if shards
    ]

    import multiprocessing as mp
    t0 = time.time()
    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.map(_reshar_worker, worker_args)

    elapsed = time.time() - t0
    total_written = sum(r["written"] for r in results if r["written"] >= 0)
    total_errors = sum(r["errors"] for r in results)
    total_duration_sec = sum(r.get("total_duration_sec", 0.0) for r in results)
    logger.info(
        f"Done in {elapsed:.1f}s: {total_written:,} recordings, "
        f"{total_duration_sec / 3600:.1f} hours, {total_errors} errors"
    )

    # Build shar_index.json
    build_shar_index(args.output_dir, num_workers=num_workers)
    mark_partition_success(args.output_dir)
    logger.info("All done!")


if __name__ == "__main__":
    main()
