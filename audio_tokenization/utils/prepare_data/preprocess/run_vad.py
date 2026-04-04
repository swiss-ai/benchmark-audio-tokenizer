#!/usr/bin/env python3
"""Run Silero VAD on a directory of audio files.

Usage:
    python -m audio_tokenization.utils.prepare_data.preprocess.run_vad \
        --audio_dir /capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/raw_audios/en \
        --output_dir /capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/vad_results \
        --dataset voxpopuli --num_workers 288

Output JSONL format:
    {"file_stem": {"timestamps": [[start, end], ...], "duration_sec": float, "sample_rate": 16000}}
    Timestamps are in sample indices at 16 kHz.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import orjson

# Heavy libs (torch, soundfile, silero_vad) are imported inside worker
# functions — with spawn, each worker is a fresh process, so module-level
# imports would only waste memory in the main process.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_audio(path: Path):
    """Load audio -> mono float32 tensor at 16 kHz + original duration."""
    import soundfile as sf
    import torch
    import torchaudio

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration_sec = len(data) / sr

    if sr != VAD_SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            torch.from_numpy(data).unsqueeze(0), sr, VAD_SAMPLE_RATE,
        ).squeeze(0)
    else:
        wav = torch.from_numpy(data)

    return wav, duration_sec


def _read_done_keys(path: Path) -> set:
    """Read already-processed keys from a JSONL file for resume."""
    done = set()
    if path.is_file():
        with open(path, "rb") as f:
            for line in f:
                try:
                    done.add(next(iter(orjson.loads(line))))
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# VAD worker
# ---------------------------------------------------------------------------

_worker_vad = None


def _init_vad_worker(onnx: bool):
    global _worker_vad
    logging.disable(logging.INFO)  # workers only log warnings/errors
    from silero_vad import load_silero_vad
    _worker_vad = load_silero_vad(onnx=onnx)


def _process_file(audio_path: str) -> Optional[bytes]:
    """Run VAD on one audio file -> JSONL line."""
    from silero_vad import get_speech_timestamps

    path = Path(audio_path)
    try:
        wav, duration_sec = _load_audio(path)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return None

    try:
        timestamps = get_speech_timestamps(wav, _worker_vad, sampling_rate=VAD_SAMPLE_RATE)
    except Exception as e:
        logger.warning(f"VAD failed on {path}: {e}")
        return None

    ts_pairs = [[ts["start"], ts["end"]] for ts in timestamps]
    return orjson.dumps({
        path.stem: {
            "timestamps": ts_pairs,
            "duration_sec": duration_sec,
            "sample_rate": VAD_SAMPLE_RATE,
        }
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Silero VAD on audio files")
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pattern", default="**/*.ogg")
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--backend", choices=["onnx", "jit"], default="onnx")
    parser.add_argument("--pool_start", choices=["fork", "spawn"], default="spawn",
                        help="Worker process creation: spawn starts fresh interpreters "
                             "(slower startup, but avoids COW page faults at scale). "
                             "fork is faster to start but may be slower overall with "
                             "many workers due to copy-on-write overhead")
    args = parser.parse_args()

    if not args.audio_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {args.audio_dir}")

    lang = args.audio_dir.name
    output = args.output_dir / f"{args.dataset}_{lang}.jsonl"
    use_onnx = args.backend == "onnx"

    # Sort largest files first so long stragglers start early (LPT scheduling)
    audio_files = sorted(
        (p for p in args.audio_dir.glob(args.pattern) if p.is_file()),
        key=lambda p: p.stat().st_size, reverse=True,
    )
    audio_files = [str(p) for p in audio_files]
    logger.info(f"Found {len(audio_files):,} audio files in {args.audio_dir}")
    if not audio_files:
        return

    done_keys = _read_done_keys(output)
    if done_keys:
        before = len(audio_files)
        audio_files = [p for p in audio_files if Path(p).stem not in done_keys]
        logger.info(f"Resuming: {before - len(audio_files):,} done, {len(audio_files):,} remaining")
    if not audio_files:
        logger.info("All files already processed")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    backend_name = "ONNX" if use_onnx else "JIT"
    logger.info(f"Running Silero VAD ({backend_name}) with {args.num_workers} workers")

    from tqdm import tqdm

    processed = failed = 0
    ctx = mp.get_context(args.pool_start)
    with ctx.Pool(args.num_workers, initializer=_init_vad_worker, initargs=(use_onnx,)) as pool:
        with open(output, "ab") as out:
            pbar = tqdm(
                pool.imap_unordered(_process_file, audio_files, chunksize=4),
                total=len(audio_files), desc=f"VAD ({backend_name})", unit="file",
            )
            for result in pbar:
                if result is not None:
                    out.write(result + b"\n")
                    out.flush()
                    processed += 1
                else:
                    failed += 1
                pbar.set_postfix(ok=processed, fail=failed)

    logger.info(f"Done: {processed:,} processed, {failed:,} failed -> {output}")


if __name__ == "__main__":
    main()
