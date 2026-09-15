"""Compare WavTokenizer-40 output with and without peak-normalization.

Tokenizes ~6000 stage-1 SHAR samples (voxpopuli, mtg_jamendo, audioset) with:
  1. raw  — no amplitude change
  2. n-3  — peak-normalized to -3 dBFS (pipeline default)

For each sample: peak/RMS, token count, per-position agreement. Aggregates
grouped by source and by |gain| applied. Also reports how much of the raw
stage-1 peak distribution lies outside WavTokenizer's training range.

All audio is resampled to 24 kHz (WavTokenizer native rate) *before*
normalization so resampling does not confound the comparison.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path("/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer")
LHOTSE_DEV = Path("/iopsstor/scratch/cscs/xyixuan/dev/lhotse")
for p in (str(LHOTSE_DEV), str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torchaudio
from lhotse import CutSet


TARGET_SR = 24000
MAX_SECONDS = 15.0
TARGET_DB = -3.0
_EPSILON = 1e-10
_WAVTOKENIZER_TRAINING_RANGE = (-6.0, -1.0)  # dBFS, see decoder/dataset.py L69


class Row(NamedTuple):
    source: str
    sid: str
    peak_db: float
    rms_db: float
    n_tok: int
    match_pct: float
    gain_db: float


def load_lhotse_samples(
    shard_pairs: list[tuple[str, str]], n: int,
) -> list[tuple[str, np.ndarray]]:
    """Walk (cuts, recording) shard pairs in order until `n` samples collected."""
    out: list[tuple[str, np.ndarray]] = []
    for cuts_path, rec_path in shard_pairs:
        if len(out) >= n:
            break
        try:
            cs = CutSet.from_shar(fields={"cuts": [cuts_path], "recording": [rec_path]})
        except Exception as e:
            print(f"  skip shard {cuts_path}: {e}")
            continue
        for c in cs:
            if len(out) >= n:
                break
            try:
                c_trunc = c.truncate(duration=min(MAX_SECONDS, c.duration))
                arr = c_trunc.load_audio()
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                if arr.shape[0] == 0:
                    continue
                if c.sampling_rate != TARGET_SR:
                    arr = torchaudio.transforms.Resample(c.sampling_rate, TARGET_SR)(
                        torch.from_numpy(arr).unsqueeze(0)).squeeze(0).numpy()
                out.append((c.id, arr.astype(np.float32)))
            except Exception as e:
                print(f"  skip {c.id}: {e}")
    return out


def peak_normalize(audio: np.ndarray, target_db: float) -> np.ndarray:
    target_peak = 10 ** (target_db / 20.0)
    peak = float(np.max(np.abs(audio)))
    if peak < _EPSILON:
        return audio.copy()
    return (audio * (target_peak / peak)).astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20 * np.log10(rms + _EPSILON)


def peak_db(audio: np.ndarray) -> float:
    return 20 * np.log10(float(np.max(np.abs(audio))) + _EPSILON)


def tokenize(tokenizer, audio: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(audio).float().unsqueeze(0)
    codes, _ = tokenizer.encode(x, sr=TARGET_SR)
    return codes.flatten().cpu()


def collect_samples() -> list[tuple[str, str, np.ndarray]]:
    """Return list of (source_tag, sample_id, audio_24k_mono)."""
    samples: list[tuple[str, str, np.ndarray]] = []

    def _shard_pairs(base: str, start: int, n_shards: int) -> list[tuple[str, str]]:
        return [(f"{base}/cuts.{i:06d}.jsonl.gz", f"{base}/recording.{i:06d}.tar")
                for i in range(start, start + n_shards)]

    vp_base = "/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/voxpopuli_shar/node_00/worker_00"
    for sid, a in load_lhotse_samples(_shard_pairs(vp_base, 0, 2), 2000):
        samples.append(("stage1/voxpopuli", sid, a))

    mj_base = "/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/mtg_jamendo_train/worker_00"
    for sid, a in load_lhotse_samples(_shard_pairs(mj_base, 0, 2), 2000):
        samples.append(("stage1/mtg_jamendo", sid, a))

    # audioset_bal part-* shards only have ~584 cuts each → walk multiple parts
    as_pairs: list[tuple[str, str]] = []
    for part in range(8):
        as_pairs += _shard_pairs(
            f"/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/audioset_bal_train_audio_only/part-{part:05d}",
            0, 1)
    for sid, a in load_lhotse_samples(as_pairs, 2000):
        samples.append(("stage1/audioset", sid, a))

    return samples


def main():
    from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40

    print("Loading WavTokenizer-40 (24 kHz, 40 tok/s)...")
    tokenizer = WavTokenizer40(device="cuda", torch_compile=False)
    print("Collecting samples...")
    samples = collect_samples()
    print(f"Got {len(samples)} samples.\n")

    rows: list[Row] = []
    for source, sid, audio in samples:
        pk0 = peak_db(audio)
        rm0 = rms_db(audio)
        a_n = peak_normalize(audio, TARGET_DB)
        gain = TARGET_DB - pk0
        c_raw = tokenize(tokenizer, audio)
        c_n = tokenize(tokenizer, a_n)
        n = min(c_raw.numel(), c_n.numel())
        match_pct = 100.0 * float((c_raw[:n] == c_n[:n]).sum()) / max(n, 1)
        rows.append(Row(source, sid, pk0, rm0, c_raw.numel(), match_pct, gain))

    csv_path = "/tmp/tokenize_norm_compare_rows.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(Row._fields))
        for r in rows:
            w.writerow(r)
    print(f"Saved per-sample rows to {csv_path}")

    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(r.source, []).append(r)

    print("\n=== aggregate by source (raw vs peak-norm -3 dBFS) ===")
    print(f"{'source':22s} {'n':>4s} {'pk0 med':>9s} {'gain med':>10s} "
          f"{'match% mean':>13s} {'match% med':>12s} {'match% p10':>12s} {'match% p90':>12s}")
    for src, rs in groups.items():
        pks = np.array([r.peak_db for r in rs])
        gains = np.array([r.gain_db for r in rs])
        m = np.array([r.match_pct for r in rs])
        print(f"{src:22s} {len(rs):4d} {np.median(pks):9.2f} {np.median(gains):10.2f} "
              f"{m.mean():13.2f} {np.median(m):12.2f} {np.percentile(m,10):12.2f} "
              f"{np.percentile(m,90):12.2f}")

    print("\n=== aggregate by |gain| applied (target -3 dBFS) ===")
    print(f"{'bin':20s} {'n':>4s} {'match% mean':>13s} {'match% med':>12s}")
    for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 10), (10, 20), (20, 100)]:
        subset = [r for r in rows if lo <= abs(r.gain_db) < hi]
        if not subset:
            continue
        m = np.array([r.match_pct for r in subset])
        label = f"{lo:>3.0f}..{hi:<3.0f} dB"
        print(f"{label:20s} {len(subset):4d} {m.mean():13.2f} {np.median(m):12.2f}")

    all_m = np.array([r.match_pct for r in rows])
    print(f"\nOverall (n={len(rows)}): match% mean={all_m.mean():.2f} "
          f"median={np.median(all_m):.2f} p10={np.percentile(all_m,10):.2f} "
          f"p90={np.percentile(all_m,90):.2f}")

    lo, hi = _WAVTOKENIZER_TRAINING_RANGE
    print(f"\n=== raw peak distribution vs WavTokenizer training range [{lo}, {hi}] dBFS ===")
    print(f"{'source':22s} {'n':>5s} {'in range':>11s} {'below '+str(lo):>10s} {'above '+str(hi):>10s}")
    for src, rs in groups.items():
        pks = np.array([r.peak_db for r in rs])
        in_range = int(((pks >= lo) & (pks <= hi)).sum())
        below = int((pks < lo).sum())
        above = int((pks > hi).sum())
        n = len(rs)
        print(f"{src:22s} {n:>5d} {in_range:>5d} ({100*in_range/n:4.1f}%) "
              f"{below:>4d} ({100*below/n:4.1f}%) {above:>4d} ({100*above/n:4.1f}%)")
    all_pk = np.array([r.peak_db for r in rows])
    in_range = int(((all_pk >= lo) & (all_pk <= hi)).sum())
    below = int((all_pk < lo).sum())
    above = int((all_pk > hi).sum())
    n = len(rows)
    print(f"{'TOTAL':22s} {n:>5d} {in_range:>5d} ({100*in_range/n:4.1f}%) "
          f"{below:>4d} ({100*below/n:4.1f}%) {above:>4d} ({100*above/n:4.1f}%)")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
