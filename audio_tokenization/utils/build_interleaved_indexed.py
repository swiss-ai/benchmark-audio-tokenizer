#!/usr/bin/env python3
"""Convert interleaved-tokenization parquet files to Megatron indexed datasets (bin/idx).

Groups consecutive clips from the same source and applies cross-clip interleaving
patterns to produce one bin/idx pair per pattern.  Remainder clips that don't fill
a complete window are handled by a cascade of progressively shorter sub-patterns
(auto-derived by truncating the main patterns), with single-clip remainders emitted
as same-clip transcription sequences into ``transcribe.bin/.idx``.

Pattern string rules:
  - Each character is 'A' (audio) or 'T' (text)
  - len(pattern) = window size (clips per sequence)
  - A→T transition: insert <|speech_switch|> between audio_end and text tokens
  - T→A transition: no extra token (audio_tokens start with <|audio_start|>)
  - A→A, T→T: no extra tokens

Remainder cascade (example with --patterns ATAT TATA, window=4):
  - 3 clips left → ATA.bin + TAT.bin  (first 3 chars of each main pattern)
  - 2 clips left → AT.bin + TA.bin    (first 2 chars)
  - 1 clip left  → transcribe.bin     [BOS, audio, speech_transcribe, text, EOS]
  - 0 clips left → nothing

Usage
-----
    # Dry run — print statistics without writing any files
    python -m audio_tokenization.utils.build_interleaved_indexed \
        --parquet-dir /path/to/emilia_yodas_interleaved_dur2-200 \
        --output-dir /path/to/emilia_yodas_indexed \
        --tokenizer-path /path/to/apertus_emu3.5_wavtok \
        --patterns ATAT TATA \
        --dry-run

    # Full run — build bin/idx files
    python -m audio_tokenization.utils.build_interleaved_indexed \
        --parquet-dir /path/to/emilia_yodas_interleaved_dur2-200 \
        --output-dir /path/to/emilia_yodas_indexed \
        --tokenizer-path /path/to/apertus_emu3.5_wavtok \
        --patterns ATAT TATA
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from transformers import AutoTokenizer

from audio_tokenization.utils.indexed_dataset.indexed_dataset_megatron import (
    _INDEX_HEADER,
    DType,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


# ---------------------------------------------------------------------------
# Module-level globals for fork-based sharing (set before Pool creation)
# ---------------------------------------------------------------------------

_shared_audio_arrow = None
_shared_text_arrow = None
_shared_run_starts = None
_shared_run_lengths = None

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

TR_KEY = "transcribe"


def build_sequence(
    audio_tokens: list[list[int]],
    text_tokens: list[list[int]],
    pattern: str,
    bos_id: int,
    eos_id: int,
    switch_id: int,
) -> list[int]:
    """Build a token sequence from parallel audio/text token lists.

    ``audio_tokens[i]`` / ``text_tokens[i]`` are the tokens for clip *i* in
    the window.  Each pattern character selects which list to draw from:
      A → audio_tokens[i]  (already includes [audio_start, ..., audio_end])
      T → text_tokens[i]

    A ``<|speech_switch|>`` token is inserted at every A→T transition.
    """
    seq: list[int] = [bos_id]
    prev_mode: str | None = None
    for i, mode in enumerate(pattern):
        if mode == "A":
            seq.extend(audio_tokens[i])
        else:  # T
            if prev_mode == "A":
                seq.append(switch_id)
            seq.extend(text_tokens[i])
        prev_mode = mode
    seq.append(eos_id)
    return seq


def find_consecutive_runs(sorted_clip_nums: list[int]) -> list[list[int]]:
    """Split sorted clip numbers into runs of consecutive integers.

    >>> find_consecutive_runs([3, 4, 5, 8, 9, 15])
    [[3, 4, 5], [8, 9], [15]]
    """
    if not sorted_clip_nums:
        return []
    runs: list[list[int]] = []
    current: list[int] = [sorted_clip_nums[0]]
    for prev, cur in zip(sorted_clip_nums, sorted_clip_nums[1:]):
        if cur == prev + 1:
            current.append(cur)
        else:
            runs.append(current)
            current = [cur]
    runs.append(current)
    return runs


def group_patterns_by_size(patterns: list[str]) -> dict[int, list[str]]:
    """Group pattern strings by their window size (length).

    >>> group_patterns_by_size(["AT", "TA", "AAT", "TTA"])
    {2: ['AT', 'TA'], 3: ['AAT', 'TTA']}
    """
    by_size: dict[int, list[str]] = defaultdict(list)
    for p in patterns:
        by_size[len(p)].append(p)
    return dict(by_size)


def derive_sub_patterns(
    patterns: list[str], min_window: int,
) -> dict[int, list[str]]:
    """Derive cascade sub-patterns by truncating each main pattern.

    For remainder size *k* (2 ≤ k < min_window), the sub-patterns are the
    unique length-k prefixes of the main patterns, preserving order.

    >>> derive_sub_patterns(["ATAT", "TATA"], 4)
    {3: ['ATA', 'TAT'], 2: ['AT', 'TA']}
    """
    sub_pats: dict[int, list[str]] = {}
    for k in range(min_window - 1, 1, -1):
        # dict.fromkeys preserves insertion order and deduplicates
        subs = list(dict.fromkeys(pat[:k] for pat in patterns if len(pat) >= k))
        if subs:
            sub_pats[k] = subs
    return sub_pats


def load_token_ids(tokenizer_path: str) -> tuple[int, int, int, int, int]:
    """Load BOS, EOS, speech_switch, speech_transcribe IDs and vocab_size."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    switch_id = tokenizer.convert_tokens_to_ids("<|speech_switch|>")
    transcribe_id = tokenizer.convert_tokens_to_ids("<|speech_transcribe|>")

    assert bos_id is not None, "BOS token must be defined in tokenizer"
    assert eos_id is not None, "EOS token must be defined in tokenizer"
    assert switch_id != tokenizer.unk_token_id, (
        "<|speech_switch|> not found in tokenizer"
    )
    assert transcribe_id != tokenizer.unk_token_id, (
        "<|speech_transcribe|> not found in tokenizer"
    )

    vocab_size = len(tokenizer)
    return bos_id, eos_id, switch_id, transcribe_id, vocab_size


def _pattern_constants(pat: str) -> tuple[list[int], list[int], int]:
    """Return (audio_positions, text_positions, n_switches) for a pattern."""
    a_pos = [i for i, c in enumerate(pat) if c == "A"]
    t_pos = [i for i, c in enumerate(pat) if c == "T"]
    n_sw = sum(1 for i in range(1, len(pat)) if pat[i - 1] == "A" and pat[i] == "T")
    return a_pos, t_pos, n_sw


# ---------------------------------------------------------------------------
# Parallel helpers
# ---------------------------------------------------------------------------


def _partition_runs(
    run_lengths: np.ndarray, num_workers: int,
) -> list[tuple[int, int]]:
    """Partition runs into balanced chunks by cumulative clip count.

    Returns a list of (start_run, end_run) tuples — one per worker.
    """
    n_runs = len(run_lengths)
    total_clips = int(run_lengths.sum())
    clips_per_worker = total_clips // num_workers
    cum_clips = np.cumsum(run_lengths)

    boundaries = [0]
    for w in range(1, num_workers):
        target = w * clips_per_worker
        idx = int(np.searchsorted(cum_clips, target))
        boundaries.append(min(idx, n_runs))
    boundaries.append(n_runs)

    return [
        (boundaries[w], boundaries[w + 1])
        for w in range(num_workers)
        if boundaries[w] < boundaries[w + 1]
    ]


def _process_run_chunk(
    worker_id: int,
    run_start: int,
    run_end: int,
    all_keys: list[str],
    patterns_by_size: dict[int, list[str]],
    sub_patterns_by_size: dict[int, list[str]],
    min_window: int,
    bos_id: int,
    eos_id: int,
    switch_id: int,
    transcribe_id: int,
    dtype: type,
    tmp_dir: str,
) -> dict[str, dict]:
    """Process a contiguous range of runs; write shard .bin + sidecar .npy.

    Reads Arrow / run arrays from module-level globals (inherited via fork COW).
    Returns lightweight counters + shard path info — no large lists cross the
    pickle boundary.
    """
    audio_arrow = _shared_audio_arrow
    text_arrow = _shared_text_arrow
    run_starts = _shared_run_starts
    run_lengths_arr = _shared_run_lengths

    # Open per-pattern builders writing to shard files
    builders: dict[str, IndexedDatasetBuilder] = {}
    counters: dict[str, dict[str, int]] = {}
    for key in all_keys:
        shard_prefix = f"{tmp_dir}/{key}_shard{worker_id:04d}"
        builders[key] = IndexedDatasetBuilder(
            get_bin_path(shard_prefix), dtype=dtype,
        )
        counters[key] = {"seqs": 0, "tokens": 0}

    for r in range(run_start, run_end):
        rs = int(run_starts[r])
        rl = int(run_lengths_arr[r])

        run_audio: list[list[int]] = audio_arrow[rs : rs + rl].to_pylist()
        run_text: list[list[int]] = text_arrow[rs : rs + rl].to_pylist()

        # --- main interleaved windows ---
        for wsz, pats in patterns_by_size.items():
            if rl < wsz:
                continue
            for w_start in range(0, rl - wsz + 1, wsz):
                for pat in pats:
                    seq: list[int] = [bos_id]
                    prev: str | None = None
                    for k, mode in enumerate(pat):
                        if mode == "A":
                            seq.extend(run_audio[w_start + k])
                        else:
                            if prev == "A":
                                seq.append(switch_id)
                            seq.extend(run_text[w_start + k])
                        prev = mode
                    seq.append(eos_id)

                    builders[pat].add_item(seq)
                    builders[pat].end_document()
                    counters[pat]["seqs"] += 1
                    counters[pat]["tokens"] += len(seq)

        # --- cascade remainder ---
        n_rem = rl % min_window
        if n_rem >= 2:
            sub_pats = sub_patterns_by_size.get(n_rem, [])
            rem_off = rl - n_rem
            for sp in sub_pats:
                seq = [bos_id]
                prev = None
                for k, mode in enumerate(sp):
                    if mode == "A":
                        seq.extend(run_audio[rem_off + k])
                    else:
                        if prev == "A":
                            seq.append(switch_id)
                        seq.extend(run_text[rem_off + k])
                    prev = mode
                seq.append(eos_id)

                builders[sp].add_item(seq)
                builders[sp].end_document()
                counters[sp]["seqs"] += 1
                counters[sp]["tokens"] += len(seq)

        elif n_rem == 1:
            seq = [bos_id]
            seq.extend(run_audio[-1])
            seq.append(transcribe_id)
            seq.extend(run_text[-1])
            seq.append(eos_id)

            builders[TR_KEY].add_item(seq)
            builders[TR_KEY].end_document()
            counters[TR_KEY]["seqs"] += 1
            counters[TR_KEY]["tokens"] += len(seq)

    # Save sidecar .npy files, close .bin (do NOT call finalize)
    result: dict[str, dict] = {}
    for key in all_keys:
        b = builders[key]
        b.data_file.close()
        shard_prefix = f"{tmp_dir}/{key}_shard{worker_id:04d}"
        np.save(f"{shard_prefix}_seqlens.npy", np.array(b.sequence_lengths, dtype=np.int32))
        np.save(f"{shard_prefix}_docidx.npy", np.array(b.document_indices, dtype=np.int64))
        result[key] = {
            "seqs": counters[key]["seqs"],
            "tokens": counters[key]["tokens"],
            "shard_prefix": shard_prefix,
        }

    return result


def _write_idx_file(
    idx_path: str,
    dtype: type,
    sequence_lengths: np.ndarray,
    document_indices: np.ndarray,
) -> None:
    """Write a Megatron .idx file from numpy arrays."""
    itemsize = DType.size(dtype)
    pointers = np.zeros(len(sequence_lengths), dtype=np.int64)
    if len(sequence_lengths) > 0:
        np.cumsum(sequence_lengths.astype(np.int64) * itemsize, out=pointers)
        # shift right: pointers[i] = cumsum[i-1], pointers[0] = 0
        pointers = np.roll(pointers, 1)
        pointers[0] = 0

    with open(idx_path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", DType.code_from_dtype(dtype)))
        f.write(struct.pack("<Q", len(sequence_lengths)))
        f.write(struct.pack("<Q", len(document_indices)))
        f.write(sequence_lengths.astype(np.int32).tobytes(order="C"))
        f.write(pointers.tobytes(order="C"))
        f.write(document_indices.astype(np.int64).tobytes(order="C"))


def _merge_shards(
    worker_results: list[dict[str, dict]],
    all_keys: list[str],
    output_dir: Path,
    dtype: type,
    tmp_dir: Path,
) -> dict[str, dict[str, int]]:
    """Concatenate shard .bin + sidecar .npy files into final output."""
    counters: dict[str, dict[str, int]] = {k: {"seqs": 0, "tokens": 0} for k in all_keys}
    copy_buf = 64 * 1024 * 1024  # 64 MB

    for key in all_keys:
        # Collect shard info in worker order
        shard_prefixes: list[str] = []
        for wr in worker_results:
            info = wr[key]
            counters[key]["seqs"] += info["seqs"]
            counters[key]["tokens"] += info["tokens"]
            if info["seqs"] > 0:
                shard_prefixes.append(info["shard_prefix"])

        if not shard_prefixes:
            # No data for this key — write empty files
            open(get_bin_path(str(output_dir / key)), "wb").close()
            _write_idx_file(
                get_idx_path(str(output_dir / key)),
                dtype,
                np.array([], dtype=np.int32),
                np.array([0], dtype=np.int64),
            )
            continue

        # Concatenate .bin shards
        bin_path = get_bin_path(str(output_dir / key))
        with open(bin_path, "wb") as out_f:
            for sp in shard_prefixes:
                src = get_bin_path(sp)
                with open(src, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, copy_buf)

        # Load and merge sidecar .npy files
        all_seqlens: list[np.ndarray] = []
        all_docidx: list[np.ndarray] = []
        seq_offset = 0
        for i, sp in enumerate(shard_prefixes):
            sl = np.load(f"{sp}_seqlens.npy")
            di = np.load(f"{sp}_docidx.npy")
            all_seqlens.append(sl)
            if i == 0:
                all_docidx.append(di)
            else:
                # Drop leading 0, apply offset
                all_docidx.append(di[1:] + seq_offset)
            seq_offset += len(sl)

        merged_seqlens = np.concatenate(all_seqlens)
        merged_docidx = np.concatenate(all_docidx)

        _write_idx_file(
            get_idx_path(str(output_dir / key)),
            dtype,
            merged_seqlens,
            merged_docidx,
        )

    return counters


# ---------------------------------------------------------------------------
# Vectorized run detection
# ---------------------------------------------------------------------------


def _detect_runs(df: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Sort by (source_id, clip_num) and detect consecutive-clip runs.

    A new run starts whenever the source changes **or** clip_num is not the
    previous clip_num + 1.

    Returns (sorted_df, run_starts, run_lengths) where *run_starts* and
    *run_lengths* are 1-D int arrays indexing into the sorted dataframe.
    """
    df = df.sort(["source_id", "clip_num"])
    is_new_run = (
        (df["source_id"] != df["source_id"].shift(1))
        | (df["clip_num"] != (df["clip_num"].shift(1) + 1))
    ).fill_null(True)  # first row is always a run start
    starts = np.where(is_new_run.to_numpy())[0]
    lengths = np.diff(np.append(starts, len(df)))
    return df, starts, lengths


# ---------------------------------------------------------------------------
# Dry run — fast statistics via token-length integers only
# ---------------------------------------------------------------------------


def _dry_run(
    df: pl.DataFrame,
    patterns: list[str],
    patterns_by_size: dict[int, list[str]],
    min_window: int,
    sub_patterns_by_size: dict[int, list[str]],
    bos_id: int,
    eos_id: int,
    switch_id: int,
    transcribe_id: int,
    dtype: type,
    parquet_dir: Path,
) -> None:
    """Compute and print per-pattern + cascade + transcribe statistics.

    Only materialises token *lengths* (integers), never the token lists
    themselves, so memory stays at ~2 GB instead of ~78 GB.

    Saves a ``dry_run_stats.json`` to *parquet_dir* for later reference.
    """
    # 1. Compute list lengths, drop heavy list columns -----------------
    print("Computing token lengths (dropping raw tokens) ...")
    df = df.with_columns(
        df["audio_tokens"].list.len().cast(pl.UInt64).alias("_alen"),
        df["text_tokens"].list.len().cast(pl.UInt64).alias("_tlen"),
    ).select(["source_id", "clip_num", "_alen", "_tlen"])

    # 2. Vectorized run detection --------------------------------------
    print("Detecting consecutive runs ...")
    df, run_starts, run_lengths = _detect_runs(df)
    audio_lens = df["_alen"].to_numpy()
    text_lens = df["_tlen"].to_numpy()

    n_runs = len(run_starts)
    print(f"  {n_runs:,} runs across {len(df):,} clips")

    # 3. Pre-compute constants for all patterns + sub-patterns ---------
    all_interleaved: list[str] = list(patterns)
    for subs in sub_patterns_by_size.values():
        all_interleaved.extend(subs)

    pat_info: dict[str, tuple[list[int], list[int], int]] = {}
    for pat in all_interleaved:
        pat_info[pat] = _pattern_constants(pat)

    # 4. Counters and collectors ---------------------------------------
    counters: dict[str, dict[str, int]] = {
        p: {"seqs": 0, "tokens": 0, "audio_tokens": 0, "text_tokens": 0}
        for p in all_interleaved
    }
    seq_lens_parts: dict[str, list[np.ndarray]] = {
        p: [] for p in all_interleaved
    }
    tr_counter: dict[str, int] = {"seqs": 0, "tokens": 0, "audio_tokens": 0, "text_tokens": 0}
    tr_seq_lens_parts: list[np.ndarray] = []

    # 5. Iterate runs --------------------------------------------------
    t0 = time.time()
    for r in range(n_runs):
        rs = int(run_starts[r])
        rl = int(run_lengths[r])

        run_a = audio_lens[rs : rs + rl]
        run_t = text_lens[rs : rs + rl]

        # --- main interleaved windows ---
        for wsz, pats in patterns_by_size.items():
            if rl < wsz:
                continue
            n_win = rl // wsz
            u = n_win * wsz
            a_mat = run_a[:u].reshape(n_win, wsz)
            t_mat = run_t[:u].reshape(n_win, wsz)
            for pat in pats:
                a_pos, t_pos, n_sw = pat_info[pat]
                a_sum = a_mat[:, a_pos].sum(axis=1)
                t_sum = t_mat[:, t_pos].sum(axis=1)
                per_win = np.int64(2 + n_sw) + a_sum + t_sum
                counters[pat]["seqs"] += n_win
                counters[pat]["tokens"] += int(per_win.sum())
                counters[pat]["audio_tokens"] += int(a_sum.sum())
                counters[pat]["text_tokens"] += int(t_sum.sum())
                seq_lens_parts[pat].append(per_win)

        # --- cascade remainder ---
        n_rem = rl % min_window
        if n_rem >= 2:
            sub_pats = sub_patterns_by_size.get(n_rem, [])
            if sub_pats:
                rem_off = rl - n_rem
                rem_a = run_a[rem_off : rem_off + n_rem].reshape(1, n_rem)
                rem_t = run_t[rem_off : rem_off + n_rem].reshape(1, n_rem)
                for sp in sub_pats:
                    a_pos, t_pos, n_sw = pat_info[sp]
                    a_s = rem_a[:, a_pos].sum(axis=1)
                    t_s = rem_t[:, t_pos].sum(axis=1)
                    sl = np.int64(2 + n_sw) + a_s + t_s
                    counters[sp]["seqs"] += 1
                    counters[sp]["tokens"] += int(sl.sum())
                    counters[sp]["audio_tokens"] += int(a_s.sum())
                    counters[sp]["text_tokens"] += int(t_s.sum())
                    seq_lens_parts[sp].append(sl)
        elif n_rem == 1:
            idx = rl - 1
            a_val = int(run_a[idx])
            t_val = int(run_t[idx])
            sl_val = 3 + a_val + t_val  # BOS + transcribe_id + EOS
            tr_counter["seqs"] += 1
            tr_counter["tokens"] += sl_val
            tr_counter["audio_tokens"] += a_val
            tr_counter["text_tokens"] += t_val
            tr_seq_lens_parts.append(np.array([sl_val]))

        if r > 0 and r % 1_000_000 == 0:
            print(f"  {r:,}/{n_runs:,} runs ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0

    # 6. Print results -------------------------------------------------
    rl_arr = run_lengths
    print("\n" + "=" * 70)
    print("CONSECUTIVE-RUN DISTRIBUTION")
    print("=" * 70)
    print(f"  Total runs:   {n_runs:,}")
    print(f"  Total clips:  {int(rl_arr.sum()) if n_runs else 0:,}")
    if n_runs:
        print(f"  Mean length:  {rl_arr.mean():.1f}")
        print(f"  Median:       {np.median(rl_arr):.0f}")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"  P{p:02d}:          {np.percentile(rl_arr, p):.0f}")

    print("\n" + "=" * 70)
    print("PER-PATTERN STATISTICS")
    print("=" * 70)
    bytes_per_tok = DType.size(dtype)

    total_seqs = 0
    total_toks = 0
    total_bytes = 0

    def _print_pattern_stats(key: str, label: str | None = None) -> None:
        nonlocal total_seqs, total_toks, total_bytes
        c = counters[key]
        sl = (
            np.concatenate(seq_lens_parts[key])
            if seq_lens_parts[key]
            else np.array([0])
        )
        b = c["tokens"] * bytes_per_tok
        total_seqs += c["seqs"]
        total_toks += c["tokens"]
        total_bytes += b
        tag = label or key
        print(f"\n  {tag}  (window={len(key)})")
        print(f"    Sequences:    {c['seqs']:>14,}")
        print(f"    Total tokens: {c['tokens']:>14,}")
        print(f"    Est .bin size:{b / 1e9:>13.2f} GB")
        if c["seqs"] > 0:
            print(
                f"    Seq length — mean: {sl.mean():.1f}  "
                f"median: {np.median(sl):.0f}  "
                f"min: {sl.min()}  max: {sl.max()}"
            )
            print(
                f"               P05: {np.percentile(sl, 5):.0f}  "
                f"P25: {np.percentile(sl, 25):.0f}  "
                f"P75: {np.percentile(sl, 75):.0f}  "
                f"P95: {np.percentile(sl, 95):.0f}"
            )

    # Main patterns
    for pat in patterns:
        _print_pattern_stats(pat)

    # Sub-patterns (cascade)
    if sub_patterns_by_size:
        print(f"\n  --- cascade sub-patterns (from remainders) ---")
        for k in sorted(sub_patterns_by_size, reverse=True):
            for sp in sub_patterns_by_size[k]:
                _print_pattern_stats(sp, label=f"{sp} (cascade)")

    # Transcribe
    tr_sl = (
        np.concatenate(tr_seq_lens_parts)
        if tr_seq_lens_parts
        else np.array([0])
    )
    tr_bytes = tr_counter["tokens"] * bytes_per_tok
    total_seqs += tr_counter["seqs"]
    total_toks += tr_counter["tokens"]
    total_bytes += tr_bytes
    print(f"\n  transcribe  (1-clip remainder)")
    print(f"    Sequences:    {tr_counter['seqs']:>14,}")
    print(f"    Total tokens: {tr_counter['tokens']:>14,}")
    print(f"    Est .bin size:{tr_bytes / 1e9:>13.2f} GB")
    if tr_counter["seqs"] > 0:
        print(
            f"    Seq length — mean: {tr_sl.mean():.1f}  "
            f"median: {np.median(tr_sl):.0f}  "
            f"min: {tr_sl.min()}  max: {tr_sl.max()}"
        )

    print(f"\n  {'─' * 50}")
    print(f"  TOTAL:")
    print(f"    Sequences:    {total_seqs:>14,}")
    print(f"    Tokens:       {total_toks:>14,}")
    print(f"    Est disk:     {total_bytes / 1e9:>13.2f} GB")
    n_sources = df["source_id"].n_unique()
    print(f"\n  Sources: {n_sources:,}  |  Time: {elapsed:.1f}s")
    print("=" * 70)

    # 7. Save plain-text summary -----------------------------------------
    lines: list[str] = []
    lines.append("Dry Run Stats")
    lines.append("=" * 60)
    total_clips = int(rl_arr.sum()) if n_runs else 0
    lines.append(f"Clips:   {total_clips:>14,}")
    lines.append(f"Sources: {n_sources:>14,}")
    lines.append(f"Runs:    {n_runs:>14,}")
    lines.append(f"Min window size: {min_window}")
    lines.append("")

    if n_runs:
        lines.append("Run length distribution")
        lines.append("-" * 40)
        lines.append(f"  mean={rl_arr.mean():.1f}  median={np.median(rl_arr):.0f}  max={rl_arr.max()}")
        for p in [25, 75, 90, 95, 99]:
            lines.append(f"  P{p:02d}={np.percentile(rl_arr, p):.0f}")
        lines.append("")

    hdr = f"{'Pattern':<12s} {'Win':>3s} {'Sequences':>14s} {'Tokens':>16s} {'Audio toks':>16s} {'Text toks':>14s} {'Est GB':>8s} {'Mean len':>8s} {'Median':>7s}"
    sep = f"{'-'*12} {'-'*3} {'-'*14} {'-'*16} {'-'*16} {'-'*14} {'-'*8} {'-'*8} {'-'*7}"
    lines.append(hdr)
    lines.append(sep)
    total_audio_toks = 0
    total_text_toks = 0
    for key in all_interleaved:
        c = counters[key]
        sl = np.concatenate(seq_lens_parts[key]) if seq_lens_parts[key] else np.array([0])
        gb = c["tokens"] * bytes_per_tok / 1e9
        mean_sl = f"{sl.mean():.0f}" if c["seqs"] > 0 else "-"
        med_sl = f"{np.median(sl):.0f}" if c["seqs"] > 0 else "-"
        total_audio_toks += c["audio_tokens"]
        total_text_toks += c["text_tokens"]
        lines.append(f"{key:<12s} {len(key):>3d} {c['seqs']:>14,} {c['tokens']:>16,} {c['audio_tokens']:>16,} {c['text_tokens']:>14,} {gb:>8.2f} {mean_sl:>8s} {med_sl:>7s}")
    # transcribe row
    tr_gb = tr_bytes / 1e9
    tr_mean = f"{tr_sl.mean():.0f}" if tr_counter["seqs"] > 0 else "-"
    tr_med = f"{np.median(tr_sl):.0f}" if tr_counter["seqs"] > 0 else "-"
    total_audio_toks += tr_counter["audio_tokens"]
    total_text_toks += tr_counter["text_tokens"]
    lines.append(f"{'transcribe':<12s} {'1':>3s} {tr_counter['seqs']:>14,} {tr_counter['tokens']:>16,} {tr_counter['audio_tokens']:>16,} {tr_counter['text_tokens']:>14,} {tr_gb:>8.2f} {tr_mean:>8s} {tr_med:>7s}")
    lines.append(sep)
    lines.append(f"{'TOTAL':<12s} {'':>3s} {total_seqs:>14,} {total_toks:>16,} {total_audio_toks:>16,} {total_text_toks:>14,} {total_bytes / 1e9:>8.2f}")

    stats_path = parquet_dir / "dry_run_stats.txt"
    with open(stats_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nStats saved to {stats_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert interleaved parquet tokens to Megatron indexed datasets."
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        required=True,
        help="Directory containing rank_*_chunk_*.parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for bin/idx files.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to the omni tokenizer (for BOS/EOS/vocab_size).",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["ATAT", "TATA"],
        help="Space-separated pattern strings (default: ATAT TATA).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-pattern statistics without writing bin/idx files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel workers for building indexed datasets. "
        "0 (default) = auto (cpu_count - 2), 1 = single-threaded.",
    )
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    output_dir = Path(args.output_dir)
    dry_run = args.dry_run

    # Validate patterns
    for p in args.patterns:
        if len(p) < 2 or not all(c in "AT" for c in p):
            parser.error(
                f"Invalid pattern '{p}': must be ≥2 chars and contain only 'A'/'T'."
            )
    dupes = [p for p in args.patterns if args.patterns.count(p) > 1]
    if dupes:
        parser.error(f"Duplicate pattern(s): {sorted(set(dupes))}")

    if dry_run:
        print("*** DRY RUN — no files will be written ***\n")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load tokenizer metadata
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from {args.tokenizer_path} ...")
    bos_id, eos_id, switch_id, transcribe_id, vocab_size = load_token_ids(
        args.tokenizer_path
    )
    print(
        f"  bos_id={bos_id}  eos_id={eos_id}  switch_id={switch_id}  "
        f"transcribe_id={transcribe_id}  vocab_size={vocab_size}"
    )

    dtype = DType.optimal_dtype(vocab_size)
    print(f"  dtype={dtype.__name__}")

    # ------------------------------------------------------------------
    # 2. Load all parquets
    # ------------------------------------------------------------------
    parquet_files = sorted(
        p
        for p in parquet_dir.glob("rank_*_chunk_*.parquet")
        if not p.name.endswith(".tmp")
    )
    print(f"\nFound {len(parquet_files)} parquet files in {parquet_dir}")

    t0 = time.time()
    df = pl.read_parquet(
        parquet_files,
        columns=["source_id", "clip_num", "audio_tokens", "text_tokens"],
    )
    elapsed = time.time() - t0
    print(f"Loaded {len(df):,} clips in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 3. Patterns + cascade sub-patterns
    # ------------------------------------------------------------------
    patterns_by_size = group_patterns_by_size(args.patterns)
    min_window = min(patterns_by_size)
    sub_patterns_by_size = derive_sub_patterns(args.patterns, min_window)

    print(f"\nMain patterns:    { {k: v for k, v in sorted(patterns_by_size.items())} }")
    if sub_patterns_by_size:
        print(f"Cascade sub-pats: { {k: v for k, v in sorted(sub_patterns_by_size.items())} }")
    print(f"Min window (remainder threshold): {min_window}")

    if dry_run:
        _dry_run(
            df, args.patterns, patterns_by_size, min_window,
            sub_patterns_by_size, bos_id, eos_id, switch_id,
            transcribe_id, dtype, parquet_dir,
        )
        return

    # ------------------------------------------------------------------
    # 4. Vectorized run detection
    # ------------------------------------------------------------------
    print("\nDetecting consecutive runs ...")
    df, run_starts, run_lengths = _detect_runs(df)
    n_runs = len(run_starts)
    print(f"  {n_runs:,} runs")

    # Cache metadata before releasing the DataFrame
    n_clips = len(df)
    n_sources = df["source_id"].n_unique()

    # Keep token columns as PyArrow chunked arrays (~50 GB in compact
    # binary buffers).  We batch-extract per run below instead of
    # converting all 42M rows to Python lists (~200+ GB OOM).
    import pyarrow as pa
    _audio_raw = df["audio_tokens"].to_arrow()
    _text_raw = df["text_tokens"].to_arrow()
    audio_arrow = _audio_raw.combine_chunks() if isinstance(_audio_raw, pa.ChunkedArray) else _audio_raw
    text_arrow = _text_raw.combine_chunks() if isinstance(_text_raw, pa.ChunkedArray) else _text_raw
    del _audio_raw, _text_raw

    # Free the polars DataFrame — token data lives in the Arrow arrays
    del df

    # ------------------------------------------------------------------
    # 5. Collect all pattern keys
    # ------------------------------------------------------------------
    all_keys: list[str] = list(args.patterns)
    for subs in sub_patterns_by_size.values():
        all_keys.extend(subs)
    all_keys.append(TR_KEY)

    # ------------------------------------------------------------------
    # 6. Determine worker count
    # ------------------------------------------------------------------
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"\nUsing {num_workers} worker(s)")

    # ------------------------------------------------------------------
    # 7. Build indexed datasets (always via fork workers)
    # ------------------------------------------------------------------
    global _shared_audio_arrow, _shared_text_arrow
    global _shared_run_starts, _shared_run_lengths
    _shared_audio_arrow = audio_arrow
    _shared_text_arrow = text_arrow
    _shared_run_starts = run_starts
    _shared_run_lengths = run_lengths

    run_ranges = _partition_runs(run_lengths, num_workers)
    actual_workers = len(run_ranges)
    print(f"  Partitioned {n_runs:,} runs into {actual_workers} chunks")

    if actual_workers == 0:
        # Empty input — write empty bin/idx for every key
        counters: dict[str, dict[str, int]] = {}
        for key in all_keys:
            open(get_bin_path(str(output_dir / key)), "wb").close()
            _write_idx_file(
                get_idx_path(str(output_dir / key)),
                dtype,
                np.array([], dtype=np.int32),
                np.array([0], dtype=np.int64),
            )
            counters[key] = {"seqs": 0, "tokens": 0}
    else:
        tmp_dir = output_dir / "_tmp_shards"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        worker_args = [
            (
                wid,
                rng[0],
                rng[1],
                all_keys,
                patterns_by_size,
                sub_patterns_by_size,
                min_window,
                bos_id,
                eos_id,
                switch_id,
                transcribe_id,
                dtype,
                str(tmp_dir),
            )
            for wid, rng in enumerate(run_ranges)
        ]

        t0 = time.time()
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(actual_workers) as pool:
            worker_results = pool.starmap(_process_run_chunk, worker_args)
        elapsed = time.time() - t0
        print(f"\nWorkers finished in {elapsed:.1f}s")

        t_merge = time.time()
        counters = _merge_shards(worker_results, all_keys, output_dir, dtype, tmp_dir)
        print(f"Merged shards in {time.time() - t_merge:.1f}s")

        shutil.rmtree(tmp_dir)

    _shared_audio_arrow = None
    _shared_text_arrow = None
    _shared_run_starts = None
    _shared_run_lengths = None

    # ------------------------------------------------------------------
    # 8. Write metadata
    # ------------------------------------------------------------------
    metadata: dict = {
        "tokenizer_path": args.tokenizer_path,
        "parquet_dir": str(parquet_dir),
        "vocab_size": vocab_size,
        "dtype": dtype.__name__,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "switch_id": switch_id,
        "transcribe_id": transcribe_id,
        "min_window_size": min_window,
        "total_clips": n_clips,
        "total_sources": n_sources,
        "patterns": {},
    }
    if sub_patterns_by_size:
        metadata["cascade_sub_patterns"] = {
            str(k): v for k, v in sub_patterns_by_size.items()
        }

    for key in all_keys:
        c = counters[key]
        metadata["patterns"][key] = {
            "window_size": len(key) if key != TR_KEY else 1,
            "sequences": c["seqs"],
            "tokens": c["tokens"],
        }
        print(f"  {key}: {c['seqs']:,} sequences, {c['tokens']:,} tokens")

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {metadata_path}")
    print("Done.")


if __name__ == "__main__":
    main()
