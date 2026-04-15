"""Variable-length accumulate-based interleaved indexed dataset builder.

For each run of N consecutive clips (N >= 2), sequences extend through
the entire run, cutting only at run boundaries or ``--max-seq-len``.
Both AT and TA directions are produced independently (doubling data).

Sequence semantics:
  - AT direction: alternate A, T, A, T, ... starting with A
  - TA direction: alternate T, A, T, A, ... starting with T
  - A sequence accumulates clips until: (a) run ends, or (b) next clip
    would exceed ``--max-seq-len``
  - **Even-clip constraint**: sequences must end on the opposite modality
    from their start (AT→...T, TA→...A).  If the last clip leaves an odd
    count, it is pushed back so the next sequence consumes it.
  - On cut, remaining clips continue into a new sequence **restarting the
    same starting direction**
  - Transition tokens: ``stt_continue_id`` at A→T, ``tts_continue_id`` at T→A
  - Single-clip runs (N=1) → transcribe.bin as [BOS, audio, stt_transcribe, text, EOS]
  - Single-clip remainder after max_seq_len cut → also transcribe
  - ``--transcribe-ratio``: guarantee a minimum fraction of transcribe
    sequences by converting randomly-selected multi-clip runs to
    individual transcribe sequences (each clip becomes its own sequence,
    preserving all data).  Acts as a floor — if the natural ratio already
    meets the target, no conversion is done.

Output files:
  - ``AT.bin/.idx``  — all AT-starting accumulated sequences
  - ``TA.bin/.idx``  — all TA-starting accumulated sequences
  - ``transcribe.bin/.idx`` — single-clip transcriptions

Usage
-----
    python -m audio_tokenization.utils.build_interleaved.accumulate \\
        --parquet-dir /path/to/parquets \\
        --output-dir /path/to/output \\
        --tokenizer-path /path/to/tokenizer \\
        --max-seq-len 8192 \\
        --transcribe-ratio 0.5 \\
        --dry-run \\
        --num-workers 0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np

from audio_tokenization.utils.build_interleaved.common import (
    TR_KEY,
    DType,
    IndexedDatasetBuilder,
    _merge_shards,
    _partition_runs,
    _write_idx_file,
    compute_ratio_adjustment,
    format_distribution,
    get_bin_path,
    get_idx_path,
    list_interleave_cache_partitions,
    load_interleave_cache,
    load_token_ids,
    print_partition_stats,
    prepare_interleave_cache_and_runs,
    prepare_length_metadata,
)

# ---------------------------------------------------------------------------
# Module-level globals for fork-based sharing (set before Pool creation)
# ---------------------------------------------------------------------------

_shared_cache = None
_shared_run_starts = None
_shared_run_lengths = None
_shared_transcribe_only_runs: set[int] = set()


# Both directions are always produced (separate bin/idx each).
DIRECTIONS = ["AT", "TA"]

# ---------------------------------------------------------------------------
# Core accumulation logic
# ---------------------------------------------------------------------------


def _build_accumulated_seq(
    run_audio: list[list[int]],
    run_text: list[list[int]],
    direction: str,
    start: int,
    count: int,
    bos_id: int,
    eos_id: int,
    stt_continue_id: int,
    tts_continue_id: int,
) -> list[int]:
    """Build an accumulated sequence from *count* clips starting at *start*.

    Helper used when we need to rebuild after trimming an odd-count sequence.
    """
    seq: list[int] = [bos_id]
    prev_mode: str | None = None
    for j in range(count):
        pos_in_window = j % len(direction)
        mode = direction[pos_in_window]
        clip_tokens = run_audio[start + j] if mode == "A" else run_text[start + j]

        if mode == "A" and prev_mode == "T":
            seq.append(tts_continue_id)
        elif mode == "T" and prev_mode == "A":
            seq.append(stt_continue_id)

        seq.extend(clip_tokens)
        prev_mode = mode
    seq.append(eos_id)
    return seq


def _accumulate_sequences(
    run_audio: list[list[int]],
    run_text: list[list[int]],
    direction: str,
    max_seq_len: int,
    bos_id: int,
    eos_id: int,
    stt_continue_id: int,
    tts_continue_id: int,
) -> tuple[list[list[int]], list[int]]:
    """Accumulate clips into variable-length sequences.

    Each sequence must contain an even number of clips so that it starts
    and ends on opposite modalities (AT→...T, TA→...A).  If the last clip
    added would leave an odd count, it is pushed back to the remainder.

    Returns:
        sequences: list of complete sequences (each includes BOS/EOS),
                   each with >= 2 clips
        single_clip_indices: indices of clips that ended up as single-clip
                             remainders (to be routed to transcribe)
    """
    sequences: list[list[int]] = []
    single_clip_indices: list[int] = []
    n = len(run_audio)
    i = 0  # clip cursor

    while i < n:
        seq_start = i
        seq = [bos_id]
        prev_mode: str | None = None
        clips_in_seq = 0
        # Track the sequence state at every even-clip boundary so we can
        # revert cheaply instead of rebuilding.
        last_even_len = len(seq)  # seq length at last even count
        last_even_clips = 0

        while i < n:
            pos_in_window = clips_in_seq % len(direction)
            mode = direction[pos_in_window]

            clip_tokens = run_audio[i] if mode == "A" else run_text[i]
            transition_cost = 1 if (
                (mode == "A" and prev_mode == "T") or
                (mode == "T" and prev_mode == "A")
            ) else 0

            # +1 for EOS
            if len(seq) + transition_cost + len(clip_tokens) + 1 > max_seq_len and clips_in_seq > 0:
                break

            if mode == "A" and prev_mode == "T":
                seq.append(tts_continue_id)
            elif mode == "T" and prev_mode == "A":
                seq.append(stt_continue_id)

            seq.extend(clip_tokens)
            prev_mode = mode
            clips_in_seq += 1
            i += 1

            # Save checkpoint at every even clip boundary
            if clips_in_seq % 2 == 0:
                last_even_len = len(seq)
                last_even_clips = clips_in_seq

        # Enforce even-clip constraint
        if clips_in_seq % 2 == 1:
            if last_even_clips >= 2:
                # Revert to last even checkpoint — remaining clips
                # (from seq_start + last_even_clips onward) will be
                # re-processed by the outer loop.
                i = seq_start + last_even_clips
                clips_in_seq = last_even_clips
                seq = seq[:last_even_len]
            else:
                # 1 clip (or 0 even checkpoint) → route to transcribe
                single_clip_indices.append(seq_start)
                # Advance past this single clip
                i = seq_start + 1
                continue

        if clips_in_seq >= 2:
            seq.append(eos_id)
            sequences.append(seq)
        elif clips_in_seq == 1:
            single_clip_indices.append(seq_start)
        # clips_in_seq == 0 shouldn't happen, but guard anyway

    return sequences, single_clip_indices


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _accumulate_run_chunk(
    worker_id: int,
    run_start: int,
    run_end: int,
    all_keys: list[str],
    directions: list[str],
    max_seq_len: int,
    bos_id: int,
    eos_id: int,
    stt_continue_id: int,
    stt_transcribe_id: int,
    tts_continue_id: int,
    dtype: type,
    tmp_dir: str,
    seq_threshold: int | None = None,
) -> dict[str, dict]:
    """Process a contiguous range of runs for accumulate mode.

    Reads prepared cache views / run arrays from module-level globals (inherited via fork COW).
    When *seq_threshold* is set, sequences are routed to ``stage2/`` or ``lct/``
    subdirectories based on whether their length exceeds the threshold.
    """
    cache = _shared_cache
    run_starts_arr = _shared_run_starts
    run_lengths_arr = _shared_run_lengths
    transcribe_only_runs = _shared_transcribe_only_runs

    # Build output keys: with routing, each key gets stage2/ and lct/ variants
    if seq_threshold is not None:
        buckets = ["stage2", "lct"]
    else:
        buckets = [None]

    builders: dict[str, IndexedDatasetBuilder] = {}
    counters: dict[str, dict[str, int]] = {}
    shard_prefixes: dict[str, str] = {}
    for key in all_keys:
        for bucket in buckets:
            if bucket is not None:
                bkey = f"{bucket}/{key}"
                shard_prefix = f"{tmp_dir}/{bucket}_{key}_shard{worker_id:04d}"
            else:
                bkey = key
                shard_prefix = f"{tmp_dir}/{key}_shard{worker_id:04d}"
            builders[bkey] = IndexedDatasetBuilder(
                get_bin_path(shard_prefix), dtype=dtype,
            )
            counters[bkey] = {"seqs": 0, "tokens": 0}
            shard_prefixes[bkey] = shard_prefix

    # Pre-compute routed keys to avoid per-sequence f-string allocation
    if seq_threshold is not None:
        _routed_stage2 = {k: f"stage2/{k}" for k in all_keys}
        _routed_lct = {k: f"lct/{k}" for k in all_keys}
        def _route_key(base_key: str, seq_len: int) -> str:
            return _routed_stage2[base_key] if seq_len <= seq_threshold else _routed_lct[base_key]
    else:
        def _route_key(base_key: str, seq_len: int) -> str:
            return base_key

    for r in range(run_start, run_end):
        rs = int(run_starts_arr[r])
        rl = int(run_lengths_arr[r])

        run_audio = cache.audio.slice(rs, rl)
        run_text = cache.text.slice(rs, rl)

        # Ratio-adjusted: convert entire run to individual transcribe seqs
        if r in transcribe_only_runs:
            for c in range(rl):
                seq = [bos_id]
                seq.extend(run_audio[c])
                seq.append(stt_transcribe_id)
                seq.extend(run_text[c])
                seq.append(eos_id)

                rk = _route_key(TR_KEY, len(seq))
                builders[rk].add_item(seq)
                builders[rk].end_document()
                counters[rk]["seqs"] += 1
                counters[rk]["tokens"] += len(seq)
            continue

        if rl == 1:
            # Single-clip run → transcribe
            seq = [bos_id]
            seq.extend(run_audio[0])
            seq.append(stt_transcribe_id)
            seq.extend(run_text[0])
            seq.append(eos_id)

            rk = _route_key(TR_KEY, len(seq))
            builders[rk].add_item(seq)
            builders[rk].end_document()
            counters[rk]["seqs"] += 1
            counters[rk]["tokens"] += len(seq)
            continue

        # Multi-clip run → accumulate for each direction
        # Collect all single-clip indices across directions to avoid duplicates
        all_single_indices: set[int] = set()

        for direction in directions:
            sequences, single_indices = _accumulate_sequences(
                run_audio, run_text, direction, max_seq_len,
                bos_id, eos_id, stt_continue_id, tts_continue_id,
            )
            for seq in sequences:
                rk = _route_key(direction, len(seq))
                builders[rk].add_item(seq)
                builders[rk].end_document()
                counters[rk]["seqs"] += 1
                counters[rk]["tokens"] += len(seq)

            all_single_indices.update(single_indices)

        # Emit single-clip remainders to transcribe (deduplicated)
        for idx in sorted(all_single_indices):
            seq = [bos_id]
            seq.extend(run_audio[idx])
            seq.append(stt_transcribe_id)
            seq.extend(run_text[idx])
            seq.append(eos_id)

            rk = _route_key(TR_KEY, len(seq))
            builders[rk].add_item(seq)
            builders[rk].end_document()
            counters[rk]["seqs"] += 1
            counters[rk]["tokens"] += len(seq)

    # Save sidecar .npy files, close .bin
    result: dict[str, dict] = {}
    for bkey, b in builders.items():
        b.data_file.close()
        shard_prefix = shard_prefixes[bkey]
        np.save(f"{shard_prefix}_seqlens.npy", np.array(b.sequence_lengths, dtype=np.int32))
        np.save(f"{shard_prefix}_docidx.npy", np.array(b.document_indices, dtype=np.int64))
        result[bkey] = {
            "seqs": counters[bkey]["seqs"],
            "tokens": counters[bkey]["tokens"],
            "shard_prefix": shard_prefix,
        }

    return result


# ---------------------------------------------------------------------------
# Dry run — fast statistics via token-length integers only
# ---------------------------------------------------------------------------


def _dry_run_accumulate_lengths(
    run_audio_lens: list[int],
    run_text_lens: list[int],
    direction: str,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Simulate accumulation using token *lengths* only (no materialization).

    Returns (seq_lengths, single_clip_indices) where seq_lengths are the
    total token counts per emitted sequence.
    """
    seq_lengths: list[int] = []
    single_clip_indices: list[int] = []
    n = len(run_audio_lens)
    i = 0

    while i < n:
        seq_start = i
        seq_len = 1  # BOS
        prev_mode: str | None = None
        clips_in_seq = 0
        last_even_len = seq_len
        last_even_clips = 0

        while i < n:
            pos_in_window = clips_in_seq % len(direction)
            mode = direction[pos_in_window]

            clip_len = run_audio_lens[i] if mode == "A" else run_text_lens[i]
            transition_cost = 1 if (
                (mode == "A" and prev_mode == "T") or
                (mode == "T" and prev_mode == "A")
            ) else 0

            if seq_len + transition_cost + clip_len + 1 > max_seq_len and clips_in_seq > 0:
                break

            seq_len += transition_cost + clip_len
            prev_mode = mode
            clips_in_seq += 1
            i += 1

            if clips_in_seq % 2 == 0:
                last_even_len = seq_len
                last_even_clips = clips_in_seq

        if clips_in_seq % 2 == 1:
            if last_even_clips >= 2:
                i = seq_start + last_even_clips
                clips_in_seq = last_even_clips
                seq_len = last_even_len
            else:
                single_clip_indices.append(seq_start)
                i = seq_start + 1
                continue

        if clips_in_seq >= 2:
            seq_lengths.append(seq_len + 1)  # +1 for EOS
        elif clips_in_seq == 1:
            single_clip_indices.append(seq_start)

    return seq_lengths, single_clip_indices


def _compute_per_run_stats_accumulate(
    audio_lens: np.ndarray,
    text_lens: np.ndarray,
    run_starts: np.ndarray,
    run_lengths: np.ndarray,
    directions: list[str],
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-pass: compute per-run interleaved and transcribe sequence counts.

    Uses a vectorized fast path for the common case (runs whose total token
    count fits within *max_seq_len* — the vast majority).  Only the rare
    long runs that might be cut fall back to per-run Python simulation.

    Returns (il_per_run, tr_per_run) arrays of shape (n_runs,).
    """
    n_runs = len(run_starts)
    n_dirs = len(directions)
    il_per_run = np.zeros(n_runs, dtype=np.int64)
    tr_per_run = np.zeros(n_runs, dtype=np.int64)

    # Single-clip runs → 0 interleaved, 1 transcribe
    tr_per_run[run_lengths == 1] = 1

    multi = run_lengths >= 2
    if not multi.any():
        return il_per_run, tr_per_run

    # --- Vectorized fast path: runs that definitely won't be cut ---
    # Upper-bound seq length:
    #   BOS(1) + sum(audio) + sum(text) + transitions(rl-1) + EOS(1)
    # This over-estimates (a clip contributes audio OR text, not both) but
    # is safe: if the bound <= max_seq_len, no cut can occur.
    audio_cs = np.empty(len(audio_lens) + 1, dtype=np.int64)
    audio_cs[0] = 0
    np.cumsum(audio_lens, out=audio_cs[1:])
    text_cs = np.empty(len(text_lens) + 1, dtype=np.int64)
    text_cs[0] = 0
    np.cumsum(text_lens, out=text_cs[1:])

    run_ends = run_starts + run_lengths
    run_audio_sum = audio_cs[run_ends] - audio_cs[run_starts]
    run_text_sum = text_cs[run_ends] - text_cs[run_starts]
    upper_bound = 2 + run_audio_sum + run_text_sum + (run_lengths - 1)

    no_cut = multi & (upper_bound <= max_seq_len)
    # Without cuts: one seq per direction; odd runs get 1 transcribe remainder
    il_per_run[no_cut] = n_dirs
    tr_per_run[no_cut & (run_lengths % 2 == 1)] = 1

    # --- Slow path: long runs that may be cut by max_seq_len ---
    slow_indices = np.where(multi & ~no_cut)[0]
    for r in slow_indices:
        rs = int(run_starts[r])
        rl = int(run_lengths[r])
        run_a = audio_lens[rs : rs + rl].tolist()
        run_t = text_lens[rs : rs + rl].tolist()

        all_single: set[int] = set()
        total_il = 0
        for direction in directions:
            seq_lens, single_indices = _dry_run_accumulate_lengths(
                run_a, run_t, direction, max_seq_len,
            )
            total_il += len(seq_lens)
            all_single.update(single_indices)

        il_per_run[r] = total_il
        tr_per_run[r] = len(all_single)

    return il_per_run, tr_per_run


def _dry_run_accumulate(
    df,
    directions: list[str],
    max_seq_len: int,
    bos_id: int,
    eos_id: int,
    stt_continue_id: int,
    stt_transcribe_id: int,
    tts_continue_id: int,
    dtype: type,
    parquet_dir: Path,
    transcribe_ratio: float | None = None,
    seq_threshold: int | None = None,
) -> None:
    """Compute and print accumulate-mode statistics without materializing tokens."""
    print("Computing token lengths (dropping raw tokens) ...")
    df = prepare_length_metadata(df)

    print("Detecting consecutive runs ...")
    df, run_starts, run_lengths = _detect_runs(df)
    audio_lens = df["_alen"].to_numpy()
    text_lens = df["_tlen"].to_numpy()

    n_runs = len(run_starts)
    n_sources = df["source_id"].n_unique()
    print(f"  {n_runs:,} runs across {len(df):,} clips")

    # --- Transcribe-ratio adjustment (pre-pass) ---
    transcribe_only_runs: set[int] = set()
    if transcribe_ratio is not None:
        print(f"\nComputing per-run stats for --transcribe-ratio {transcribe_ratio} ...")
        t_pre = time.time()
        il_per_run, tr_per_run = _compute_per_run_stats_accumulate(
            audio_lens, text_lens, run_starts, run_lengths, directions, max_seq_len,
        )
        print(f"  Pre-pass done in {time.time() - t_pre:.1f}s")
        transcribe_only_runs = compute_ratio_adjustment(
            il_per_run, tr_per_run, run_lengths, transcribe_ratio,
        )
        if transcribe_only_runs:
            print(f"  Converting {len(transcribe_only_runs):,} runs to transcribe-only")
        else:
            print("  Natural ratio already meets target — no adjustment needed")

    # Per-direction counters
    dir_counters: dict[str, dict[str, int]] = {
        d: {"seqs": 0, "tokens": 0, "audio_tokens": 0, "text_tokens": 0}
        for d in directions
    }
    dir_seq_lens: dict[str, list[int]] = {d: [] for d in directions}
    tr_counter: dict[str, int] = {"seqs": 0, "tokens": 0, "audio_tokens": 0, "text_tokens": 0}
    tr_seq_lens: list[int] = []

    t0 = time.time()
    for r in range(n_runs):
        rs = int(run_starts[r])
        rl = int(run_lengths[r])

        run_a = audio_lens[rs : rs + rl].tolist()
        run_t = text_lens[rs : rs + rl].tolist()

        # Ratio-adjusted: convert entire run to individual transcribe seqs
        if r in transcribe_only_runs:
            for c in range(rl):
                sl_val = 3 + run_a[c] + run_t[c]
                tr_counter["seqs"] += 1
                tr_counter["tokens"] += sl_val
                tr_counter["audio_tokens"] += run_a[c]
                tr_counter["text_tokens"] += run_t[c]
                tr_seq_lens.append(sl_val)
            continue

        if rl == 1:
            sl_val = 3 + run_a[0] + run_t[0]  # BOS + stt_transcribe + EOS
            tr_counter["seqs"] += 1
            tr_counter["tokens"] += sl_val
            tr_counter["audio_tokens"] += run_a[0]
            tr_counter["text_tokens"] += run_t[0]
            tr_seq_lens.append(sl_val)
            continue

        all_single_indices: set[int] = set()

        for direction in directions:
            seq_lens, single_indices = _dry_run_accumulate_lengths(
                run_a, run_t, direction, max_seq_len,
            )
            dir_counters[direction]["seqs"] += len(seq_lens)
            dir_counters[direction]["tokens"] += sum(seq_lens)
            dir_seq_lens[direction].extend(seq_lens)
            all_single_indices.update(single_indices)

        for idx in sorted(all_single_indices):
            sl_val = 3 + run_a[idx] + run_t[idx]
            tr_counter["seqs"] += 1
            tr_counter["tokens"] += sl_val
            tr_counter["audio_tokens"] += run_a[idx]
            tr_counter["text_tokens"] += run_t[idx]
            tr_seq_lens.append(sl_val)

        if r > 0 and r % 1_000_000 == 0:
            print(f"  {r:,}/{n_runs:,} runs ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0

    # Print results
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
    print("PER-DIRECTION STATISTICS (accumulate mode)")
    print(f"  max_seq_len = {max_seq_len}")
    print("=" * 70)
    bytes_per_tok = DType.size(dtype)

    total_seqs = 0
    total_toks = 0
    total_bytes = 0
    all_sl_arrays: list[np.ndarray] = []

    for direction in directions:
        c = dir_counters[direction]
        sl = np.array(dir_seq_lens[direction]) if dir_seq_lens[direction] else np.array([0])
        all_sl_arrays.append(sl)
        b = c["tokens"] * bytes_per_tok
        total_seqs += c["seqs"]
        total_toks += c["tokens"]
        total_bytes += b
        print(f"\n  {direction}  (accumulated)")
        print(f"    Sequences:    {c['seqs']:>14,}")
        print(f"    Total tokens: {c['tokens']:>14,}")
        print(f"    Est .bin size:{b / 1e9:>13.2f} GB")
        if c["seqs"] > 0:
            for line in format_distribution(sl, indent="    "):
                print(line)

    # Transcribe
    tr_sl = np.array(tr_seq_lens) if tr_seq_lens else np.array([0])
    all_sl_arrays.append(tr_sl)
    tr_bytes = tr_counter["tokens"] * bytes_per_tok
    total_seqs += tr_counter["seqs"]
    total_toks += tr_counter["tokens"]
    total_bytes += tr_bytes
    print(f"\n  transcribe  (single-clip)")
    print(f"    Sequences:    {tr_counter['seqs']:>14,}")
    print(f"    Total tokens: {tr_counter['tokens']:>14,}")
    print(f"    Est .bin size:{tr_bytes / 1e9:>13.2f} GB")
    if tr_counter["seqs"] > 0:
        for line in format_distribution(tr_sl, indent="    "):
            print(line)

    print(f"\n  {'─' * 50}")
    print(f"  TOTAL:")
    print(f"    Sequences:    {total_seqs:>14,}")
    print(f"    Tokens:       {total_toks:>14,}")
    print(f"    Est disk:     {total_bytes / 1e9:>13.2f} GB")
    if total_seqs > 0:
        actual_ratio = tr_counter["seqs"] / total_seqs
        print(f"    Transcribe ratio: {actual_ratio:.4f} ({actual_ratio * 100:.2f}%)")
        if transcribe_ratio is not None:
            print(f"    Target ratio:     {transcribe_ratio:.4f} ({transcribe_ratio * 100:.2f}%)")
            print(f"    Runs converted:   {len(transcribe_only_runs):,}")

    # Combined distribution
    combined_sl = np.concatenate(all_sl_arrays)
    if len(combined_sl) > 0:
        print(f"\n  COMBINED sequence length distribution ({len(combined_sl):,} sequences)")
        for line in format_distribution(combined_sl, indent="    "):
            print(line)

    # Stage2 / LCT routing summary
    if seq_threshold is not None and len(combined_sl) > 0:
        stage2_mask = combined_sl <= seq_threshold
        lct_mask = ~stage2_mask
        s2_seqs = int(stage2_mask.sum())
        lct_seqs = int(lct_mask.sum())
        s2_toks = int(combined_sl[stage2_mask].sum())
        lct_toks = int(combined_sl[lct_mask].sum())
        print(f"\n  {'─' * 50}")
        print(f"  ROUTING (seq_threshold = {seq_threshold:,})")
        print(f"    stage2: {s2_seqs:>12,} seqs  {s2_toks:>14,} tokens ({100*s2_toks/(s2_toks+lct_toks):.1f}%)")
        print(f"    lct:    {lct_seqs:>12,} seqs  {lct_toks:>14,} tokens ({100*lct_toks/(s2_toks+lct_toks):.1f}%)")

    print(f"\n  Sources: {n_sources:,}  |  Time: {elapsed:.1f}s")
    print("=" * 70)

    # Save plain-text summary
    lines: list[str] = []
    lines.append("Dry Run Stats (accumulate mode)")
    lines.append("=" * 60)
    total_clips = int(rl_arr.sum()) if n_runs else 0
    lines.append(f"Clips:       {total_clips:>14,}")
    lines.append(f"Sources:     {n_sources:>14,}")
    lines.append(f"Runs:        {n_runs:>14,}")
    lines.append(f"max_seq_len: {max_seq_len}")
    lines.append(f"Directions:  {' '.join(directions)}")
    lines.append("")

    if n_runs:
        lines.append("Run length distribution")
        lines.append("-" * 40)
        lines.append(f"  mean={rl_arr.mean():.1f}  median={np.median(rl_arr):.0f}  max={rl_arr.max()}")
        for p in [25, 75, 90, 95, 99]:
            lines.append(f"  P{p:02d}={np.percentile(rl_arr, p):.0f}")
        lines.append("")

    hdr = f"{'Direction':<12s} {'Sequences':>14s} {'Tokens':>16s} {'Est GB':>8s} {'Mean len':>8s} {'Median':>7s}"
    sep = f"{'-' * 12} {'-' * 14} {'-' * 16} {'-' * 8} {'-' * 8} {'-' * 7}"
    lines.append(hdr)
    lines.append(sep)
    for direction in directions:
        c = dir_counters[direction]
        sl = np.array(dir_seq_lens[direction]) if dir_seq_lens[direction] else np.array([0])
        gb = c["tokens"] * bytes_per_tok / 1e9
        lines.append(f"{direction:<12s} {c['seqs']:>14,} {c['tokens']:>16,} {gb:>8.2f}")
        if c["seqs"] > 0:
            lines.extend(format_distribution(sl, indent="  "))
        lines.append("")
    tr_gb = tr_bytes / 1e9
    lines.append(f"{'transcribe':<12s} {tr_counter['seqs']:>14,} {tr_counter['tokens']:>16,} {tr_gb:>8.2f}")
    if tr_counter["seqs"] > 0:
        lines.extend(format_distribution(tr_sl, indent="  "))
    lines.append("")
    lines.append(sep)
    lines.append(f"{'TOTAL':<12s} {total_seqs:>14,} {total_toks:>16,} {total_bytes / 1e9:>8.2f}")
    lines.append("")
    lines.append(f"COMBINED sequence length distribution ({len(combined_sl):,} sequences)")
    lines.extend(format_distribution(combined_sl, indent="  "))

    stats_path = parquet_dir / "dry_run_stats.txt"
    with open(stats_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nStats saved to {stats_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert interleaved parquet tokens to Megatron indexed datasets (accumulate mode)."
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
        "--max-seq-len",
        type=int,
        default=8192,
        help="Maximum sequence length in tokens (default: 8192).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing bin/idx files.",
    )
    parser.add_argument(
        "--transcribe-ratio",
        type=float,
        default=None,
        help="Minimum fraction of transcribe sequences (e.g. 0.1 = 10%%). "
        "If the natural ratio is already at or above the target, no "
        "adjustment is made. Otherwise, multi-clip runs are randomly "
        "converted to individual transcribe sequences until the target "
        "is met.",
    )
    parser.add_argument(
        "--seq-threshold",
        type=int,
        default=None,
        help="Sequence length threshold for routing. Sequences <= threshold "
        "go to stage2/, longer to lct/. If not set, no routing.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Directory for temporary shard files during build. "
        "Defaults to output_dir/_tmp_shards. Use fast node-local "
        "storage (e.g. /tmp) for best performance.",
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
    transcribe_ratio: float | None = args.transcribe_ratio

    if transcribe_ratio is not None and not (0.0 < transcribe_ratio < 1.0):
        parser.error("--transcribe-ratio must be between 0 and 1 (exclusive).")

    if dry_run:
        print("*** DRY RUN — no files will be written ***\n")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load tokenizer metadata
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from {args.tokenizer_path} ...")
    bos_id, eos_id, stt_continue_id, stt_transcribe_id, tts_continue_id, vocab_size = load_token_ids(
        args.tokenizer_path
    )
    print(
        f"  bos_id={bos_id}  eos_id={eos_id}  stt_continue_id={stt_continue_id}  "
        f"stt_transcribe_id={stt_transcribe_id}  tts_continue_id={tts_continue_id}  "
        f"vocab_size={vocab_size}"
    )

    dtype = DType.optimal_dtype(vocab_size)
    print(f"  dtype={dtype.__name__}")

    partition_dirs = list_interleave_cache_partitions(parquet_dir)

    # ------------------------------------------------------------------
    # 3. Print configuration
    # ------------------------------------------------------------------
    print(f"\nDirections:   {DIRECTIONS}")
    print(f"max_seq_len:  {args.max_seq_len}")

    if dry_run:
        if len(partition_dirs) > 1:
            raise RuntimeError(
                "Dry-run on a partitioned v2 cache root is not supported yet. "
                "Pass a leaf partition directory instead."
            )
        df, _cache_reader = load_interleave_cache(partition_dirs[0])
        _dry_run_accumulate(
            df, DIRECTIONS, args.max_seq_len,
            bos_id, eos_id, stt_continue_id, stt_transcribe_id,
            tts_continue_id, dtype, partition_dirs[0],
            transcribe_ratio=transcribe_ratio,
            seq_threshold=args.seq_threshold,
        )
        return

    # ------------------------------------------------------------------
    # 4. Transcribe-ratio adjustment (global across partitions)
    # ------------------------------------------------------------------
    transcribe_only_runs_by_partition: dict[Path, set[int]] = {}
    runs_converted_to_transcribe = 0
    if transcribe_ratio is not None:
        all_il_per_run = []
        all_tr_per_run = []
        all_run_lengths = []
        partition_run_counts = []
        print(f"\nComputing per-run stats for --transcribe-ratio {transcribe_ratio} ...")
        t_pre = time.time()
        for partition_dir in partition_dirs:
            df, _cache_reader = load_interleave_cache(partition_dir)
            length_df = prepare_length_metadata(df)
            sorted_df, run_starts, run_lengths = _detect_runs(length_df)
            audio_lens_arr = sorted_df["_alen"].to_numpy()
            text_lens_arr = sorted_df["_tlen"].to_numpy()
            il_per_run, tr_per_run = _compute_per_run_stats_accumulate(
                audio_lens_arr, text_lens_arr, run_starts, run_lengths,
                DIRECTIONS, args.max_seq_len,
            )
            all_il_per_run.append(il_per_run)
            all_tr_per_run.append(tr_per_run)
            all_run_lengths.append(run_lengths.astype(np.int64))
            partition_run_counts.append(len(run_lengths))
        print(f"  Pre-pass done in {time.time() - t_pre:.1f}s")
        selected_global_runs = compute_ratio_adjustment(
            np.concatenate(all_il_per_run) if all_il_per_run else np.array([], dtype=np.int64),
            np.concatenate(all_tr_per_run) if all_tr_per_run else np.array([], dtype=np.int64),
            np.concatenate(all_run_lengths) if all_run_lengths else np.array([], dtype=np.int64),
            transcribe_ratio,
        )
        runs_converted_to_transcribe = len(selected_global_runs)
        if selected_global_runs:
            print(f"  Converting {len(selected_global_runs):,} runs to transcribe-only")
        else:
            print("  Natural ratio already meets target — no adjustment needed")
        offset = 0
        for partition_dir, count in zip(partition_dirs, partition_run_counts):
            local = {
                run_idx - offset
                for run_idx in selected_global_runs
                if offset <= run_idx < offset + count
            }
            if local:
                transcribe_only_runs_by_partition[partition_dir] = local
            offset += count

    # ------------------------------------------------------------------
    # 5. Collect all output keys
    # ------------------------------------------------------------------
    all_keys: list[str] = list(DIRECTIONS) + [TR_KEY]
    if args.seq_threshold is not None:
        # Create output subdirectories for routing
        (output_dir / "stage2").mkdir(parents=True, exist_ok=True)
        (output_dir / "lct").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 6. Determine worker count
    # ------------------------------------------------------------------
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"\nUsing {num_workers} worker(s)")

    # ------------------------------------------------------------------
    # 7. Build indexed datasets partition-by-partition
    # ------------------------------------------------------------------
    global _shared_cache
    global _shared_run_starts, _shared_run_lengths
    global _shared_transcribe_only_runs
    if args.tmp_dir:
        tmp_dir = Path(args.tmp_dir) / f"_greedy_shards_{os.getpid()}"
    else:
        tmp_dir = output_dir / "_tmp_shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    total_clips = 0
    total_sources = 0
    partition_stats = []
    worker_results = []
    try:
        t0 = time.time()
        for part_idx, partition_dir in enumerate(partition_dirs):
            print(f"\nProcessing partition {part_idx + 1}/{len(partition_dirs)}: {partition_dir.name}")
            df, cache_reader = load_interleave_cache(partition_dir)
            cache, run_starts, run_lengths, n_clips, n_sources = (
                prepare_interleave_cache_and_runs(df, cache_reader)
            )
            audio_token_total = int(cache.audio_lengths.sum())
            text_token_total = int(cache.text_lengths.sum())
            total_clips += n_clips
            total_sources += n_sources
            n_runs = len(run_starts)
            _shared_cache = cache
            _shared_run_starts = run_starts
            _shared_run_lengths = run_lengths
            _shared_transcribe_only_runs = transcribe_only_runs_by_partition.get(partition_dir, set())

            run_ranges = _partition_runs(run_lengths, num_workers)
            actual_workers = len(run_ranges)
            print(f"  Partitioned {n_runs:,} runs into {actual_workers} chunks")
            partition_stats.append(
                {
                    "name": partition_dir.name,
                    "clips": n_clips,
                    "runs": n_runs,
                    "sources": n_sources,
                    "audio_tokens": audio_token_total,
                    "text_tokens": text_token_total,
                    "workers": actual_workers,
                }
            )
            if actual_workers == 0:
                _shared_cache = None
                _shared_run_starts = None
                _shared_run_lengths = None
                _shared_transcribe_only_runs = set()
                continue

            part_tmp_dir = tmp_dir / f"part_{part_idx:04d}_{partition_dir.name}"
            part_tmp_dir.mkdir(parents=True, exist_ok=True)
            worker_args = [
                (
                    wid,
                    rng[0],
                    rng[1],
                    all_keys,
                    DIRECTIONS,
                    args.max_seq_len,
                    bos_id,
                    eos_id,
                    stt_continue_id,
                    stt_transcribe_id,
                    tts_continue_id,
                    dtype,
                    str(part_tmp_dir),
                    args.seq_threshold,
                )
                for wid, rng in enumerate(run_ranges)
            ]

            # Pool creation must stay inside the partition loop, after the
            # current partition's globals are assigned. fork() captures these
            # globals for worker read-only access.
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(actual_workers) as pool:
                worker_results.extend(pool.starmap(_accumulate_run_chunk, worker_args))

            _shared_cache = None
            _shared_run_starts = None
            _shared_run_lengths = None
            _shared_transcribe_only_runs = set()
            del cache, df, cache_reader
        elapsed = time.time() - t0
        print(f"\nWorkers finished in {elapsed:.1f}s")

        if args.seq_threshold is not None:
            merge_keys = [f"{bucket}/{k}" for bucket in ("stage2", "lct") for k in all_keys]
        else:
            merge_keys = all_keys

        if worker_results:
            t_merge = time.time()
            counters = _merge_shards(worker_results, merge_keys, output_dir, dtype, tmp_dir)
            print(f"Merged shards in {time.time() - t_merge:.1f}s")
        else:
            counters = {}
            for key in merge_keys:
                open(get_bin_path(str(output_dir / key)), "wb").close()
                _write_idx_file(
                    get_idx_path(str(output_dir / key)),
                    dtype,
                    np.array([], dtype=np.int32),
                    np.array([0], dtype=np.int64),
                )
                counters[key] = {"seqs": 0, "tokens": 0}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _shared_cache = None
        _shared_run_starts = None
        _shared_run_lengths = None
        _shared_transcribe_only_runs = set()

    # ------------------------------------------------------------------
    # 8. Write metadata
    # ------------------------------------------------------------------
    metadata: dict = {
        "mode": "accumulate",
        "tokenizer_path": args.tokenizer_path,
        "parquet_dir": str(parquet_dir),
        "vocab_size": vocab_size,
        "dtype": dtype.__name__,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "stt_continue_id": stt_continue_id,
        "stt_transcribe_id": stt_transcribe_id,
        "tts_continue_id": tts_continue_id,
        "max_seq_len": args.max_seq_len,
        "directions": DIRECTIONS,
        "transcribe_ratio": transcribe_ratio,
        "runs_converted_to_transcribe": runs_converted_to_transcribe,
        "total_clips": total_clips,
        "total_sources": total_sources,
        "outputs": {},
    }
    metadata["partition_summary"] = print_partition_stats(partition_stats)
    metadata["partition_stats"] = partition_stats

    for key in sorted(counters.keys()):
        c = counters[key]
        metadata["outputs"][key] = {
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
