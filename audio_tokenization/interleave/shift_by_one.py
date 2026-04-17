"""Shift-by-one interleaved indexed dataset builder.

Each sequence is always AT-ordered and always ends with text:
  BOS, Audio[i], stt, Text[i+1], tts, Audio[i+2], stt, Text[i+3], ..., EOS

Two offsets are produced independently (doubling data):
  - offset=0: pairs (A[0],T[1]), (A[2],T[3]), ...
  - offset=1: pairs (A[1],T[2]), (A[3],T[4]), ...

Every sequence ends with text tokens, maximizing training signal when
audio token loss is 0.

Leftover clips (odd remainder) → transcribe.bin as single-clip sequences.

Output files:
  - ``offset_0.bin/.idx``  — even-offset accumulated sequences
  - ``offset_1.bin/.idx``  — odd-offset accumulated sequences
  - ``transcribe.bin/.idx`` — single-clip transcriptions

Usage::

    python -m audio_tokenization.interleave.shift_by_one \\
        --parquet-dir /path/to/interleave_cache/dataset \\
        --output-dir /path/to/output \\
        --tokenizer-path /path/to/tokenizer \\
        --max-seq-len 262144 \\
        --seq-threshold 8192 \\
        --transcribe-ratio 0.5 \\
        --dry-run
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .common import (
    TR_KEY,
    DType,
    _detect_runs,
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

logger = logging.getLogger(__name__)

OFFSETS = [0, 1]
OFFSET_KEYS = ["offset_0", "offset_1"]

_shared_cache = None
_shared_run_starts = None
_shared_run_lengths = None
_shared_transcribe_only_runs: set[int] = set()


# ---------------------------------------------------------------------------
# Core: build one AT-pair sequence
# ---------------------------------------------------------------------------

def _build_shift_sequence(
    run_audio, run_text,
    start: int, count: int,
    bos_id: int, eos_id: int,
    stt_continue_id: int, tts_continue_id: int,
) -> list[int]:
    """Build one shift-by-one sequence: A[start] T[start+1] A[start+2] T[start+3] ...

    Always starts with audio and ends with text.
    ``count`` must be even (number of clips consumed).
    """
    seq = [bos_id]
    for j in range(count):
        idx = start + j
        if j % 2 == 0:
            # Audio clip
            if j > 0:
                seq.append(tts_continue_id)
            seq.extend(run_audio[idx])
        else:
            # Text clip
            seq.append(stt_continue_id)
            seq.extend(run_text[idx])
    seq.append(eos_id)
    return seq


def _accumulate_shift_sequences(
    run_audio, run_text,
    offset: int,
    max_seq_len: int,
    bos_id: int, eos_id: int,
    stt_continue_id: int, tts_continue_id: int,
) -> tuple[list[list[int]], list[int]]:
    """Accumulate shift-by-one sequences for a single run at a given offset.

    Returns (sequences, leftover_indices).
    """
    n = len(run_audio)
    sequences: list[list[int]] = []
    leftover_indices: list[int] = []

    i = offset
    while i + 1 < n:
        # Greedily pack pairs into one sequence
        seq_start = i
        pairs = 0
        est_len = 1  # BOS

        while i + 1 < n:
            a_len = len(run_audio[i])
            t_len = len(run_text[i + 1])
            transition_cost = 2 if pairs > 0 else 0  # tts + stt tokens
            pair_cost = a_len + t_len + transition_cost + (1 if pairs == 0 else 0)  # +1 for first stt
            # +1 for EOS
            if est_len + pair_cost + 1 > max_seq_len and pairs > 0:
                break
            est_len += a_len + t_len + (2 if pairs > 0 else 1)  # transitions
            pairs += 1
            i += 2

        if pairs > 0:
            count = pairs * 2
            seq = _build_shift_sequence(
                run_audio, run_text, seq_start, count,
                bos_id, eos_id, stt_continue_id, tts_continue_id,
            )
            sequences.append(seq)

    # Leftover: single clip at the end that couldn't pair
    if i < n:
        leftover_indices.append(i)

    return sequences, leftover_indices


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _shift_run_chunk(
    worker_id: int,
    run_start: int,
    run_end: int,
    all_keys: list[str],
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
    """Process a range of runs with shift-by-one accumulation."""
    from audio_tokenization.utils.indexed_dataset.indexed_dataset_megatron import IndexedDatasetBuilder

    cache = _shared_cache
    run_starts_arr = _shared_run_starts
    run_lengths_arr = _shared_run_lengths
    transcribe_only_runs = _shared_transcribe_only_runs

    # Route helper
    if seq_threshold is not None:
        buckets = ["stage2", "lct"]
        _s2 = {k: f"stage2/{k}" for k in all_keys}
        _lct = {k: f"lct/{k}" for k in all_keys}
        def _route(base_key, seq_len):
            return _s2[base_key] if seq_len <= seq_threshold else _lct[base_key]
    else:
        buckets = [None]
        def _route(base_key, seq_len):
            return base_key

    builders = {}
    counters = {}
    shard_prefixes = {}
    for key in all_keys:
        for bucket in buckets:
            if bucket is not None:
                bkey = f"{bucket}/{key}"
                sp = f"{tmp_dir}/{bucket}_{key}_shard{worker_id:04d}"
            else:
                bkey = key
                sp = f"{tmp_dir}/{key}_shard{worker_id:04d}"
            builders[bkey] = IndexedDatasetBuilder(get_bin_path(sp), dtype=dtype)
            counters[bkey] = {"seqs": 0, "tokens": 0}
            shard_prefixes[bkey] = sp

    def _emit(base_key, seq):
        rk = _route(base_key, len(seq))
        builders[rk].add_item(seq)
        builders[rk].end_document()
        counters[rk]["seqs"] += 1
        counters[rk]["tokens"] += len(seq)

    for r in range(run_start, run_end):
        rs = int(run_starts_arr[r])
        rl = int(run_lengths_arr[r])

        run_audio = cache.audio.slice(rs, rl)
        run_text = cache.text.slice(rs, rl)

        # Ratio-adjusted: entire run → individual transcribe
        if r in transcribe_only_runs:
            for c in range(rl):
                seq = [bos_id]
                seq.extend(run_audio[c])
                seq.append(stt_transcribe_id)
                seq.extend(run_text[c])
                seq.append(eos_id)
                _emit(TR_KEY, seq)
            continue

        if rl == 1:
            seq = [bos_id]
            seq.extend(run_audio[0])
            seq.append(stt_transcribe_id)
            seq.extend(run_text[0])
            seq.append(eos_id)
            _emit(TR_KEY, seq)
            continue

        # Multi-clip run → shift-by-one for each offset
        all_leftover: set[int] = set()

        for oi, offset in enumerate(OFFSETS):
            sequences, leftovers = _accumulate_shift_sequences(
                run_audio, run_text, offset, max_seq_len,
                bos_id, eos_id, stt_continue_id, tts_continue_id,
            )
            for seq in sequences:
                _emit(OFFSET_KEYS[oi], seq)
            all_leftover.update(leftovers)

        # Emit leftovers as transcribe
        for idx in sorted(all_leftover):
            seq = [bos_id]
            seq.extend(run_audio[idx])
            seq.append(stt_transcribe_id)
            seq.extend(run_text[idx])
            seq.append(eos_id)
            _emit(TR_KEY, seq)

    # Finalize
    result = {}
    for bkey, b in builders.items():
        b.data_file.close()
        sp = shard_prefixes[bkey]
        np.save(f"{sp}_seqlens.npy", np.array(b.sequence_lengths, dtype=np.int32))
        np.save(f"{sp}_docidx.npy", np.array(b.document_indices, dtype=np.int64))
        result[bkey] = {
            "seqs": counters[bkey]["seqs"],
            "tokens": counters[bkey]["tokens"],
            "shard_prefix": sp,
        }
    return result


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def _dry_run_shift(
    df,
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
    """Compute and print shift-by-one statistics without materializing tokens."""
    print("Computing token lengths ...")
    df = prepare_length_metadata(df)

    print("Detecting consecutive runs ...")
    df, run_starts, run_lengths = _detect_runs(df)
    audio_lens = df["_alen"].to_numpy()
    text_lens = df["_tlen"].to_numpy()

    n_runs = len(run_starts)
    n_sources = df["source_id"].n_unique()
    print(f"  {n_runs:,} runs across {len(df):,} clips")

    # Transcribe ratio adjustment — vectorized per-run sequence counts
    transcribe_only_runs: set[int] = set()
    if transcribe_ratio is not None:
        rl = run_lengths.astype(np.int64)
        single = rl == 1
        # offset=0: pairs = rl // 2, leftover = rl % 2
        # offset=1: pairs = (rl - 1) // 2, leftover = (rl - 1) % 2
        il_per_run = np.where(single, 0,
            np.maximum(1, rl // 2) + np.maximum(1, (rl - 1) // 2))
        tr_per_run = np.where(single, 1,
            (rl % 2) + ((rl - 1) % 2))

        transcribe_only_runs = compute_ratio_adjustment(
            il_per_run, tr_per_run, run_lengths, transcribe_ratio,
        )
        if transcribe_only_runs:
            print(f"  Converting {len(transcribe_only_runs):,} runs to transcribe-only")
        else:
            print("  Natural ratio already meets target")

    # Simulate
    offset_counters = {k: {"seqs": 0, "tokens": 0} for k in OFFSET_KEYS}
    offset_seq_lens = {k: [] for k in OFFSET_KEYS}
    tr_counter = {"seqs": 0, "tokens": 0}
    tr_seq_lens = []

    t0 = time.time()
    for r in range(n_runs):
        rs = int(run_starts[r])
        rl = int(run_lengths[r])
        run_a = audio_lens[rs: rs + rl].tolist()
        run_t = text_lens[rs: rs + rl].tolist()

        if r in transcribe_only_runs:
            for c in range(rl):
                sl = 3 + run_a[c] + run_t[c]
                tr_counter["seqs"] += 1
                tr_counter["tokens"] += sl
                tr_seq_lens.append(sl)
            continue

        if rl == 1:
            sl = 3 + run_a[0] + run_t[0]
            tr_counter["seqs"] += 1
            tr_counter["tokens"] += sl
            tr_seq_lens.append(sl)
            continue

        all_leftover: set[int] = set()
        for oi, offset in enumerate(OFFSETS):
            # Simulate accumulation
            i = offset
            while i + 1 < rl:
                seq_start = i
                pairs = 0
                est_len = 1  # BOS
                while i + 1 < rl:
                    a_len = run_a[i]
                    t_len = run_t[i + 1]
                    tc = 2 if pairs > 0 else 1
                    if est_len + a_len + t_len + tc + 1 > max_seq_len and pairs > 0:
                        break
                    est_len += a_len + t_len + tc
                    pairs += 1
                    i += 2
                if pairs > 0:
                    sl = est_len + 1  # +EOS
                    offset_counters[OFFSET_KEYS[oi]]["seqs"] += 1
                    offset_counters[OFFSET_KEYS[oi]]["tokens"] += sl
                    offset_seq_lens[OFFSET_KEYS[oi]].append(sl)
            if i < rl:
                all_leftover.add(i)

        for idx in sorted(all_leftover):
            sl = 3 + run_a[idx] + run_t[idx]
            tr_counter["seqs"] += 1
            tr_counter["tokens"] += sl
            tr_seq_lens.append(sl)

    elapsed = time.time() - t0
    bytes_per_tok = DType.size(dtype)

    print(f"\n{'=' * 70}")
    print("SHIFT-BY-ONE STATISTICS")
    print(f"  max_seq_len = {max_seq_len}")
    print(f"{'=' * 70}")

    total_seqs = 0
    total_toks = 0
    all_sl = []

    for key in OFFSET_KEYS:
        c = offset_counters[key]
        sl = np.array(offset_seq_lens[key]) if offset_seq_lens[key] else np.array([0])
        all_sl.append(sl)
        total_seqs += c["seqs"]
        total_toks += c["tokens"]
        print(f"\n  {key}")
        print(f"    Sequences:    {c['seqs']:>14,}")
        print(f"    Total tokens: {c['tokens']:>14,}")
        if c["seqs"] > 0:
            for line in format_distribution(sl, indent="    "):
                print(line)

    tr_sl = np.array(tr_seq_lens) if tr_seq_lens else np.array([0])
    all_sl.append(tr_sl)
    total_seqs += tr_counter["seqs"]
    total_toks += tr_counter["tokens"]
    print(f"\n  transcribe")
    print(f"    Sequences:    {tr_counter['seqs']:>14,}")
    print(f"    Total tokens: {tr_counter['tokens']:>14,}")
    if tr_counter["seqs"] > 0:
        for line in format_distribution(tr_sl, indent="    "):
            print(line)

    print(f"\n  {'─' * 50}")
    print(f"  TOTAL: {total_seqs:,} seqs, {total_toks:,} tokens, {total_toks * bytes_per_tok / 1e9:.2f} GB")
    if total_seqs > 0:
        actual_ratio = tr_counter["seqs"] / total_seqs
        print(f"  Transcribe ratio: {actual_ratio:.4f} ({actual_ratio * 100:.2f}%)")

    combined_sl = np.concatenate(all_sl)
    if seq_threshold is not None and len(combined_sl) > 0:
        s2_mask = combined_sl <= seq_threshold
        s2_toks = int(combined_sl[s2_mask].sum())
        lct_toks = int(combined_sl[~s2_mask].sum())
        print(f"\n  ROUTING (seq_threshold = {seq_threshold:,})")
        print(f"    stage2: {int(s2_mask.sum()):>12,} seqs  {s2_toks:>14,} tokens ({100*s2_toks/(s2_toks+lct_toks):.1f}%)")
        print(f"    lct:    {int((~s2_mask).sum()):>12,} seqs  {lct_toks:>14,} tokens ({100*lct_toks/(s2_toks+lct_toks):.1f}%)")

    print(f"\n  Sources: {n_sources:,}  |  Time: {elapsed:.1f}s")
    print("=" * 70)

    stats_path = parquet_dir / "dry_run_shift_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"shift-by-one dry run\ntotal_seqs={total_seqs}\ntotal_tokens={total_toks}\n")
    print(f"\nStats saved to {stats_path}")


def _compute_per_run_stats_shift(run_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rl = run_lengths.astype(np.int64)
    single = rl == 1
    il_per_run = np.where(
        single,
        0,
        np.maximum(1, rl // 2) + np.maximum(1, (rl - 1) // 2),
    )
    tr_per_run = np.where(single, 1, (rl % 2) + ((rl - 1) % 2))
    return il_per_run, tr_per_run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert interleaved parquet tokens to Megatron indexed datasets (shift-by-one mode)."
    )
    parser.add_argument("--parquet-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--max-seq-len", type=int, default=262144)
    parser.add_argument("--seq-threshold", type=int, default=None,
                        help="Route sequences <= threshold to stage2/, longer to lct/.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--transcribe-ratio", type=float, default=None)
    parser.add_argument("--tmp-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    output_dir = Path(args.output_dir)

    bos_id, eos_id, stt_continue_id, stt_transcribe_id, tts_continue_id, vocab_size = (
        load_token_ids(args.tokenizer_path)
    )
    dtype = DType.optimal_dtype(vocab_size)

    partition_dirs = list_interleave_cache_partitions(parquet_dir)

    if args.dry_run:
        if len(partition_dirs) > 1:
            raise RuntimeError(
                "Dry-run on a partitioned v2 cache root is not supported yet. "
                "Pass a leaf partition directory instead."
            )
        df, _cache_reader = load_interleave_cache(partition_dirs[0])
        _dry_run_shift(
            df, args.max_seq_len, bos_id, eos_id,
            stt_continue_id, stt_transcribe_id, tts_continue_id,
            dtype, partition_dirs[0],
            transcribe_ratio=args.transcribe_ratio,
            seq_threshold=args.seq_threshold,
        )
        return

    # Full build
    global _shared_cache, _shared_run_starts, _shared_run_lengths, _shared_transcribe_only_runs

    transcribe_only_runs_by_partition: dict[Path, set[int]] = {}
    runs_converted_to_transcribe = 0
    if args.transcribe_ratio is not None:
        all_il_per_run = []
        all_tr_per_run = []
        all_run_lengths = []
        partition_run_counts = []
        for partition_dir in partition_dirs:
            df, _cache_reader = load_interleave_cache(partition_dir)
            length_df = prepare_length_metadata(df)
            _sorted_df, _run_starts, run_lengths = _detect_runs(length_df)
            il_per_run, tr_per_run = _compute_per_run_stats_shift(run_lengths)
            all_il_per_run.append(il_per_run)
            all_tr_per_run.append(tr_per_run)
            all_run_lengths.append(run_lengths.astype(np.int64))
            partition_run_counts.append(len(run_lengths))
        selected_global_runs = compute_ratio_adjustment(
            np.concatenate(all_il_per_run) if all_il_per_run else np.array([], dtype=np.int64),
            np.concatenate(all_tr_per_run) if all_tr_per_run else np.array([], dtype=np.int64),
            np.concatenate(all_run_lengths) if all_run_lengths else np.array([], dtype=np.int64),
            args.transcribe_ratio,
        )
        runs_converted_to_transcribe = len(selected_global_runs)
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

    all_keys = OFFSET_KEYS + [TR_KEY]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.seq_threshold is not None:
        (output_dir / "stage2").mkdir(parents=True, exist_ok=True)
        (output_dir / "lct").mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers or max(1, multiprocessing.cpu_count() - 2)
    total_clips = 0
    total_sources = 0
    partition_stats = []
    worker_results = []
    if args.tmp_dir:
        tmp_dir = Path(args.tmp_dir) / f"_shift_shards_{os.getpid()}"
    else:
        tmp_dir = output_dir / "_tmp_shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
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
            partition_stats.append(
                {
                    "name": partition_dir.name,
                    "clips": n_clips,
                    "runs": n_runs,
                    "sources": n_sources,
                    "audio_tokens": audio_token_total,
                    "text_tokens": text_token_total,
                    "workers": len(run_ranges),
                }
            )
            if not run_ranges:
                _shared_cache = None
                _shared_run_starts = None
                _shared_run_lengths = None
                _shared_transcribe_only_runs = set()
                continue

            part_tmp_dir = tmp_dir / f"part_{part_idx:04d}_{partition_dir.name}"
            part_tmp_dir.mkdir(parents=True, exist_ok=True)
            worker_args = [
                (wid, rng[0], rng[1], all_keys, args.max_seq_len,
                 bos_id, eos_id, stt_continue_id, stt_transcribe_id, tts_continue_id,
                 dtype, str(part_tmp_dir), args.seq_threshold)
                for wid, rng in enumerate(run_ranges)
            ]
            # Pool creation must stay inside the partition loop, after the
            # current partition's globals are assigned. fork() captures these
            # globals for worker read-only access.
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(len(run_ranges)) as pool:
                worker_results.extend(pool.starmap(_shift_run_chunk, worker_args))

            _shared_cache = None
            _shared_run_starts = None
            _shared_run_lengths = None
            _shared_transcribe_only_runs = set()
            del cache, df, cache_reader
        print(f"\nWorkers finished in {time.time() - t0:.1f}s")

        if args.seq_threshold is not None:
            merge_keys = [f"{b}/{k}" for b in ("stage2", "lct") for k in all_keys]
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

    partition_summary = print_partition_stats(partition_stats)
    for key in sorted(counters.keys()):
        c = counters[key]
        print(f"  {key}: {c['seqs']:,} sequences, {c['tokens']:,} tokens")

    metadata = {
        "mode": "shift_by_one",
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
        "seq_threshold": args.seq_threshold,
        "transcribe_ratio": args.transcribe_ratio,
        "runs_converted_to_transcribe": runs_converted_to_transcribe,
        "total_clips": total_clips,
        "total_sources": total_sources,
        "partition_summary": partition_summary,
        "partition_stats": partition_stats,
        "outputs": {
            key: {"sequences": counters[key]["seqs"], "tokens": counters[key]["tokens"]}
            for key in sorted(counters.keys())
        },
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        import json

        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {metadata_path}")


if __name__ == "__main__":
    main()
