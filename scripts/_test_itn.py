#!/usr/bin/env python3
"""Test ITN on one GigaSpeech parquet using 4 GPUs (data parallel)."""

import os
import time
from collections import Counter
from multiprocessing import Process, Queue
from queue import Empty


def gpu_worker(gpu_id, texts_chunk, indices_chunk, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from audio_tokenization.prepare.preprocess.clean_gigaspeech import _itn_batch

    MODEL = "/capstor/store/cscs/swissai/infra01/MLLM/model_baseline/Qwen3-4B-Instruct-2507"
    BATCH_SIZE = 512

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    llm = LLM(model=MODEL, trust_remote_code=True, tensor_parallel_size=1, max_model_len=1024, dtype="auto")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, stop=["\n"])

    results = {}
    for start in range(0, len(texts_chunk), BATCH_SIZE):
        batch_texts = texts_chunk[start : start + BATCH_SIZE]
        batch_indices = indices_chunk[start : start + BATCH_SIZE]
        normed = _itn_batch(llm, tokenizer, sampling_params, batch_texts)
        for idx, n in zip(batch_indices, normed):
            results[idx] = n
        done = min(start + BATCH_SIZE, len(texts_chunk))
        print(f"  GPU {gpu_id}: {done}/{len(texts_chunk)}")

    result_queue.put(results)


def _await_worker_result(proc, result_queue, *, poll_timeout_s=5.0):
    def _join(timeout=None):
        try:
            if timeout is None:
                proc.join()
            else:
                proc.join(timeout=timeout)
        except TypeError:
            proc.join()

    while True:
        try:
            results = result_queue.get(timeout=poll_timeout_s)
            # Bound joins: a wedged queue-feeder thread shouldn't hang the
            # parent. terminate() if the worker doesn't exit cleanly.
            _join(timeout=poll_timeout_s)
            if proc.is_alive():
                terminate = getattr(proc, "terminate", None)
                if terminate is not None:
                    terminate()
                _join()
                raise RuntimeError(
                    f"Worker pid={proc.pid} returned results but did not "
                    f"exit within {poll_timeout_s}s; killed."
                )
            if proc.exitcode not in (None, 0):
                raise RuntimeError(
                    f"Worker pid={proc.pid} exited with code {proc.exitcode} "
                    "after writing results."
                )
            return results
        except Empty:
            if not proc.is_alive():
                _join(timeout=poll_timeout_s)
                try:
                    get_nowait = getattr(result_queue, "get_nowait", None)
                    if get_nowait is not None:
                        results = get_nowait()
                    else:
                        results = result_queue.get(timeout=0)
                except Empty:
                    pass
                else:
                    if proc.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"Worker pid={proc.pid} exited with code {proc.exitcode} "
                            "after writing results."
                        )
                    return results
                raise RuntimeError(
                    f"Worker pid={proc.pid} exited with code {proc.exitcode} "
                    "before writing results."
                )


def main():
    import polars as pl

    from audio_tokenization.prepare.preprocess.clean_gigaspeech import clean_text

    PQ = "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___speechcolab___gigaspeech/parquet-data/xl/train-00000-of-00258.parquet"
    NUM_GPUS = 4

    print("Reading parquet...")
    df = pl.read_parquet(PQ, columns=["segment_id", "text"])
    print(f"Rows: {df.height}")

    entries = []
    for row in df.iter_rows(named=True):
        rid = str(row["segment_id"])
        raw = row.get("text") or ""
        entries.append({"id": rid, "text": clean_text(raw), "text_norm": None})

    kept = sum(1 for e in entries if e["text"] is not None)
    skipped = len(entries) - kept
    print(f"Mechanical: {kept} kept, {skipped} skipped")

    # Split work across GPUs
    to_norm = [(i, e["text"]) for i, e in enumerate(entries) if e["text"] is not None]
    chunk_size = (len(to_norm) + NUM_GPUS - 1) // NUM_GPUS

    print(f"Running ITN on {len(to_norm)} entries across {NUM_GPUS} GPUs...")
    t0 = time.time()

    queues = []
    procs = []
    for gpu_id in range(NUM_GPUS):
        chunk = to_norm[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]
        if not chunk:
            continue
        indices = [c[0] for c in chunk]
        texts = [c[1] for c in chunk]
        q = Queue()
        queues.append(q)
        p = Process(target=gpu_worker, args=(gpu_id, texts, indices, q))
        procs.append(p)
        p.start()

    for p, q in zip(procs, queues):
        results = _await_worker_result(p, q)
        for idx, norm in results.items():
            entries[idx]["text_norm"] = norm

    total_time = time.time() - t0
    print(f"\nITN done in {total_time:.1f}s ({len(to_norm) / total_time:.0f} samples/s)")

    # Diff metrics
    n_changed = n_unchanged = 0
    n_shorter = n_longer = 0
    total_char_diff = 0
    word_changes = Counter()

    for e in entries:
        if e["text"] is None or e["text_norm"] is None:
            continue
        if e["text"] == e["text_norm"]:
            n_unchanged += 1
            continue
        n_changed += 1
        total_char_diff += abs(len(e["text_norm"]) - len(e["text"]))
        if len(e["text_norm"]) < len(e["text"]):
            n_shorter += 1
        else:
            n_longer += 1
        for b, a in zip(e["text"].split(), e["text_norm"].split()):
            if b != a:
                word_changes[f"{b} → {a}"] += 1

    total = n_changed + n_unchanged
    print(f"\n{'='*60}")
    print(f"Changed: {n_changed}/{total} ({100*n_changed/total:.1f}%)")
    print(f"Shorter after ITN: {n_shorter}, Longer: {n_longer}")
    if n_changed:
        print(f"Avg char diff: {total_char_diff/n_changed:.1f}")
    print(f"\nTop 15 word changes:")
    for c, n in word_changes.most_common(15):
        print(f"  {n:5d}x  {c}")
    print(f"\nSample diffs:")
    shown = 0
    for e in entries:
        if e["text"] and e["text_norm"] and e["text"] != e["text_norm"]:
            print(f"  B: {e['text'][:120]}")
            print(f"  A: {e['text_norm'][:120]}")
            print()
            shown += 1
            if shown >= 10:
                break


if __name__ == "__main__":
    main()
