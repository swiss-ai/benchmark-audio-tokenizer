#!/usr/bin/env python3
"""Run ITN on 1 parquet with 4 GPUs, save all results, show biggest diffs."""

import json
import tempfile
import time
from multiprocessing import Process
from pathlib import Path


def gpu_worker(gpu_id, texts_chunk, indices_chunk, out_path):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from audio_tokenization.prepare.preprocess.clean_gigaspeech import _itn_batch

    MODEL = "/capstor/store/cscs/swissai/infra01/MLLM/model_baseline/Qwen3-8B-Instruct"
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
        print(f"  GPU {gpu_id}: {min(start+BATCH_SIZE, len(texts_chunk))}/{len(texts_chunk)}")

    with open(out_path, "w") as f:
        json.dump(results, f)


def main():
    import polars as pl
    from audio_tokenization.prepare.preprocess.clean_gigaspeech import clean_text

    PQ_DIR = "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___speechcolab___gigaspeech/parquet-data/xl"
    OUTPUT_DIR = "/tmp/gs_test_2pq"
    SAVE_PATH = "/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/scripts/_itn_results.jsonl"
    NUM_GPUS = 4

    pq_files = sorted(Path(PQ_DIR).glob("*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"No parquet files in {PQ_DIR}")
    PQ = str(pq_files[0])
    print(f"Reading parquet {PQ}...")
    df = pl.read_parquet(PQ, columns=["segment_id", "text"])

    entries = []
    for row in df.iter_rows(named=True):
        rid = str(row["segment_id"])
        raw = row.get("text") or ""
        entries.append({"id": rid, "text": clean_text(raw), "text_norm": None})

    to_norm = [(i, e["text"]) for i, e in enumerate(entries) if e["text"] is not None]
    chunk_size = (len(to_norm) + NUM_GPUS - 1) // NUM_GPUS
    print(f"Running ITN on {len(to_norm)} entries across {NUM_GPUS} GPUs...")
    t0 = time.time()

    tmpdir = tempfile.mkdtemp()
    procs = []
    out_paths = []
    for gpu_id in range(NUM_GPUS):
        chunk = to_norm[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]
        if not chunk:
            continue
        indices = [c[0] for c in chunk]
        texts = [c[1] for c in chunk]
        out_path = f"{tmpdir}/gpu_{gpu_id}.json"
        out_paths.append(out_path)
        p = Process(target=gpu_worker, args=(gpu_id, texts, indices, out_path))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    for out_path in out_paths:
        with open(out_path) as f:
            results = json.load(f)
        for idx_str, norm in results.items():
            entries[int(idx_str)]["text_norm"] = norm

    total_time = time.time() - t0
    print(f"\nITN done in {total_time:.1f}s ({len(to_norm) / total_time:.0f} samples/s)")

    # Save all results
    with open(SAVE_PATH, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Saved all {len(entries)} entries to {SAVE_PATH}")

    # Show biggest diffs
    diffs = []
    for e in entries:
        if e["text"] and e["text_norm"] and e["text"] != e["text_norm"]:
            diffs.append((abs(len(e["text"]) - len(e["text_norm"])), e))

    diffs.sort(key=lambda x: x[0], reverse=True)

    n_changed = len(diffs)
    n_total = sum(1 for e in entries if e["text"] is not None)
    print(f"\nChanged: {n_changed}/{n_total} ({100*n_changed/n_total:.1f}%)")

    print(f"\nTop 30 biggest diffs:")
    print("=" * 70)
    for char_diff, e in diffs[:30]:
        print(f"\n[{e['id']}] char_diff={char_diff}")
        print(f"  B: {e['text'][:200]}")
        print(f"  A: {e['text_norm'][:200]}")


if __name__ == "__main__":
    main()
