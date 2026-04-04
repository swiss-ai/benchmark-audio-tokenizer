#!/usr/bin/env python3
"""Generic LLM text normalization for any audio dataset.

Reads (id, text) from parquet or Arrow files, normalizes via vLLM,
writes per-rank jsonl.zst shards. Merge shards with:
    cat shard_*.jsonl.zst > merged.jsonl.zst

Output per line:
    {"id": "...", "text": "<original>", "text_norm": "<normalized>"}

Usage:
    # People's Speech (Arrow)
    python -m audio_tokenization.utils.prepare_data.preprocess.normalizer.normalize_text \
        --input-dir /path/to/arrow-data --input-format arrow \
        --model /path/to/Qwen3-8B-Instruct \
        --output /path/to/peoples_speech_metadata.jsonl.zst \
        --rank 0 --world-size 64

    # Any parquet dataset
    python -m audio_tokenization.utils.prepare_data.preprocess.normalizer.normalize_text \
        --input-dir /path/to/parquets --input-format parquet \
        --input-glob 'train-*.parquet' --id-column segment_id \
        --model /path/to/Qwen3-8B-Instruct \
        --output /path/to/metadata.jsonl.zst
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .common import itn_batch, load_llm, write_jsonl_zst

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def _iter_parquet(input_dir: Path, glob: str, id_col: str, text_col: str):
    import polars as pl
    for pq in sorted(input_dir.glob(glob)):
        df = pl.read_parquet(pq, columns=[id_col, text_col])
        for row in df.iter_rows(named=True):
            yield str(row[id_col]), row.get(text_col) or ""


def _iter_arrow(input_dir: Path, glob: str, id_col: str, text_col: str):
    import pyarrow.ipc as ipc
    for arrow_file in sorted(input_dir.glob(glob)):
        reader = ipc.open_stream(arrow_file)
        table = reader.read_all()
        ids = table.column(id_col)
        texts = table.column(text_col)
        for i in range(table.num_rows):
            yield str(ids[i].as_py()), texts[i].as_py() or ""


_READERS = {
    "parquet": (_iter_parquet, "*.parquet"),
    "arrow": (_iter_arrow, "*.arrow"),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="LLM text normalization")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-format", choices=list(_READERS), required=True)
    parser.add_argument("--input-glob", type=str, default=None)
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--output", type=Path, required=True)
    # LLM
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    # Sharding
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    reader_fn, default_glob = _READERS[args.input_format]
    glob = args.input_glob or default_glob

    # Shard input files across ranks (not entries) to avoid reading all data
    all_files = sorted(args.input_dir.glob(glob))
    my_files = [f for i, f in enumerate(all_files) if i % args.world_size == args.rank]
    logger.info("Rank %d/%d: %d/%d files", args.rank, args.world_size, len(my_files), len(all_files))

    if not my_files:
        logger.info("Nothing to do")
        return

    # Build a glob that matches only this rank's files
    logger.info("Reading %s from %d files...", args.input_format, len(my_files))
    t0 = time.time()
    my_pairs = []
    for f in my_files:
        my_pairs.extend(reader_fn(f.parent, f.name, args.id_column, args.text_column))
    logger.info("Read %d entries in %.1fs", len(my_pairs), time.time() - t0)

    if not my_pairs:
        logger.info("Nothing to do")
        return

    entries = [{"id": rid, "text": text.strip() if text else None} for rid, text in my_pairs]
    del my_pairs

    # Load LLM and run ITN
    logger.info("Loading model: %s", args.model)
    llm, tokenizer, sampling_params = load_llm(
        args.model, args.temperature, args.max_tokens,
    )

    to_norm = [(i, e["text"]) for i, e in enumerate(entries) if e["text"]]
    logger.info("Running ITN on %d entries...", len(to_norm))
    t0 = time.time()

    for start in range(0, len(to_norm), args.batch_size):
        batch = to_norm[start : start + args.batch_size]
        indices, texts = zip(*batch)
        normed = itn_batch(llm, tokenizer, sampling_params, list(texts))
        for idx, n in zip(indices, normed):
            entries[idx]["text_norm"] = n
        done = min(start + args.batch_size, len(to_norm))
        elapsed = time.time() - t0
        logger.info("  %d/%d (%.0f samples/s)", done, len(to_norm),
                    done / elapsed if elapsed > 0 else 0)

    # Write per-rank shard
    if args.world_size > 1:
        base = args.output.name
        for ext in (".jsonl.zst", ".jsonl.gz", ".jsonl"):
            if base.endswith(ext):
                stem = base[: -len(ext)]
                out_path = args.output.parent / f"{stem}.{args.rank:04d}{ext}"
                break
        else:
            out_path = args.output.parent / f"{args.output.stem}.{args.rank:04d}{args.output.suffix}"
    else:
        out_path = args.output

    write_jsonl_zst(entries, out_path)

    changed = sum(1 for e in entries if e.get("text_norm") and e["text"] != e["text_norm"])
    logger.info("Rank %d/%d: wrote %d entries (%d changed) to %s",
                args.rank, args.world_size, len(entries), changed, out_path)


if __name__ == "__main__":
    main()
