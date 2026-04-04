#!/usr/bin/env python3
"""Clean GigaSpeech v1 (English) text, write sidecar jsonl.zst per parquet.

Cleaning:
  - <COMMA> → ,  <PERIOD> → .  <QUESTIONMARK> → ?  <EXCLAMATIONPOINT> → !
  - Entries with non-speech tags (<OTHER>, <MUSIC>, <NOISE>, <SIL>) → text=null
  - Lowercase
  - All non-audio columns from the parquet are preserved in the output
  - Optional LLM ITN via Qwen: "three thousand" → "3000"

Each GPU runs one independent process. SLURM handles distribution via
bash background jobs, each with a unique --rank / --world-size.

Usage:
    # Mechanical only (no GPU)
    python -m audio_tokenization.utils.prepare_data.preprocess.clean_gigaspeech \
        --parquet-dir /path/to/xl --output-dir /path/to/out

    # Single GPU with LLM ITN
    python -m audio_tokenization.utils.prepare_data.preprocess.clean_gigaspeech \
        --parquet-dir /path/to/xl --output-dir /path/to/out \
        --model /path/to/Qwen3-8B-Instruct --rank 0 --world-size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from audio_tokenization.utils.prepare_data.preprocess.normalizer.common import (
    itn_batch,
    load_llm,
    write_jsonl_zst,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mechanical cleaning
# ---------------------------------------------------------------------------

_TAG_MAP = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!",
}
_TAG_RE = re.compile("|".join(re.escape(k) for k in _TAG_MAP))
_SKIP_TAGS = frozenset({"<OTHER>", "<MUSIC>", "<NOISE>", "<SIL>"})
_PUNCT_SPACE_RE = re.compile(r"\s+([.,?!])")


def clean_text(text: str) -> str | None:
    """Tag replacement + lowercase. Returns None for non-speech entries."""
    text = text.strip()
    if not text:
        return None
    for tag in _SKIP_TAGS:
        if tag in text:
            return None
    text = _TAG_RE.sub(lambda m: _TAG_MAP[m.group()], text)
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    return text.lower()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Clean GigaSpeech v1 text")
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parquet-glob", type=str, default="train-*.parquet")
    parser.add_argument("--id-column", type=str, default="segment_id")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--audio-column", type=str, default="audio")
    # LLM ITN
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    # Sharding (set by SLURM bash wrapper)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    import polars as pl

    pqs = sorted(args.parquet_dir.glob(args.parquet_glob))
    if not pqs:
        raise FileNotFoundError(f"No files match {args.parquet_dir / args.parquet_glob}")

    # Partition: each rank takes its slice
    pqs = [p for i, p in enumerate(pqs) if i % args.world_size == args.rank]

    if not pqs:
        logger.info("Rank %d/%d: nothing to do", args.rank, args.world_size)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    llm = tokenizer = sampling_params = None
    if args.model:
        logger.info("Rank %d/%d: loading model %s", args.rank, args.world_size, args.model)
        llm, tokenizer, sampling_params = load_llm(
            args.model, args.temperature, args.max_tokens,
        )

    logger.info(
        "Rank %d/%d: processing %d parquets (model=%s)",
        args.rank, args.world_size, len(pqs), args.model or "none",
    )

    skip_columns = {args.audio_column}
    total_kept = total_skipped = 0

    for i, pq in enumerate(pqs):
        t0 = time.time()
        all_columns = pl.read_parquet_schema(pq).names()
        read_columns = [c for c in all_columns if c not in skip_columns]
        df = pl.read_parquet(pq, columns=read_columns)

        entries = []
        for row in df.iter_rows(named=True):
            rid = str(row[args.id_column])
            raw = row.get(args.text_column) or ""
            entry = {k: v for k, v in row.items()}
            entry["id"] = rid
            entry["text"] = clean_text(raw)
            if llm is not None:
                entry["text_norm"] = None
            entries.append(entry)

        if llm is not None:
            to_norm = [(j, e["text"]) for j, e in enumerate(entries) if e["text"] is not None]
            for start in range(0, len(to_norm), args.batch_size):
                batch = to_norm[start : start + args.batch_size]
                indices, texts = zip(*batch)
                normed = itn_batch(llm, tokenizer, sampling_params, list(texts))
                for idx, n in zip(indices, normed):
                    entries[idx]["text_norm"] = n

        out_path = args.output_dir / (pq.stem + ".jsonl.zst")
        write_jsonl_zst(entries, out_path)

        kept = sum(1 for e in entries if e["text"] is not None)
        skipped = len(entries) - kept
        total_kept += kept
        total_skipped += skipped
        elapsed = time.time() - t0
        logger.info(
            "Rank %d/%d [%d/%d] %s: %d kept, %d skipped (%.1fs)",
            args.rank, args.world_size, i + 1, len(pqs),
            pq.name, kept, skipped, elapsed,
        )

    logger.info(
        "Rank %d/%d done: %d kept, %d skipped",
        args.rank, args.world_size, total_kept, total_skipped,
    )


if __name__ == "__main__":
    main()
