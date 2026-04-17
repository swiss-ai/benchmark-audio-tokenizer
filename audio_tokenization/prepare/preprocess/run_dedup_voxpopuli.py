#!/usr/bin/env python3
"""Deduplicate VoxPopuli per_lang_year VAD JSONL files.

Groups {lang}_{year}.jsonl files by language prefix, then runs a three-level
dedup pipeline:
  1. Local dedup per file (group by duration+timestamps, verify with audio MD5)
  2. Global dedup across all years within the same language
  3. Cross-language dedup across all languages

Usage:
    python -m audio_tokenization.prepare.preprocess.run_dedup_voxpopuli \
        --audio_dir /capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/raw_audios \
        --input /capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/vad_results_merged/per_lang_year

Output: {lang}_{year}.jsonl in --output directory (default: {input}_dedup).
"""

import argparse
import logging
import multiprocessing as mp
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import orjson

from audio_tokenization.prepare.runtime import audio_md5, build_audio_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core dedup logic
# ---------------------------------------------------------------------------

def _load_entries(input_paths: List[Path]) -> List[Tuple[Path, str, dict, bytes]]:
    """Read all entries from JSONL files. Returns (source_path, key, info, line_bytes)."""
    entries = []
    for input_path in input_paths:
        with open(input_path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                d = orjson.loads(line)
                key = next(iter(d))
                entries.append((input_path, key, d[key], line.strip()))
    return entries


def _dedup(
    entries: List[Tuple[Path, str, dict, bytes]],
    audio_index: Dict[str, str],
) -> Tuple[Set[int], int, int, float]:
    """Run two-step dedup. Returns (indices_to_remove, empty, confirmed, hours_removed)."""

    # Step 1: group by (duration, timestamps) — zero I/O
    groups: Dict[tuple, List[int]] = defaultdict(list)
    empty = 0
    for idx, (_, key, info, _) in enumerate(entries):
        ts = info.get("timestamps", [])
        if not ts:
            empty += 1
            continue
        dur = round(info.get("duration_sec", 0), 3)
        dedup_key = (dur, tuple(tuple(t) for t in ts))
        groups[dedup_key].append(idx)

    candidate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    num_candidates = sum(len(v) for v in candidate_groups.values())
    logger.info(
        f"Step 1 (timestamps): {len(groups):,} unique, "
        f"{len(candidate_groups):,} candidate groups ({num_candidates:,} entries), "
        f"{empty:,} empty skipped"
    )

    # Step 2: MD5 verify candidates — reads only duplicate candidates
    to_remove: Set[int] = set()
    confirmed = 0
    for dedup_key, indices in candidate_groups.items():
        md5_groups: Dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            key = entries[idx][1]
            audio_path = audio_index.get(key.lower())
            if audio_path is None:
                continue
            md5 = audio_md5(audio_path)
            md5_groups[md5].append(idx)

        for md5, md5_indices in md5_groups.items():
            if len(md5_indices) > 1:
                confirmed += len(md5_indices) - 1
                to_remove.update(md5_indices[1:])

    hours_removed = sum(entries[idx][2].get("duration_sec", 0) for idx in to_remove) / 3600
    logger.info(f"Step 2 (MD5 verify): {confirmed:,} confirmed duplicates ({hours_removed:.2f}h)")
    return to_remove, empty, confirmed, hours_removed


def _write_outputs(
    entries: List[Tuple[Path, str, dict, bytes]],
    to_remove: Set[int],
    output_dir: Path,
):
    """Write per-input deduped JSONL files into *output_dir*.

    Each input file ``{lang}_{year}.jsonl`` produces
    ``output_dir/{lang}_{year}.jsonl`` (same basename, different directory).
    Skips empty-timestamp entries.
    """
    kept_by_input: Dict[Path, List[bytes]] = defaultdict(list)
    for idx, (input_path, _, info, line_bytes) in enumerate(entries):
        if idx in to_remove:
            continue
        if not info.get("timestamps", []):
            continue
        kept_by_input[input_path].append(line_bytes)

    for input_path, lines in kept_by_input.items():
        output = output_dir / input_path.name
        with open(output, "wb") as out:
            for line_bytes in lines:
                out.write(line_bytes + b"\n")
        logger.info(f"  {output.name}: {len(lines):,} entries")

    return sum(len(v) for v in kept_by_input.values())


def _dedup_group(
    input_paths: List[Path],
    audio_index: Dict[str, str],
    output_dir: Path,
) -> None:
    """Run local+global dedup on a group of JSONL files.

    Single file: local dedup only.
    Multiple files: local dedup each, then global dedup across all.
    Results are written to *output_dir*.
    """
    if len(input_paths) == 1:
        logger.info(f"Local dedup: {input_paths[0].name}")
        entries = _load_entries(input_paths)
        to_remove, empty, confirmed, hours_removed = _dedup(entries, audio_index)
        total_kept = _write_outputs(entries, to_remove, output_dir)
        logger.info(
            f"  {total_kept:,} kept, {confirmed:,} duplicates removed "
            f"({hours_removed:.2f}h), {empty:,} empty skipped"
        )
        return

    with tempfile.TemporaryDirectory(prefix="dedup_") as tmpdir:
        tmpdir = Path(tmpdir)
        local_output_paths = []

        for input_path in input_paths:
            logger.info(f"Local dedup: {input_path.name}")
            entries = _load_entries([input_path])
            to_remove, empty, confirmed, hours_removed = _dedup(entries, audio_index)
            tmp_out = tmpdir / f"{input_path.stem}_local.jsonl"
            kept = 0
            with open(tmp_out, "wb") as out:
                for idx, (_, _, info, line_bytes) in enumerate(entries):
                    if idx in to_remove or not info.get("timestamps", []):
                        continue
                    out.write(line_bytes + b"\n")
                    kept += 1
            logger.info(
                f"  {kept:,} kept, {confirmed:,} duplicates removed "
                f"({hours_removed:.2f}h), {empty:,} empty skipped"
            )
            local_output_paths.append(tmp_out)

        logger.info(f"Global dedup across {len(local_output_paths)} locally-deduped files")
        entries = _load_entries(local_output_paths)
        tmp_to_orig = {
            tmpdir / f"{p.stem}_local.jsonl": p for p in input_paths
        }
        entries = [
            (tmp_to_orig.get(src, src), key, info, line_bytes)
            for src, key, info, line_bytes in entries
        ]
        logger.info(f"Loaded {len(entries):,} entries")
        to_remove, empty, confirmed, hours_removed = _dedup(entries, audio_index)
        total_kept = _write_outputs(entries, to_remove, output_dir)
        logger.info(
            f"  {total_kept:,} kept, {confirmed:,} duplicates removed "
            f"({hours_removed:.2f}h), {empty:,} empty skipped"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _group_by_lang_prefix(paths: List[Path]) -> Dict[str, List[Path]]:
    """Group JSONL files by language prefix (e.g. en_2009.jsonl -> 'en')."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        lang = p.stem.split("_")[0]
        groups[lang].append(p)
    return dict(groups)


def _dedup_group_worker(args_tuple):
    """Multiprocessing wrapper for _dedup_group."""
    lang, lang_paths, audio_index, output_dir = args_tuple
    logger.info(f"Dedup group: {lang} ({len(lang_paths)} files)")
    _dedup_group(lang_paths, audio_index, output_dir)
    return lang


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate VoxPopuli VAD JSONL files per language",
    )
    parser.add_argument("--audio_dir", type=Path, required=True,
                        help="Root audio directory (indexed recursively via --pattern)")
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory of per_lang_year JSONL files")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for deduped JSONL files "
                             "(default: {input}_dedup)")
    parser.add_argument("--pattern", default="**/*.ogg")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.input}")

    output_dir = args.output or args.input.with_name(args.input.name + "_dedup")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    input_paths = sorted(args.input.glob("*.jsonl"))
    if not input_paths:
        raise FileNotFoundError(f"No *.jsonl files in {args.input}")
    logger.info(f"Found {len(input_paths)} JSONL file(s)")

    audio_index = build_audio_index(args.audio_dir, args.pattern)
    logger.info(f"Indexed {len(audio_index):,} audio files from {args.audio_dir}")

    groups = _group_by_lang_prefix(input_paths)
    logger.info(f"Grouped into {len(groups)} language(s): {sorted(groups.keys())}")

    worker_args = [
        (lang, lang_paths, audio_index, output_dir)
        for lang, lang_paths in sorted(groups.items())
    ]

    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=len(worker_args)) as pool:
        finished = pool.map(_dedup_group_worker, worker_args)

    logger.info(f"Per-language dedup done for {len(finished)} languages.")

    # Cross-language dedup pass: load all deduped files, run global dedup
    logger.info("Cross-language dedup pass...")
    deduped_paths = sorted(output_dir.glob("*.jsonl"))
    entries = _load_entries(deduped_paths)
    logger.info(f"Loaded {len(entries):,} entries across all languages")
    to_remove, empty, confirmed, hours_removed = _dedup(entries, audio_index)
    if confirmed > 0:
        total_kept = _write_outputs(entries, to_remove, output_dir)
        logger.info(
            f"Cross-language: {total_kept:,} kept, {confirmed:,} duplicates removed "
            f"({hours_removed:.2f}h)"
        )
    else:
        logger.info("Cross-language: no duplicates found")

    logger.info("All done.")


if __name__ == "__main__":
    main()
