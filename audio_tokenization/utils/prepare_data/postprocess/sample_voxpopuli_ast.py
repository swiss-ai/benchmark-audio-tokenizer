#!/usr/bin/env python3
"""Sample 1 language per session from VoxPopuli AST to avoid duplicated English text.

VoxPopuli sessions are shared across ~22 languages. For AST (translation to English),
each session produces near-identical English text across all languages. This script
picks 1 random language per session and creates a new SHAR with only the selected cuts.

Audio tars are rebuilt (not symlinked) to contain only kept cuts, ensuring 1:1
positional correspondence with the filtered cuts JSONL.

Usage:
    python -m audio_tokenization.utils.prepare_data.postprocess.sample_voxpopuli_ast \
        --src-dir /capstor/.../voxpopuli_ast \
        --dst-dir /capstor/.../voxpopuli_ast_sampled \
        --seed 42 \
        --workers 64
"""

import argparse
import gzip
import logging
import os
import random
import shutil
import tarfile
import time
from collections import defaultdict
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import orjson

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_session(source_id: str) -> str:
    """Extract session from source_id by stripping the language suffix."""
    last_underscore = source_id.rfind("_")
    if last_underscore > 0:
        return source_id[:last_underscore]
    return source_id


def _scan_lang(args: tuple) -> dict:
    """Scan one language's SHAR to find all sessions."""
    lang_dir, = args
    lang = lang_dir.name
    sessions = set()
    cuts_files = sorted(lang_dir.glob("**/cuts.*.jsonl.gz"))
    for cf in cuts_files:
        with gzip.open(str(cf), "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = orjson.loads(line)
                source_id = d.get("custom", {}).get("source_id", "")
                if source_id:
                    sessions.add(_extract_session(source_id))
    return {"lang": lang, "sessions": sessions, "num_cuts_files": len(cuts_files)}


def _filter_shard(args: tuple):
    """Filter one shard: rewrite cuts JSONL and rebuild audio tar with only kept cuts."""
    src_cuts, src_tar, dst_cuts, dst_tar, kept_sessions = args

    # Read all cuts
    all_cuts = []
    with gzip.open(str(src_cuts), "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                all_cuts.append(orjson.loads(line))

    # Determine which indices to keep
    keep_indices = set()
    for i, d in enumerate(all_cuts):
        source_id = d.get("custom", {}).get("source_id", "")
        session = _extract_session(source_id)
        if session in kept_sessions:
            keep_indices.add(i)

    if not keep_indices:
        return 0, len(all_cuts)

    # Write filtered cuts
    dst_cuts.parent.mkdir(parents=True, exist_ok=True)
    tmp_cuts = Path(f"{dst_cuts}.tmp.{os.getpid()}")
    with gzip.open(str(tmp_cuts), "wb") as f:
        for i in sorted(keep_indices):
            f.write(orjson.dumps(all_cuts[i], option=orjson.OPT_APPEND_NEWLINE))
    tmp_cuts.rename(dst_cuts)

    # Rebuild audio tar with only kept entries
    if src_tar.exists():
        tmp_tar = Path(f"{dst_tar}.tmp.{os.getpid()}")
        dst_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(src_tar), "r") as src_tf:
            members = src_tf.getmembers()
            # Audio tar has 2 entries per cut: {id}.flac and {id}.json
            # Entries are ordered: cut0.flac, cut0.json, cut1.flac, cut1.json, ...
            kept_ids = set()
            for i in sorted(keep_indices):
                kept_ids.add(all_cuts[i]["id"])

            with tarfile.open(str(tmp_tar), "w") as dst_tf:
                for m in members:
                    # Strip extension to get cut ID
                    base = m.name.rsplit(".", 1)[0] if "." in m.name else m.name
                    if base in kept_ids:
                        data = src_tf.extractfile(m)
                        if data is not None:
                            dst_tf.addfile(m, data)
                        else:
                            dst_tf.addfile(m)
        tmp_tar.rename(dst_tar)

    return len(keep_indices), len(all_cuts)


def main():
    parser = argparse.ArgumentParser(
        description="Sample 1 language per session from VoxPopuli AST"
    )
    parser.add_argument("--src-dir", type=Path, required=True,
                        help="Source VoxPopuli AST SHAR dir (with per-language subdirs)")
    parser.add_argument("--dst-dir", type=Path, required=True,
                        help="Output sampled SHAR dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute and print sampling stats, don't write anything")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. Scan all languages to find sessions
    lang_dirs = sorted([d for d in args.src_dir.iterdir() if d.is_dir()])
    logger.info(f"Scanning {len(lang_dirs)} languages...")
    t0 = time.time()

    work = [(d,) for d in lang_dirs]
    with Pool(min(args.workers, len(work))) as pool:
        results = list(pool.imap_unordered(_scan_lang, work))

    # 2. Build session -> available languages mapping
    session_langs: dict[str, list[str]] = defaultdict(list)
    for r in results:
        lang = r["lang"]
        for session in r["sessions"]:
            session_langs[session].append(lang)

    logger.info(
        f"Found {len(session_langs):,} unique sessions across "
        f"{len(lang_dirs)} languages in {time.time() - t0:.1f}s"
    )

    # 3. For each session, pick 1 language using balanced round-robin.
    lang_assignment_count: dict[str, int] = defaultdict(int)
    session_to_lang: dict[str, str] = {}
    for session in sorted(session_langs.keys()):
        available = sorted(session_langs[session])
        rng.shuffle(available)
        chosen = min(available, key=lambda l: lang_assignment_count[l])
        session_to_lang[session] = chosen
        lang_assignment_count[chosen] += 1

    # Stats
    lang_counts = defaultdict(int)
    for lang in session_to_lang.values():
        lang_counts[lang] += 1
    logger.info("Sessions per language after sampling:")
    for lang in sorted(lang_counts):
        logger.info(f"  {lang}: {lang_counts[lang]}")

    if args.dry_run:
        logger.info("Dry run — no files written.")
        return

    # 4. For each language, determine which sessions to keep
    lang_kept_sessions: dict[str, set[str]] = defaultdict(set)
    for session, lang in session_to_lang.items():
        lang_kept_sessions[lang].add(session)

    # 5. Build work list: filter cuts + rebuild tars per shard
    t1 = time.time()
    filter_work = []
    for lang_dir in lang_dirs:
        lang = lang_dir.name
        kept = lang_kept_sessions.get(lang, set())
        if not kept:
            logger.info(f"  {lang}: no sessions selected, skipping")
            continue

        for cf in sorted(lang_dir.glob("**/cuts.*.jsonl.gz")):
            # cuts.000003.jsonl.gz -> recording.000003.tar
            shard_idx = cf.name.split(".")[1]
            src_tar = cf.parent / f"recording.{shard_idx}.tar"
            rel_cf = cf.relative_to(args.src_dir)
            rel_tar = src_tar.relative_to(args.src_dir)
            dst_cf = args.dst_dir / rel_cf
            dst_tar = args.dst_dir / rel_tar
            filter_work.append((cf, src_tar, dst_cf, dst_tar, kept))

    logger.info(f"Filtering {len(filter_work)} shards (cuts + tars)...")
    total_kept = 0
    total_orig = 0
    with Pool(min(args.workers, len(filter_work))) as pool:
        for kept_n, orig_n in pool.imap_unordered(_filter_shard, filter_work):
            total_kept += kept_n
            total_orig += orig_n

    # 6. Generate shar_index.json per language (only listing shards that exist)
    for lang_dir in lang_dirs:
        lang = lang_dir.name
        if lang not in lang_kept_sessions:
            continue
        dst_lang = args.dst_dir / lang
        if not dst_lang.exists():
            continue

        # Find all cuts/recording files that were actually written
        cuts_files = sorted(dst_lang.glob("**/cuts.*.jsonl.gz"))
        rec_files = sorted(dst_lang.glob("**/recording.*.tar"))
        if not cuts_files:
            continue

        cuts_rel = [str(f.relative_to(dst_lang)) for f in cuts_files]
        rec_rel = [str(f.relative_to(dst_lang)) for f in rec_files]

        index = {"version": 1, "fields": {"cuts": cuts_rel, "recording": rec_rel}}
        idx_path = dst_lang / "shar_index.json"
        with open(idx_path, "wb") as f:
            f.write(orjson.dumps(index, option=orjson.OPT_INDENT_2))

        # Write _SUCCESS marker
        (dst_lang / "_SUCCESS").touch()

    logger.info(
        f"Done in {time.time() - t1:.1f}s: "
        f"{total_kept:,} kept / {total_orig:,} total ({100 * total_kept / max(total_orig, 1):.1f}%)"
    )


if __name__ == "__main__":
    main()
