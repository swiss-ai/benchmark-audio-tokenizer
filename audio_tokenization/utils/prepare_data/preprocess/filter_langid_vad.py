#!/usr/bin/env python3
"""Filter VAD results by language ID predictions.

Reads lang_id_results.jsonl and vad_results.jsonl, keeps only entries
whose language prediction is in the allowed set, and writes a filtered
VAD JSONL that can be used directly with vad_segmenting.py.

Usage:
    python -m audio_tokenization.utils.prepare_data.preprocess.filter_langid_vad \
        --data_dir /path/to/unsupervised_peoples_speech_commercial_wds \
        --output-per-shard-dir /path/to/output/vad_per_shard \
        --languages european
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import orjson

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ISO 639-1 codes for European languages.
EUROPEAN_LANGUAGES: Set[str] = {
    # Western European
    "en", "fr", "de", "nl", "es", "pt", "it", "ca", "gl", "eu", "oc",
    # Northern European
    "sv", "da", "no", "nn", "fi", "is", "fo",
    # Eastern European
    "pl", "cs", "sk", "hu", "ro", "bg", "hr", "sr", "bs", "sl", "mk",
    "uk", "ru", "be", "lt", "lv", "et",
    # Other European
    "el", "sq", "mt", "cy", "br", "ka", "yi", "la",
}

# ISO 639-1 codes for Southeast Asian languages.
SOUTHEAST_ASIAN_LANGUAGES: Set[str] = {
    "vi",   # Vietnamese
    "th",   # Thai
    "id",   # Indonesian
    "ms",   # Malay
    "tl",   # Tagalog / Filipino
    "jw",   # Javanese (Whisper uses "jw")
    "km",   # Khmer
    "my",   # Burmese / Myanmar
}

# ISO 639-1 codes for African languages (matching Whisper lang_id output).
AFRICAN_LANGUAGES: Set[str] = {
    "sw",   # Swahili
    "yo",   # Yoruba
    "sn",   # Shona
    "so",   # Somali
    "am",   # Amharic
    "af",   # Afrikaans
}

# ISO 639-1 codes for East Asian languages.
EAST_ASIAN_LANGUAGES: Set[str] = {
    "zh",   # Chinese (Mandarin/Cantonese)
    "ja",   # Japanese
    "ko",   # Korean
    "mn",   # Mongolian
    "yue",  # Cantonese (explicit Whisper code)
}

LANGUAGE_PRESETS: Dict[str, Set[str]] = {
    "european": EUROPEAN_LANGUAGES,
    "southeast_asian": SOUTHEAST_ASIAN_LANGUAGES,
    "african": AFRICAN_LANGUAGES,
    "east_asian": EAST_ASIAN_LANGUAGES,
    "all": None,  # special: accept every language except nospeech
}


def _normalize_key(key: str) -> str:
    """Canonical sample key used for cross-file joins."""
    return key.strip().lower()


def _langid_key(filepath: str) -> str:
    """Convert lang_id filepath to VAD key format.

    Lang_id: /data/folder/file.mp3  ->  folder/file
    VAD key: folder/file
    """
    # Strip leading /data/ prefix
    path = filepath
    if path.startswith("/data/"):
        path = path[len("/data/"):]
    # Remove file extension
    path = os.path.splitext(path)[0]
    return _normalize_key(path)


def _parse_vad_entry(
    line: str,
) -> Optional[Tuple[str, str, Dict[str, Dict[str, object]]]]:
    """Parse one VAD JSONL entry and return (norm_key, raw_key, payload)."""
    try:
        entry = orjson.loads(line)
    except (orjson.JSONDecodeError, ValueError):
        return None

    if not isinstance(entry, dict) or len(entry) != 1:
        return None

    raw_key = next(iter(entry))
    value = entry[raw_key]
    if not isinstance(value, dict):
        return None
    return _normalize_key(raw_key), raw_key, entry


def _collect_metadata_key_to_shard(
    *, metadata_shards_dir: Path
) -> Tuple[Dict[str, str], Dict[str, Optional[float]], Dict[str, Optional[int]], int, int]:
    """Build key->shard_name, key->duration, and key->sample_rate mappings from metadata TSVs.

    Returns: (key_to_shard, key_to_duration, key_to_sample_rate, duplicate_key_count, scanned_tsv_count)
    """
    if not metadata_shards_dir.is_dir():
        raise NotADirectoryError(
            f"Metadata shard directory does not exist: {metadata_shards_dir}"
        )

    key_to_shard: Dict[str, str] = {}
    key_to_duration: Dict[str, Optional[float]] = {}
    key_to_sample_rate: Dict[str, Optional[int]] = {}
    duplicate_key_count = 0
    tsv_files = sorted(metadata_shards_dir.glob("*.tsv"))
    for tsv_path in tsv_files:
        shard_name = tsv_path.stem
        with open(tsv_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("error"):
                    continue
                key = row.get("sample_key", "")
                if not key:
                    continue
                key = _normalize_key(key)
                prev = key_to_shard.get(key)
                if prev is not None:
                    if prev != shard_name:
                        duplicate_key_count += 1
                    continue
                key_to_shard[key] = shard_name
                dur_str = row.get("duration", "")
                try:
                    key_to_duration[key] = float(dur_str) if dur_str else None
                except (TypeError, ValueError):
                    key_to_duration[key] = None
                sr_str = row.get("sample_rate", "")
                try:
                    key_to_sample_rate[key] = int(sr_str) if sr_str else None
                except (TypeError, ValueError):
                    key_to_sample_rate[key] = None

    return key_to_shard, key_to_duration, key_to_sample_rate, duplicate_key_count, len(tsv_files)


def _prepare_per_shard_output_dir(*, out_dir: Path, overwrite: bool) -> None:
    """Prepare output directory for per-shard JSONL writing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.jsonl"))
    if existing and not overwrite:
        raise RuntimeError(
            f"Output directory {out_dir} already has {len(existing)} .jsonl files. "
            "Pass --overwrite-output to replace them."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def _collect_lang_for_metadata_keys(
    *,
    lang_id_path: Path,
    metadata_keys: Set[str],
    allowed_langs: Optional[Set[str]],
) -> Tuple[Dict[str, str], int, int]:
    """Collect key->lang for keys present in selected metadata shards only.

    This avoids building a huge language map for unrelated shards.
    """
    key_to_lang: Dict[str, str] = {}
    total_lines = 0
    skipped_nospeech = 0

    with open(lang_id_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                entry = orjson.loads(line)
            except (orjson.JSONDecodeError, ValueError):
                continue

            prediction = entry.get("prediction", "")
            if prediction == "nospeech":
                skipped_nospeech += 1
                continue
            if allowed_langs is not None and prediction not in allowed_langs:
                continue

            filepath = entry.get("filepath")
            if not isinstance(filepath, str) or not filepath:
                continue
            key = _langid_key(filepath)
            if key not in metadata_keys:
                continue
            key_to_lang[key] = prediction

    return key_to_lang, total_lines, skipped_nospeech


def _build_target_lookup_with_polars(
    *,
    key_to_lang: Dict[str, str],
    key_to_shard: Dict[str, str],
    key_to_duration: Dict[str, Optional[float]],
    key_to_sample_rate: Dict[str, Optional[int]],
) -> Tuple[Dict[str, Tuple[str, str, Optional[float], Optional[int]]], pl.DataFrame]:
    """Build key->(lang, shard, duration, sample_rate) lookup with Polars-backed table creation."""
    keys = list(key_to_lang.keys())
    target_df = pl.DataFrame(
        {
            "key": keys,
            "lang": [key_to_lang[k] for k in keys],
            "shard": [key_to_shard[k] for k in keys],
        }
    )
    target_lookup = {
        row["key"]: (row["lang"], row["shard"], key_to_duration.get(row["key"]), key_to_sample_rate.get(row["key"]))
        for row in target_df.iter_rows(named=True)
    }
    return target_lookup, target_df


def _flush_shard_buffers(
    *, shard_to_lines: Dict[str, List[str]], output_per_shard_dir: Path
) -> int:
    """Flush buffered JSONL lines to per-shard output files."""
    written = 0
    for shard, lines in shard_to_lines.items():
        if not lines:
            continue
        out_path = output_per_shard_dir / f"{shard}.jsonl"
        with open(out_path, "a", encoding="utf-8") as fout:
            fout.writelines(lines)
        written += len(lines)
    shard_to_lines.clear()
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Filter VAD results by language ID predictions."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing lang_id_results.jsonl and vad_results.jsonl",
    )
    parser.add_argument(
        "--lang_id_file",
        type=str,
        default="lang_id_results.jsonl",
        help="Name of lang_id results file (default: lang_id_results.jsonl)",
    )
    parser.add_argument(
        "--vad_file",
        type=str,
        default="vad_results.jsonl",
        help="Name of VAD results file (default: vad_results.jsonl)",
    )
    parser.add_argument(
        "--output-per-shard-dir",
        type=Path,
        required=True,
        help=(
            "Output directory for per-shard VAD JSONL files, one file per "
            "metadata shard (e.g., audio_000001.jsonl)"
        ),
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="european",
        help=(
            "Comma-separated language codes, or a preset name: "
            "'european', 'southeast_asian', 'african', 'east_asian' "
            "(default: european)"
        ),
    )
    parser.add_argument(
        "--vad-batch-lines",
        type=int,
        default=200000,
        help="Buffered matched rows before per-shard flush (default: 200000)",
    )
    parser.add_argument(
        "--metadata-shards-dir",
        type=Path,
        default=None,
        help=(
            "Metadata shard TSV directory used by --output-per-shard-dir. "
            "Default: <data_dir>/metadata/shards"
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing existing .jsonl files in --output-per-shard-dir",
    )
    args = parser.parse_args()
    logger.info("Using polars backend for key joins")

    lang_arg = args.languages.strip().lower()
    if lang_arg in LANGUAGE_PRESETS:
        allowed_langs = LANGUAGE_PRESETS[lang_arg]  # None for "all"
    else:
        allowed_langs = {l.strip() for l in args.languages.split(",")}
    if allowed_langs is None:
        logger.info("Filtering for ALL languages (except nospeech)")
    else:
        logger.info(f"Filtering for {len(allowed_langs)} languages: {sorted(allowed_langs)}")

    lang_id_path = args.data_dir / args.lang_id_file
    vad_path = args.data_dir / args.vad_file
    metadata_shards_dir = (
        args.metadata_shards_dir
        if args.metadata_shards_dir is not None
        else args.data_dir / "metadata" / "shards"
    )

    # Phase 1: read metadata once to scope this run to selected shards only.
    logger.info(f"Reading metadata shards from {metadata_shards_dir}")
    key_to_shard, key_to_duration, key_to_sample_rate, duplicate_keys, scanned_tsv_count = _collect_metadata_key_to_shard(
        metadata_shards_dir=metadata_shards_dir
    )
    logger.info(
        f"Metadata shard map: {len(key_to_shard)} keys from {scanned_tsv_count} TSVs"
    )
    if duplicate_keys:
        logger.warning(
            f"Found {duplicate_keys} duplicate sample keys across metadata shards; "
            "first shard occurrence was kept."
        )
    _prepare_per_shard_output_dir(
        out_dir=args.output_per_shard_dir,
        overwrite=args.overwrite_output,
    )

    # Phase 2: parse lang_id but only keep keys in selected metadata.
    logger.info(f"Reading language IDs from {lang_id_path}")
    key_to_lang, total_lang_id_lines, skipped_nospeech = _collect_lang_for_metadata_keys(
        lang_id_path=lang_id_path,
        metadata_keys=set(key_to_shard.keys()),
        allowed_langs=allowed_langs,
    )
    logger.info(
        f"Lang_id pass done: {total_lang_id_lines} total, {skipped_nospeech} nospeech, "
        f"{len(key_to_lang)} keys in selected metadata shards"
    )

    target_lookup, target_df = _build_target_lookup_with_polars(
        key_to_lang=key_to_lang,
        key_to_shard=key_to_shard,
        key_to_duration=key_to_duration,
        key_to_sample_rate=key_to_sample_rate,
    )
    logger.info(
        f"Per-shard join map built with {len(target_lookup)} keys "
        "(lang+metadata intersection)"
    )
    if target_df.height:
        lang_counts_df = target_df.group_by("lang").len().sort("len", descending=True)
        for row in lang_counts_df.iter_rows(named=True):
            logger.info(f"  {row['lang']}: {row['len']}")
    if not target_lookup:
        logger.warning("No matching keys after metadata+lang filtering; nothing to write.")
        return

    # Phase 3: stream VAD once and stop early when all target keys are found.
    # Optimization: extract key via string slicing before JSON parse.
    # Lines are {"key": {...}}, so key is between the first two quotes.
    # This skips orjson.loads() for ~95% of non-matching lines.
    logger.info(f"Reading VAD results from {vad_path}")
    vad_total = 0
    matched = 0
    skipped_fast = 0
    remaining_keys = set(target_lookup.keys())
    shard_to_lines: Dict[str, List[str]] = {}
    buffered_lines = 0

    with open(vad_path, "r", encoding="utf-8") as fin:
        for line in fin:
            vad_total += 1

            # Fast pre-check: extract key between first two quotes.
            # Use orjson to decode the quoted key (handles \u00f3 -> ó).
            try:
                first = line.index('"')
                second = line.index('"', first + 1)
                raw_key_fast = orjson.loads(line[first:second + 1])
            except (ValueError, orjson.JSONDecodeError):
                continue
            key_norm = _normalize_key(raw_key_fast)
            if key_norm not in remaining_keys:
                skipped_fast += 1
                continue

            # Full parse only for matching keys.
            parsed = _parse_vad_entry(line)
            if parsed is None:
                continue
            key_norm, raw_key, entry = parsed

            lang, shard, duration, sample_rate = target_lookup[key_norm]
            entry[raw_key]["lang"] = lang
            if duration is not None:
                entry[raw_key]["duration_sec"] = duration
            if sample_rate is not None:
                entry[raw_key]["sample_rate"] = sample_rate
            line_out = orjson.dumps(entry).decode("utf-8") + "\n"
            shard_to_lines.setdefault(shard, []).append(line_out)
            matched += 1
            buffered_lines += 1
            remaining_keys.remove(key_norm)

            if buffered_lines >= args.vad_batch_lines:
                _flush_shard_buffers(
                    shard_to_lines=shard_to_lines,
                    output_per_shard_dir=args.output_per_shard_dir,
                )
                buffered_lines = 0

            if not remaining_keys:
                logger.info(f"Matched all target keys; early stop at VAD line {vad_total}")
                break

            if vad_total % 500000 == 0:
                logger.info(f"  VAD progress: {vad_total} lines, {matched} matched")

    if shard_to_lines:
        _flush_shard_buffers(
            shard_to_lines=shard_to_lines,
            output_per_shard_dir=args.output_per_shard_dir,
        )

    missing_in_vad = len(remaining_keys)
    logger.info(
        f"VAD pass done: {vad_total} scanned, {matched} matched, "
        f"{skipped_fast} skipped (fast pre-check), {missing_in_vad} missing"
    )
    if missing_in_vad:
        sample_missing = list(remaining_keys)[:10]
        logger.warning(
            f"Examples of missing keys in VAD (showing {len(sample_missing)}): "
            + ", ".join(sample_missing)
        )
    logger.info(f"Per-shard filtered VAD written to {args.output_per_shard_dir}")


if __name__ == "__main__":
    main()
