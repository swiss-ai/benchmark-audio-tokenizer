#!/usr/bin/env python3
"""Compute WER/CER per FLEURS language using Open ASR Leaderboard normalization.

- EnglishTextNormalizer for English; BasicTextNormalizer for the rest.
- CER for non-whitespace-segmented languages (cmn, yue, ko, th); WER otherwise.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import jiwer
from transformers.models.whisper.english_normalizer import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
)

CER_LANGS = {"cmn_hans_cn", "yue_hant_hk", "ko_kr", "th_th"}
EN_NORM = EnglishTextNormalizer({})
BASIC_NORM = BasicTextNormalizer()
DEFAULT_ROOT = Path("/capstor/scratch/cscs/xyixuan/recon_examples/it478000_transcribe")


def lang_from_dataset_dir(name: str) -> str:
    return name[len("fleurs_") :]


def normalize_pair(ref: str, hyp: str, lang: str) -> tuple[str, str]:
    if lang == "en_us":
        n = EN_NORM
    else:
        n = BASIC_NORM
    return n(ref), n(hyp)


def compute(records: list[dict], lang: str) -> tuple[float, int, int]:
    refs, hyps = [], []
    skipped_empty = 0
    for r in records:
        ref = (r.get("reference_text") or "").strip()
        hyp = (r.get("prediction_text") or "").strip()
        nref, nhyp = normalize_pair(ref, hyp, lang)
        if not nref:
            skipped_empty += 1
            continue
        if lang in CER_LANGS:
            refs.append(" ".join(list(nref)))
            hyps.append(" ".join(list(nhyp)))
        else:
            refs.append(nref)
            hyps.append(nhyp)
    if not refs:
        return float("nan"), 0, skipped_empty
    err = jiwer.wer(refs, hyps)
    return err, len(refs), skipped_empty


def resolve_prediction_json(ds_dir: Path, pattern: str) -> Path | None:
    matches = sorted(ds_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        joined = ", ".join(p.name for p in matches)
        raise RuntimeError(
            f"ambiguous prediction JSONs in {ds_dir}: {joined}; "
            "pass a stricter --pattern"
        )
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FLEURS WER/CER using Open ASR Leaderboard normalization."
    )
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--pattern",
        default="*_transcribe.json",
        help="Prediction JSON glob inside each fleurs_* directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    rows = []
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.startswith("fleurs_"):
            continue
        json_path = resolve_prediction_json(ds_dir, args.pattern)
        if json_path is None:
            continue
        with open(json_path) as f:
            records = json.load(f)["records"]
        lang = lang_from_dataset_dir(ds_dir.name)
        err, n, skipped = compute(records, lang)
        metric = "CER" if lang in CER_LANGS else "WER"
        rows.append((lang, metric, err, n, skipped))

    rows.sort(key=lambda r: r[2])
    print(f"{'language':<18} {'metric':<6} {'value':>8} {'n':>6} {'skipped':>8}")
    print("-" * 50)
    for lang, metric, err, n, skipped in rows:
        print(f"{lang:<18} {metric:<6} {err*100:>7.2f}% {n:>6} {skipped:>8}")
    print("-" * 50)
    wer_rows = [r for r in rows if r[1] == "WER"]
    cer_rows = [r for r in rows if r[1] == "CER"]
    if wer_rows:
        avg_wer = sum(r[2] for r in wer_rows) / len(wer_rows)
        print(f"{'macro-avg WER':<18} {'':<6} {avg_wer*100:>7.2f}% (over {len(wer_rows)} langs)")
    if cer_rows:
        avg_cer = sum(r[2] for r in cer_rows) / len(cer_rows)
        print(f"{'macro-avg CER':<18} {'':<6} {avg_cer*100:>7.2f}% (over {len(cer_rows)} langs)")


if __name__ == "__main__":
    main()
