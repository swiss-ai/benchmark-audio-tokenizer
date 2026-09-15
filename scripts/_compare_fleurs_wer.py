#!/usr/bin/env python3
"""Compare FLEURS WER/CER between two prediction roots.

For each fleurs language, computes WER (or CER for cmn/yue/ko/th) on the
intersection of sample_ids between the two checkpoints' prediction JSONs.
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

NEW_ROOT = Path("/capstor/scratch/cscs/xyixuan/recon_examples/it478000_transcribe")
OLD_FULL = Path("/capstor/scratch/cscs/xyixuan/recon_examples/it430000_transcribe_full")


def normalize(text: str, lang: str) -> str:
    n = EN_NORM if lang == "en_us" else BASIC_NORM
    return n((text or "").strip())


def load_records(p: Path) -> dict[str, dict]:
    with open(p) as f:
        d = json.load(f)
    out = {}
    for r in d["records"]:
        sid = str(r.get("sample_id"))
        if sid is not None:
            out[sid] = r
    return out


def resolve_prediction_json(ds_dir: Path, pattern: str) -> Path | None:
    matches = sorted(ds_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        joined = ", ".join(p.name for p in matches)
        raise RuntimeError(
            f"ambiguous prediction JSONs in {ds_dir}: {joined}; "
            "pass a stricter pattern"
        )
    return matches[0]


def compute_pair(refs: list[str], hyps: list[str], lang: str) -> float:
    if lang in CER_LANGS:
        refs = [" ".join(list(s)) for s in refs]
        hyps = [" ".join(list(s)) for s in hyps]
    if not refs:
        return float("nan")
    return jiwer.wer(refs, hyps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FLEURS WER/CER between two prediction roots."
    )
    parser.add_argument("new_root", nargs="?", type=Path, default=NEW_ROOT)
    parser.add_argument("old_root", nargs="?", type=Path, default=OLD_FULL)
    parser.add_argument("--new-pattern", default="*_transcribe.json")
    parser.add_argument("--old-pattern", default="*_transcribe.json")
    parser.add_argument("--new-label", default=None)
    parser.add_argument("--old-label", default=None)
    parser.add_argument(
        "--old-fallback-root",
        type=Path,
        default=None,
        help="Optional fallback root used when old_root has no matching file/overlap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    new_label = args.new_label or args.new_root.name
    old_label = args.old_label or args.old_root.name
    rows = []
    for ds_dir in sorted(args.new_root.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.startswith("fleurs_"):
            continue
        lang = ds_dir.name[len("fleurs_") :]
        new_p = resolve_prediction_json(ds_dir, args.new_pattern)
        if new_p is None:
            continue
        new_recs = load_records(new_p)

        old_ds_dir = args.old_root / ds_dir.name
        old_p = resolve_prediction_json(old_ds_dir, args.old_pattern) if old_ds_dir.is_dir() else None
        fallback_ds_dir = args.old_fallback_root / ds_dir.name if args.old_fallback_root else None
        fallback_p = (
            resolve_prediction_json(fallback_ds_dir, args.old_pattern)
            if fallback_ds_dir and fallback_ds_dir.is_dir()
            else None
        )
        ref_label = args.old_root.name
        if old_p is None and fallback_p is not None:
            old_p = fallback_p
            ref_label = args.old_fallback_root.name
        if old_p is None:
            rows.append((lang, "—", float("nan"), float("nan"), 0, f"no {old_label} ref"))
            continue
        old_recs = load_records(old_p)

        common = sorted(set(new_recs) & set(old_recs))
        if not common and fallback_p is not None and old_p != fallback_p:
            old_p = fallback_p
            old_recs = load_records(old_p)
            common = sorted(set(new_recs) & set(old_recs))
            ref_label = args.old_fallback_root.name
        if not common:
            rows.append((lang, ref_label, float("nan"), float("nan"), 0, "no overlap"))
            continue

        # Build aligned ref/hyp lists for both checkpoints on the common subset.
        ref_text, hyp_new, hyp_old = [], [], []
        skipped = 0
        for sid in common:
            ref = normalize(new_recs[sid].get("reference_text") or "", lang)
            if not ref:
                skipped += 1
                continue
            new_hyp = normalize(new_recs[sid].get("prediction_text") or "", lang)
            old_hyp = normalize(old_recs[sid].get("prediction_text") or "", lang)
            ref_text.append(ref)
            hyp_new.append(new_hyp)
            hyp_old.append(old_hyp)

        new_err = compute_pair(ref_text, hyp_new, lang)
        old_err = compute_pair(ref_text, hyp_old, lang)
        rows.append((lang, ref_label, new_err, old_err, len(ref_text), ""))

    metric_for = lambda l: "CER" if l in CER_LANGS else "WER"
    rows.sort(key=lambda r: (r[2] if r[2] == r[2] else 1.0))
    print(f"{'language':<18} {'metric':<5} {'ref':<24} {new_label:>14} {old_label:>14} {'delta':>8} {'n':>6} note")
    print("-" * 78)
    for lang, ref, new_err, old_err, n, note in rows:
        m = metric_for(lang)
        if new_err != new_err or old_err != old_err:  # NaN
            print(f"{lang:<18} {m:<5} {ref:<24} {'—':>14} {'—':>14} {'—':>8} {n:>6} {note}")
            continue
        delta = (new_err - old_err) * 100
        print(f"{lang:<18} {m:<5} {ref:<24} {new_err*100:>13.2f}% {old_err*100:>13.2f}% {delta:+7.2f}pp {n:>6} {note}")
    print("-" * 78)
    valid = [(r[2], r[3]) for r in rows if r[2] == r[2] and r[3] == r[3] and metric_for(r[0]) == "WER"]
    if valid:
        avg_new = sum(a for a, _ in valid) / len(valid)
        avg_old = sum(b for _, b in valid) / len(valid)
        print(f"{'macro-avg WER':<29} {avg_new*100:>9.2f}% {avg_old*100:>9.2f}% {(avg_new-avg_old)*100:+.2f}pp ({len(valid)} langs)")
    valid_c = [(r[2], r[3]) for r in rows if r[2] == r[2] and r[3] == r[3] and metric_for(r[0]) == "CER"]
    if valid_c:
        avg_new = sum(a for a, _ in valid_c) / len(valid_c)
        avg_old = sum(b for _, b in valid_c) / len(valid_c)
        print(f"{'macro-avg CER':<29} {avg_new*100:>9.2f}% {avg_old*100:>9.2f}% {(avg_new-avg_old)*100:+.2f}pp ({len(valid_c)} langs)")


if __name__ == "__main__":
    main()
