#!/usr/bin/env python3
"""Compare fleurs WER/CER between it478000 and it430000 with Open ASR Leaderboard normalization.

For each fleurs language, computes WER (or CER for cmn/yue/ko/th) on the
intersection of sample_ids between the two checkpoints' prediction JSONs.
"""
from __future__ import annotations
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
NEW_FN = "Apertus-1p5-8B-stage3-it478000_transcribe.json"
OLD_FN = "Apertus-1p5-8B-it430000_transcribe.json"
OLD_FULL = Path("/capstor/scratch/cscs/xyixuan/recon_examples/it430000_transcribe_full")
OLD_50 = Path("/capstor/scratch/cscs/xyixuan/recon_examples/it430000_transcribe")


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


def compute_pair(refs: list[str], hyps: list[str], lang: str) -> float:
    if lang in CER_LANGS:
        refs = [" ".join(list(s)) for s in refs]
        hyps = [" ".join(list(s)) for s in hyps]
    if not refs:
        return float("nan")
    return jiwer.wer(refs, hyps)


def main() -> None:
    rows = []
    for ds_dir in sorted(NEW_ROOT.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.startswith("fleurs_"):
            continue
        lang = ds_dir.name[len("fleurs_") :]
        new_p = ds_dir / NEW_FN
        if not new_p.is_file():
            continue
        new_recs = load_records(new_p)

        old_p = OLD_FULL / ds_dir.name / OLD_FN
        ref_label = "full"
        if not old_p.is_file():
            old_p = OLD_50 / ds_dir.name / OLD_FN
            ref_label = "50"
        if not old_p.is_file():
            rows.append((lang, "—", float("nan"), float("nan"), 0, "no it430000 ref"))
            continue
        old_recs = load_records(old_p)

        common = sorted(set(new_recs) & set(old_recs))
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
    print(f"{'language':<18} {'metric':<5} {'ref':<5} {'it478000':>10} {'it430000':>10} {'delta':>8} {'n':>6} note")
    print("-" * 78)
    for lang, ref, new_err, old_err, n, note in rows:
        m = metric_for(lang)
        if new_err != new_err or old_err != old_err:  # NaN
            print(f"{lang:<18} {m:<5} {ref:<5} {'—':>10} {'—':>10} {'—':>8} {n:>6} {note}")
            continue
        delta = (new_err - old_err) * 100
        sign = "+" if delta >= 0 else ""
        print(f"{lang:<18} {m:<5} {ref:<5} {new_err*100:>9.2f}% {old_err*100:>9.2f}% {sign}{delta:>6.2f}pp {n:>6} {note}")
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
