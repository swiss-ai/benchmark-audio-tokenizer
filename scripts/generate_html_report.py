#!/usr/bin/env python3
"""Generate a self-contained HTML report from audio inference JSON results.

Usage:
    python scripts/generate_html_report.py --results-dir results/inference_greedy_v2
    python scripts/generate_html_report.py --results-dir results/inference_greedy_v2 --wav-root results/inference
"""

import argparse
import base64
import os
from glob import glob
from html import escape
from pathlib import Path

from audio_tokenization.contracts import InferenceRun, read_inference_run


_AUDIO_MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}
# v1 fallback only: probed extensions when audio_uri is missing. v2 readers
# never hit this list.
_LEGACY_EXT_PROBES = (".wav", ".mp3", ".flac")

# Task → (subtitle shown next to the task badge, label of the reference column).
# Reference column meaning differs per task: it's the ground-truth transcript
# for transcribe, the actual continuation for continue, and the source-language
# text for translate.
_TASK_SUBTITLE = {
    "transcribe": "Speech-to-text transcription",
    "continue": "Text continuation from speech",
    "translate": "Speech translation",
}
_TASK_REF_LABEL = {
    "transcribe": "Ground Truth",
    "continue": "Transcription",
    "translate": "Source Transcription",
}


def load_report_runs(results_dir: str) -> dict[str, dict[str, InferenceRun]]:
    """Load every {results_dir}/{ds_name}/{ckpt_task}.json into typed runs.

    Returns ``{dataset_name: {ckpt_task_key: InferenceRun}}``. Dataset order
    follows DATASET_ORDER first, then alphabetical for the remainder.
    """
    unordered: dict[str, dict[str, InferenceRun]] = {}
    for ds_dir in sorted(glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(ds_dir):
            continue
        ds_name = os.path.basename(ds_dir)
        unordered[ds_name] = {}
        for jf in sorted(glob(os.path.join(ds_dir, "*.json"))):
            key = os.path.splitext(os.path.basename(jf))[0]
            unordered[ds_name][key] = read_inference_run(Path(jf))
    out: dict[str, dict[str, InferenceRun]] = {}
    for ds in DATASET_ORDER:
        if ds in unordered:
            out[ds] = unordered.pop(ds)
    for ds in sorted(unordered):
        out[ds] = unordered[ds]
    return out


def _embed_b64(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = _AUDIO_MIME.get(ext, "audio/wav")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def resolve_audio_src(
    record,
    *,
    wav_root: str,
    ds_name: str,
    sample_idx: int,
    embed: bool,
    audio_url_prefix: str,
) -> str:
    """Return the value for an HTML ``<audio src=...>`` for one record.

    v2 (``record.audio_uri`` set):
      - ``embed=True``: read the canonical file and base64-encode it.
      - ``embed=False``: emit ``{audio_url_prefix}/{ds_name}/{basename}``
        derived from the canonical filename. Portable for static hosting
        (GitHub Pages, copied report tree) — the operator's contract is to
        place a copy of the audio at that URL with the same basename.
        Returning the absolute path would break any report served off the
        producing machine. Caveat: basename uniqueness within ``ds_name``
        is the operator's responsibility; if the producer can emit colliding
        basenames across different source dirs, a copy/bundle step that
        rewrites paths is needed.

    v1 legacy (``audio_uri is None``): probe ``wav_root/ds_name`` for
    ``{sample_id}.{wav|mp3|flac}`` then ``sample_{i}.{wav|mp3|flac}``. This
    path exists only for files written before schema v2.
    """
    if record.audio_uri:
        if embed:
            if not os.path.isfile(record.audio_uri):
                return ""
            return _embed_b64(record.audio_uri)
        basename = os.path.basename(record.audio_uri)
        return f"{audio_url_prefix}/{ds_name}/{basename}"

    for name in (record.sample_id, f"sample_{sample_idx}"):
        for ext in _LEGACY_EXT_PROBES:
            candidate = os.path.join(wav_root, ds_name, name + ext)
            if os.path.isfile(candidate):
                if embed:
                    return _embed_b64(candidate)
                return f"{audio_url_prefix}/{ds_name}/{name + ext}"
    return ""


DATASET_LABELS = {
    "fleurs_en_us": "FLEURS (English)",
    "commonvoice_de": "CommonVoice (German)",
    "commonvoice_fr": "CommonVoice (French)",
    "eurospeech_uk": "EuroSpeech (UK)",
    "aishell1_test": "AISHELL-1 (Mandarin)",
    "spc_r_test": "SPC-R (Swiss German)",
}

# Use this ordering instead of alphabetical
DATASET_ORDER = list(DATASET_LABELS.keys())


def render_report_html(
    datasets: dict[str, dict[str, InferenceRun]],
    wav_root: str,
    *,
    embed_audio: bool = True,
    audio_url_prefix: str = "audio",
) -> str:
    """Pure: build the HTML string. No I/O except embed_audio reads."""

    all_keys = set()
    for ds_results in datasets.values():
        all_keys.update(ds_results.keys())

    ckpt_task_pairs = []
    for key in sorted(all_keys):
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            ckpt_task_pairs.append((parts[0], parts[1]))

    # Sort tasks so transcribe comes first
    tasks = sorted(set(t for _, t in ckpt_task_pairs), key=lambda t: (t != "transcribe", t))
    checkpoints = sorted(set(c for c, _ in ckpt_task_pairs))

    ds_names = list(datasets.keys())

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Apertus Audio Stage 2 &mdash; Ablation Report (v2)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --brand: #4f6df5;
    --brand-light: #eef1fe;
    --brand-soft: #dbe1fc;
    --teal: #0ea5a0;
    --teal-light: #e6f7f6;
    --amber: #d97706;
    --amber-light: #fef9ec;
    --purple: #8b5cf6;
    --purple-light: #f3f0ff;
    --rose: #e11d48;
    --text: #334155;
    --text-light: #64748b;
    --text-muted: #94a3b8;
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --border-light: #f1f5f9;
    --radius: 16px;
    --radius-sm: 10px;
    --shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.06);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text);
    padding: 40px 32px; max-width: 1440px; margin: 0 auto;
    line-height: 1.6; -webkit-font-smoothing: antialiased;
  }

  /* Header */
  .header {
    text-align: center; margin-bottom: 40px; padding: 48px 32px;
    background: linear-gradient(135deg, var(--brand-light) 0%, #f0e6ff 50%, var(--teal-light) 100%);
    border-radius: var(--radius); border: 1px solid var(--border);
  }
  .header h1 {
    font-size: 32px; font-weight: 700; color: var(--text);
    margin-bottom: 8px; letter-spacing: -0.5px;
  }
  .header .subtitle {
    color: var(--text-light); font-size: 16px; font-weight: 400;
  }

  /* Info cards */
  .info-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 20px; margin-bottom: 32px;
  }
  .info-card {
    background: var(--card); border-radius: var(--radius); padding: 24px;
    border: 1px solid var(--border); transition: box-shadow 0.2s ease;
  }
  .info-card:hover { box-shadow: var(--shadow-md); }
  .info-card .card-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 10px; margin-bottom: 14px;
    font-size: 18px;
  }
  .info-card .card-icon.setup { background: var(--brand-light); }
  .info-card .card-icon.ckpt { background: var(--purple-light); }
  .info-card .card-icon.task { background: var(--amber-light); }
  .info-card h3 {
    font-size: 15px; font-weight: 600; color: var(--text);
    margin-bottom: 10px;
  }
  .info-card p { font-size: 13.5px; color: var(--text-light); line-height: 1.65; }
  .info-card p + p { margin-top: 10px; }
  .info-card code {
    background: var(--border-light); padding: 2px 7px; border-radius: 5px;
    font-size: 12px; font-weight: 500; color: var(--text);
  }
  .info-card strong { color: var(--text); font-weight: 600; }

  /* Legend */
  .legend {
    display: flex; flex-wrap: wrap; gap: 24px; margin-bottom: 28px;
    background: var(--card); padding: 14px 24px; border-radius: var(--radius-sm);
    border: 1px solid var(--border); font-size: 13px; align-items: center;
    color: var(--text-light);
  }
  .legend strong { color: var(--text); }
  .legend-item { display: flex; align-items: center; gap: 8px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
  .legend-dot.gt { background: var(--teal); }
  .legend-dot.w0 { background: var(--purple); }
  .legend-dot.w1 { background: var(--brand); }

  /* Dataset tabs */
  .tabs {
    display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 0;
    padding: 6px; background: var(--card); border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    border: 1px solid var(--border); border-bottom: none;
  }
  .tab {
    padding: 9px 18px; background: transparent; border: none; cursor: pointer;
    font-size: 13px; font-weight: 500; color: var(--text-light);
    border-radius: 8px; transition: all 0.15s ease;
    font-family: inherit;
  }
  .tab:hover { background: var(--border-light); color: var(--text); }
  .tab.active {
    background: var(--brand); color: #fff; font-weight: 600;
    box-shadow: 0 2px 8px rgba(79,109,245,0.25);
  }
  .tab-content {
    display: none; padding: 24px;
    background: var(--card); border: 1px solid var(--border); border-top: none;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm);
  }
  .tab-content.active { display: block; }

  /* Task sections */
  .task-section { margin-bottom: 36px; }
  .task-section:last-child { margin-bottom: 0; }
  .task-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 16px;
  }
  .task-badge {
    display: inline-block; padding: 5px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;
  }
  .task-badge.transcribe { background: var(--teal-light); color: var(--teal); }
  .task-badge.continue { background: var(--amber-light); color: var(--amber); }
  .task-title { font-size: 17px; font-weight: 600; color: var(--text); }

  /* Tables */
  table {
    border-collapse: separate; border-spacing: 0; width: 100%;
    background: var(--card); border-radius: var(--radius-sm);
    border: 1px solid var(--border); overflow: hidden;
  }
  thead th {
    background: var(--border-light); color: var(--text);
    padding: 12px 16px; text-align: left;
    font-size: 11.5px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.6px;
    border-bottom: 2px solid var(--border);
  }
  thead th.w0-col { color: var(--purple); }
  thead th.w1-col { color: var(--brand); }
  tbody td {
    padding: 14px 16px; border-bottom: 1px solid var(--border-light);
    font-size: 13.5px; vertical-align: top;
  }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: #fafbff; }
  .gt-text { color: var(--teal); }
  .w0-text { color: var(--purple); }
  .w1-text { color: var(--brand); }
  .meta { color: var(--text-muted); font-size: 11px; margin-top: 4px; }
  audio {
    height: 34px; width: 200px; border-radius: 8px;
  }
  .sample-info { font-weight: 600; font-size: 14px; color: var(--text); }

  /* Footer */
  .footer {
    text-align: center; margin-top: 48px; padding: 20px;
    color: var(--text-muted); font-size: 12px;
  }
</style>
</head>
<body>

<div class="header">
  <h1>Apertus Audio Stage 2 &mdash; Ablation Report</h1>
  <p class="subtitle">Qualitative evaluation of speech understanding across 6 multilingual benchmarks</p>
</div>

<div class="info-grid">
  <div class="info-card">
    <div class="card-icon setup">&#9881;</div>
    <h3>Experiment Setup</h3>
    <p>Both models are trained on the <strong>same 50/50 audio + text data mix</strong> (36.5 B audio tokens
       interleaved with text data) for continued pretraining from the Apertus-1.5 8B Stage 1 checkpoint.
       The only difference is whether the model <em>learns from</em> audio tokens or ignores them.</p>
  </div>
  <div class="info-card">
    <div class="card-icon ckpt">&#9878;</div>
    <h3>Checkpoints</h3>
    <p><code>weight-0</code> &mdash; <strong>Audio loss weight = 0.</strong>
       Audio data is present in the training mix, but the loss on audio tokens is zeroed out.
       The model sees audio context but is not trained to predict audio tokens.</p>
    <p><code>weight-1</code> &mdash; <strong>Audio loss weight = 1.</strong>
       Standard training where the model learns from both audio and text tokens equally.
       This is the full multimodal training objective.</p>
  </div>
  <div class="info-card">
    <div class="card-icon task">&#9836;</div>
    <h3>Tasks</h3>
    <p><strong>Transcribe</strong> &mdash; Given an audio clip, the model produces a text transcription.
       Prompt: <code>[audio] &lt;|stt_transcribe|&gt;</code>. Measures speech recognition quality.</p>
    <p><strong>Continue</strong> &mdash; Given an audio clip, the model generates
       a natural text continuation of what was said.
       Prompt: <code>[audio] &lt;|stt_continue|&gt;</code>. Measures speech understanding and coherence.</p>
  </div>
</div>

<div class="legend">
  <strong>Legend:</strong>
  <div class="legend-item"><div class="legend-dot gt"></div> Ground truth / transcription</div>
  <div class="legend-item"><div class="legend-dot w0"></div> weight-0 prediction</div>
  <div class="legend-item"><div class="legend-dot w1"></div> weight-1 prediction</div>
</div>
""")

    # Tab buttons
    html_parts.append('<div class="tabs">')
    for i, ds_name in enumerate(ds_names):
        active = " active" if i == 0 else ""
        label = DATASET_LABELS.get(ds_name, ds_name)
        html_parts.append(
            f'<button class="tab{active}" onclick="switchTab(\'{ds_name}\')">{label}</button>'
        )
    html_parts.append('</div>')

    # Per-dataset tab content
    for tab_idx, (ds_name, ds_results) in enumerate(datasets.items()):
        active = " active" if tab_idx == 0 else ""
        html_parts.append(f'<div class="tab-content{active}" id="tab-{ds_name}">')

        for task in tasks:
            html_parts.append('<div class="task-section">')
            html_parts.append('<div class="task-header">')
            html_parts.append(f'<span class="task-badge {task}">{task}</span>')
            task_desc = _TASK_SUBTITLE.get(task, task)
            html_parts.append(f'<span class="task-title">{task_desc}</span>')
            html_parts.append('</div>')

            task_data = {}
            for ckpt in checkpoints:
                key = f"{ckpt}_{task}"
                if key in ds_results:
                    task_data[ckpt] = ds_results[key]

            if not task_data:
                html_parts.append('<p>No results available.</p>')
                html_parts.append('</div>')
                continue

            html_parts.append('<table>')
            html_parts.append('<thead><tr>')
            html_parts.append('<th style="width:60px">#</th><th style="width:220px">Audio</th>')
            ref_label = _TASK_REF_LABEL.get(task, "Reference")
            html_parts.append(f'<th>{ref_label}</th>')
            for ckpt in checkpoints:
                if ckpt in task_data:
                    if "weight-0" in ckpt:
                        html_parts.append(f'<th class="w0-col">weight-0</th>')
                    else:
                        html_parts.append(f'<th class="w1-col">weight-1</th>')
            html_parts.append('</tr></thead>')
            html_parts.append('<tbody>')

            first_ckpt: InferenceRun = next(iter(task_data.values()))
            num_samples = len(first_ckpt.records)

            for i in range(num_samples):
                html_parts.append('<tr>')

                sample = first_ckpt.records[i]
                duration = sample.duration_s if sample.duration_s is not None else "?"

                html_parts.append(
                    f'<td><span class="sample-info">{i+1}</span>'
                    f'<div class="meta">{duration}s</div></td>'
                )

                audio_src = resolve_audio_src(
                    sample,
                    wav_root=wav_root,
                    ds_name=ds_name,
                    sample_idx=i,
                    embed=embed_audio,
                    audio_url_prefix=audio_url_prefix,
                )

                if audio_src:
                    html_parts.append(f'<td><audio controls src="{audio_src}"></audio></td>')
                else:
                    html_parts.append('<td><em>no audio</em></td>')

                html_parts.append(f'<td class="gt-text">{escape(sample.reference_text)}</td>')

                for ckpt in checkpoints:
                    if ckpt not in task_data:
                        continue
                    ckpt_run: InferenceRun = task_data[ckpt]
                    css_class = "w0-text" if "weight-0" in ckpt else "w1-text"
                    if i < len(ckpt_run.records):
                        rec = ckpt_run.records[i]
                        pred = rec.prediction_text
                        gen_time = rec.gen_time_s if rec.gen_time_s is not None else "?"
                        html_parts.append(
                            f'<td><span class="{css_class}">{escape(pred)}</span>'
                            f'<div class="meta">{gen_time}s</div></td>'
                        )
                    else:
                        html_parts.append('<td>-</td>')

                html_parts.append('</tr>')

            html_parts.append('</tbody></table>')
            html_parts.append('</div>')  # task-section

        html_parts.append('</div>')  # tab-content

    html_parts.append("""
<script>
function switchTab(ds) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + ds).classList.add('active');
  document.querySelectorAll('.tab').forEach(el => {
    if (el.getAttribute('onclick').includes("'" + ds + "'")) el.classList.add('active');
  });
  window.scrollTo({top: document.querySelector('.tabs').offsetTop - 10, behavior: 'smooth'});
}
</script>
<div class="footer">Apertus &mdash; Swiss AI</div>
</body></html>""")

    return "\n".join(html_parts)


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from inference results.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to results directory (e.g. results/inference_greedy_v2)")
    parser.add_argument("--wav-root", type=str, default="results/inference",
                        help="Root directory with wav files per dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path. Default: {results-dir}/report.html")
    parser.add_argument("--no-embed-audio", action="store_true",
                        help="Use relative paths for audio instead of base64 embedding. "
                             "Useful for GitHub Pages where wav files are served separately.")
    parser.add_argument("--audio-url-prefix", type=str, default="audio",
                        help="URL prefix for audio files when --no-embed-audio is set.")
    args = parser.parse_args()

    output = args.output or os.path.join(args.results_dir, "report.html")
    datasets = load_report_runs(args.results_dir)
    print(f"Loaded {sum(len(v) for v in datasets.values())} result files across {len(datasets)} datasets")
    html = render_report_html(
        datasets,
        args.wav_root,
        embed_audio=not args.no_embed_audio,
        audio_url_prefix=args.audio_url_prefix,
    )
    with open(output, "w") as f:
        f.write(html)
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
