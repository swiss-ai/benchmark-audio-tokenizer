"""Tests for the prediction-record contract and the report's pure helpers.

scripts/ is added to sys.path by audio_tokenization/tests/conftest.py so the
report module can be imported without per-file path hacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_tokenization.contracts import (
    CURRENT_INFERENCE_OUTPUT_VERSION,
    InferenceRun,
    PredictionRecord,
    read_inference_run,
    write_inference_run,
)


def _make_run(records: list[PredictionRecord], **overrides) -> InferenceRun:
    base = dict(
        task="transcribe",
        model_path="/models/foo",
        dataset_name="test_ds",
        data_source="/data/test",
        backend="transformers",
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        records=records,
    )
    base.update(overrides)
    return InferenceRun(**base)


def _v1_payload(*, task: str, key: str, records: list[dict]) -> dict:
    return {
        "model_path": "/models/foo",
        "data_source": "/data/test",
        "dataset_name": "test_ds",
        "task": task,
        "backend": "transformers",
        "num_samples": len(records),
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "results": [
            {
                "sample_idx": i,
                "sample_id": f"s{i}",
                "duration_s": 1.0,
                "ground_truth": f"gt-{i}",
                key: f"pred-{i}",
            }
            for i in range(len(records))
        ],
    }


# ---------------------------------------------------------------------------
# Round-trip + writer contract
# ---------------------------------------------------------------------------


def test_round_trip_v2(tmp_path):
    rec = PredictionRecord(
        sample_idx=0, sample_id="s0", duration_s=1.5,
        audio_uri="/abs/path/s0.flac", reference_text="hello world",
        prediction_text="hi", gen_time_s=0.42,
    )
    run = _make_run([rec])
    out = tmp_path / "run.json"
    write_inference_run(out, run)
    loaded = read_inference_run(out)
    assert loaded.records == [rec]
    assert loaded.task == "transcribe"
    # schema_version lives on the wire, not on the dataclass.
    assert json.loads(out.read_text())["schema_version"] == CURRENT_INFERENCE_OUTPUT_VERSION


def test_write_emits_num_samples_and_schema_version(tmp_path):
    """schema_version + num_samples are wire-format fields injected on write,
    not part of the in-memory dataclass; they always reflect ground truth."""
    recs = [
        PredictionRecord(sample_idx=i, sample_id=f"s{i}", duration_s=1.0,
                         audio_uri=None, reference_text="", prediction_text="x")
        for i in range(3)
    ]
    out = tmp_path / "run.json"
    write_inference_run(out, _make_run(recs))
    payload = json.loads(out.read_text())
    assert payload["num_samples"] == 3
    assert payload["schema_version"] == CURRENT_INFERENCE_OUTPUT_VERSION


def test_read_rejects_num_samples_mismatch(tmp_path):
    payload = {
        "schema_version": CURRENT_INFERENCE_OUTPUT_VERSION,
        "task": "transcribe", "model_path": "/m", "dataset_name": "d",
        "data_source": "/s", "backend": "transformers", "num_samples": 5,
        "max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0,
        "results": [
            {"sample_idx": 0, "sample_id": "s0", "duration_s": 1.0,
             "audio_uri": None, "reference_text": "", "prediction_text": "x"},
        ],
    }
    out = tmp_path / "run.json"
    out.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="num_samples"):
        read_inference_run(out)


# ---------------------------------------------------------------------------
# v1 → v2 migration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "key"),
    [
        ("transcribe", "transcription"),
        ("continue", "continuation"),
        ("translate", "translation"),
    ],
)
def test_v1_legacy_migrates(tmp_path, task, key):
    out = tmp_path / "run.json"
    out.write_text(json.dumps(_v1_payload(task=task, key=key, records=[{}, {}])))
    run = read_inference_run(out)
    assert run.task == task
    assert len(run.records) == 2
    assert run.records[0].prediction_text == "pred-0"
    assert run.records[1].prediction_text == "pred-1"
    assert run.records[0].reference_text == "gt-0"
    assert run.records[0].audio_uri is None  # v1 had no canonical path


def test_v1_legacy_missing_task_rejected(tmp_path):
    payload = _v1_payload(task="transcribe", key="transcription", records=[{}])
    payload.pop("task")
    out = tmp_path / "run.json"
    out.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match=r"missing required field"):
        read_inference_run(out)


def test_unknown_future_version_rejected(tmp_path):
    out = tmp_path / "run.json"
    out.write_text(json.dumps({"schema_version": CURRENT_INFERENCE_OUTPUT_VERSION + 99}))
    with pytest.raises(RuntimeError, match="only knows how to read up to"):
        read_inference_run(out)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_atomic_write_target_unchanged_on_failure(tmp_path, monkeypatch):
    """If json.dumps raises mid-write, the previously-written target file
    must remain byte-identical."""
    rec = PredictionRecord(sample_idx=0, sample_id="s0", duration_s=1.0,
                           audio_uri=None, reference_text="r", prediction_text="p")
    out = tmp_path / "run.json"

    write_inference_run(out, _make_run([rec]))
    original_bytes = out.read_bytes()

    import audio_tokenization.contracts.prediction as pred_mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated serialize failure")

    monkeypatch.setattr(pred_mod.json, "dumps", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        write_inference_run(out, _make_run([rec, rec]))

    assert out.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def test_resolve_audio_src_v2_no_embed_emits_relative_url(tmp_path):
    """Regression: v2 + --no-embed-audio must NOT leak the absolute
    audio_uri into the HTML — the report would break under static hosting
    or when copied to another machine. Uses basename + audio_url_prefix +
    ds_name so the operator's serving copy keeps URLs portable.
    """
    import generate_html_report as ghr

    audio = tmp_path / "/capstor/store/scratch" / "x.flac"  # arbitrary absolute
    rec = PredictionRecord(
        sample_idx=0, sample_id="s0", duration_s=1.0,
        audio_uri="/capstor/store/scratch/x.flac",
        reference_text="", prediction_text="",
    )
    src = ghr.resolve_audio_src(
        rec, wav_root="/unused", ds_name="ds_a",
        sample_idx=0, embed=False, audio_url_prefix="audio",
    )
    assert src == "audio/ds_a/x.flac"
    assert "/capstor" not in src


def test_resolve_audio_src_v2_embed_reads_canonical_uri(tmp_path):
    import generate_html_report as ghr

    audio = tmp_path / "abs" / "x.flac"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"flacbytes")

    rec = PredictionRecord(
        sample_idx=0, sample_id="s0", duration_s=1.0,
        audio_uri=str(audio), reference_text="", prediction_text="",
    )
    src = ghr.resolve_audio_src(
        rec, wav_root="/unused", ds_name="ds_a",
        sample_idx=0, embed=True, audio_url_prefix="audio",
    )
    assert src.startswith("data:audio/flac;base64,")


def test_resolve_audio_src_v1_falls_back_to_flac(tmp_path):
    import generate_html_report as ghr

    ds_dir = tmp_path / "ds_a"
    ds_dir.mkdir()
    (ds_dir / "s0.flac").write_bytes(b"")

    rec = PredictionRecord(
        sample_idx=0, sample_id="s0", duration_s=1.0,
        audio_uri=None, reference_text="", prediction_text="",
    )
    src = ghr.resolve_audio_src(
        rec, wav_root=str(tmp_path), ds_name="ds_a",
        sample_idx=0, embed=False, audio_url_prefix="audio",
    )
    assert src == "audio/ds_a/s0.flac"


def test_render_report_html_v2_includes_predictions(tmp_path):
    import generate_html_report as ghr

    audio = tmp_path / "s0.flac"
    audio.write_bytes(b"")

    rec = PredictionRecord(
        sample_idx=0, sample_id="s0", duration_s=1.0,
        audio_uri=str(audio), reference_text="hello world",
        prediction_text="HELLO WORLD",
    )
    run = _make_run([rec])
    datasets = {"ds_a": {"weight-0_transcribe": run}}

    html = ghr.render_report_html(
        datasets, wav_root=str(tmp_path), embed_audio=False,
    )
    assert "HELLO WORLD" in html
    assert "hello world" in html
    assert "no audio" not in html


@pytest.mark.parametrize(
    ("task", "expected_subtitle", "expected_ref_label"),
    [
        ("transcribe", "Speech-to-text transcription", "Ground Truth"),
        ("continue", "Text continuation from speech", "Transcription"),
        ("translate", "Speech translation", "Source Transcription"),
    ],
)
def test_render_report_html_labels_each_task_correctly(
    tmp_path, task, expected_subtitle, expected_ref_label
):
    """Regression for P3: translate runs must not be rendered as continue.
    All three tasks need their own subtitle and reference-column label.
    """
    import generate_html_report as ghr

    rec = PredictionRecord(
        sample_idx=0, sample_id="s0", duration_s=1.0,
        audio_uri=str(tmp_path / "s0.flac"),
        reference_text="ref", prediction_text="pred",
    )
    run = _make_run([rec], task=task)
    datasets = {"ds_a": {f"weight-0_{task}": run}}

    html = ghr.render_report_html(
        datasets, wav_root=str(tmp_path), embed_audio=False,
    )
    assert expected_subtitle in html, f"missing task subtitle for {task}"
    assert f">{expected_ref_label}<" in html, f"missing ref label for {task}"
