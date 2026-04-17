"""Typed contract for audio inference output JSON.

Shared by ``scripts/audio_inference.py`` (producer) and
``scripts/generate_html_report.py`` (consumer).

Two invariants worth keeping in mind:
- ``write_inference_run`` is the sole authority for ``schema_version`` and
  ``num_samples`` on disk — those are wire-format concerns, not domain
  fields, so they don't appear on ``InferenceRun``.
- ``audio_uri`` is the canonical *source* location. Bundling for portability
  (open HTML on a laptop) is a separate report-side concern, not a schema
  feature.

v1 (legacy, unversioned) files are auto-upgraded on read.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal


CURRENT_INFERENCE_OUTPUT_VERSION = 2

Task = Literal["transcribe", "continue", "translate"]
Backend = Literal["transformers", "vllm"]

# v1 → key inside results[i] holding the model output. v2 stops using these.
_V1_TASK_OUTPUT_KEY: dict[Task, str] = {
    "transcribe": "transcription",
    "continue": "continuation",
    "translate": "translation",
}

# Top-level fields that must be present on a v1 file.
_V1_REQUIRED_RUN_FIELDS: tuple[str, ...] = (
    "model_path",
    "data_source",
    "dataset_name",
    "task",
    "backend",
    "num_samples",
    "max_new_tokens",
    "temperature",
    "top_p",
    "results",
)


@dataclass(frozen=True)
class PredictionRecord:
    sample_idx: int
    sample_id: str
    duration_s: float | None
    audio_uri: str | None
    reference_text: str
    prediction_text: str
    audio_codes: int | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    text_tokens: int | None = None
    audio_tokens_out: int | None = None
    gen_time_s: float | None = None
    dataset: str | None = None  # legacy --wav-dir metadata.tsv field


@dataclass(frozen=True)
class InferenceRun:
    task: Task
    model_path: str
    dataset_name: str
    data_source: str
    backend: Backend
    max_new_tokens: int
    temperature: float
    top_p: float
    records: list[PredictionRecord] = field(default_factory=list)


def _migrate_v1_to_v2(payload: dict) -> dict:
    """Promote a legacy unversioned inference output to v2."""
    missing = [k for k in _V1_REQUIRED_RUN_FIELDS if k not in payload]
    if missing:
        raise RuntimeError(
            f"v1 inference output missing required field(s) {missing}; "
            f"cannot migrate to v{CURRENT_INFERENCE_OUTPUT_VERSION}."
        )
    task = payload["task"]
    if task not in _V1_TASK_OUTPUT_KEY:
        raise RuntimeError(
            f"v1 inference output has unknown task {task!r}; cannot pick the "
            f"right output key. Known: {sorted(_V1_TASK_OUTPUT_KEY)}"
        )
    output_key = _V1_TASK_OUTPUT_KEY[task]

    upgraded_records = [
        {
            **{k: v for k, v in rec.items() if k not in (output_key, "ground_truth")},
            "prediction_text": rec.get(output_key, ""),
            "reference_text": rec.get("ground_truth", ""),
            "audio_uri": None,
        }
        for rec in payload.get("results", [])
    ]

    upgraded = {k: v for k, v in payload.items() if k != "results"}
    upgraded["schema_version"] = 2
    upgraded["records"] = upgraded_records
    return upgraded


_INFERENCE_OUTPUT_MIGRATIONS: dict[tuple[int, int], Callable[[dict], dict]] = {
    (1, 2): _migrate_v1_to_v2,
}


def _detect_version(payload: dict) -> int:
    v = payload.get("schema_version")
    if v is None:
        return 1
    if not isinstance(v, int):
        raise RuntimeError(
            f"Invalid inference output: schema_version must be int, got "
            f"{type(v).__name__}"
        )
    return v


def _instantiate(payload: dict) -> InferenceRun:
    """Build an InferenceRun from a v2-shaped dict.

    Drops unknown record keys silently — additive fields within the same
    schema_version are allowed (a future writer may add optional metadata
    that this reader doesn't model). Breaking changes must bump the version
    and add a migration; ``_detect_version`` already rejects future versions.
    """
    record_fields = {f.name for f in dataclasses.fields(PredictionRecord)}
    records = [
        PredictionRecord(**{k: v for k, v in rec.items() if k in record_fields})
        for rec in payload.get("records", [])
    ]
    run_fields = {f.name for f in dataclasses.fields(InferenceRun)} - {"records"}
    run_kwargs = {k: payload[k] for k in run_fields if k in payload}
    return InferenceRun(records=records, **run_kwargs)


def read_inference_run(path: Path) -> InferenceRun:
    """Read an inference output JSON, auto-migrating v1 → v2 in memory.

    Does not write the upgrade back to disk — files are regenerable, and
    silent rewrites would surprise readers expecting them unchanged.

    Raises:
        FileNotFoundError: if path does not exist.
        RuntimeError: malformed JSON, future schema version, missing required
            v1 fields, unknown v1 task, or num_samples mismatch.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid inference output format: {path}")

    version = _detect_version(payload)
    if version > CURRENT_INFERENCE_OUTPUT_VERSION:
        raise RuntimeError(
            f"Inference output at {path} is version {version}, but this code "
            f"only knows how to read up to version "
            f"{CURRENT_INFERENCE_OUTPUT_VERSION}."
        )
    while version < CURRENT_INFERENCE_OUTPUT_VERSION:
        step = (version, version + 1)
        migrate = _INFERENCE_OUTPUT_MIGRATIONS.get(step)
        if migrate is None:
            raise RuntimeError(
                f"Missing inference-output migration for {step}; this is a "
                f"bug. File: {path}"
            )
        payload = migrate(payload)
        version += 1

    run = _instantiate(payload)

    if "num_samples" in payload and payload["num_samples"] != len(run.records):
        raise RuntimeError(
            f"Inference output at {path} has num_samples="
            f"{payload['num_samples']} but {len(run.records)} records. "
            "File is truncated or hand-edited."
        )

    return run


def write_inference_run(path: Path, run: InferenceRun) -> None:
    """Write an InferenceRun to *path* atomically as v2 JSON.

    Injects ``schema_version`` and ``num_samples`` on serialize; these are
    wire-format fields, not part of the in-memory dataclass. Preserves
    record list order on disk.

    No fsync: regenerable artifact, atomic visibility (``os.replace``) is
    sufficient. A node crash between write and replace leaves the target
    file unchanged; loss is at worst the current run's output, recoverable
    by re-running inference.
    """
    payload = asdict(run)
    payload["schema_version"] = CURRENT_INFERENCE_OUTPUT_VERSION
    payload["num_samples"] = len(run.records)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        # Clean up if write_text or replace raised; safe no-op if replace
        # already moved tmp away.
        tmp.unlink(missing_ok=True)
