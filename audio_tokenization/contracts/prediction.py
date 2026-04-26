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

Readers require the current schema version. Regenerate stale inference JSON
instead of migrating it in place.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


logger = logging.getLogger(__name__)


CURRENT_INFERENCE_OUTPUT_VERSION = 2

Task = Literal["transcribe", "continue", "translate"]
Backend = Literal["transformers", "vllm"]

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
    dataset: str | None = None


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


def _detect_version(payload: dict) -> int:
    v = payload.get("schema_version")
    if v is None:
        raise RuntimeError(
            "Inference output is missing schema_version. Regenerate it with "
            "the current audio_inference.py writer."
        )
    if not isinstance(v, int):
        raise RuntimeError(
            f"Invalid inference output: schema_version must be int, got "
            f"{type(v).__name__}"
        )
    return v


def _instantiate(payload: dict) -> InferenceRun:
    """Build an InferenceRun from a v2-shaped dict.

    Drops unknown record keys silently — additive fields within the same
    schema_version are allowed. Breaking changes must bump the version;
    ``_detect_version`` rejects future versions.
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
    """Read a current-version inference output JSON.

    Raises:
        FileNotFoundError: if path does not exist.
        RuntimeError: malformed JSON, missing/currently unsupported schema
            version, or num_samples mismatch.
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
    if version < CURRENT_INFERENCE_OUTPUT_VERSION:
        raise RuntimeError(
            f"Inference output at {path} is stale version {version}; expected "
            f"{CURRENT_INFERENCE_OUTPUT_VERSION}. Regenerate it with the "
            "current audio_inference.py writer."
        )

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

    Also writes a companion ``<path>.compare.txt`` with one block per record
    laying out REF/PRED on separate lines — the JSON keeps everything on a
    single line per field, which makes it hard to eyeball transcription
    quality.

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
    _write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    # Sidecar is a debug aid; failures here must not invalidate the JSON
    # we just published.
    try:
        _write_compare_txt(path.with_suffix(".compare.txt"), run)
    except OSError as e:
        logger.warning("compare.txt write failed for %s: %s", path, e)


def _write_text_atomic(path: Path, text: str) -> None:
    """Write *text* to *path* via tmp + os.replace.

    Failure semantics:
    - On any failure (write_text or os.replace), the target file is left
      unchanged — os.replace is the only thing that publishes new content,
      and it's atomic on POSIX.
    - The tmp file is always cleaned up (no orphan on partial failure,
      no-op when os.replace already moved it away).

    No fsync: regenerable artifact, atomic visibility is sufficient.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        # Safe no-op if replace already moved tmp away.
        tmp.unlink(missing_ok=True)


def _write_compare_txt(path: Path, run: InferenceRun) -> None:
    """Write a human-scannable ref/pred comparison next to the JSON."""
    lines: list[str] = []
    header = (
        f"# {run.dataset_name}  task={run.task}  "
        f"model={run.model_path}  n={len(run.records)}"
    )
    lines.append(header)
    lines.append("")
    for r in run.records:
        ref = (r.reference_text or "").strip()
        pred = (r.prediction_text or "").strip()
        lines.append(
            f"--- sample {r.sample_idx}  id={r.sample_id}  dur={r.duration_s}s ---"
        )
        lines.append(f"REF : {ref}")
        lines.append(f"PRED: {pred}")
        lines.append("")

    _write_text_atomic(path, "\n".join(lines))
