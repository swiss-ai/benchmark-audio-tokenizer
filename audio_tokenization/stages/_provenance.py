"""Resolved-input provenance helpers for stage-level resume fingerprints.

Why this exists:
- stage configs can derive concrete inputs from upstream stages
- resume safety must track the *resolved* inputs, not only the explicit YAML
- when upstream prepare state is available, it should participate in the
  downstream fingerprint so upstream drift invalidates stale outputs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from audio_tokenization.config.schema import DatasetSpec, InterleaveProductSpec
from audio_tokenization.prepare.constants import (
    PREPARE_STATE_FILE,
    state_version_for_filename,
)
from audio_tokenization.prepare.metadata import normalize_optional_path
from audio_tokenization.prepare.runtime import read_prepare_state


def read_stage_provenance(
    input_dirs: Sequence[str | Path], *, state_filename: str
) -> dict[str, dict[str, Any]]:
    """Best-effort map of resolved dir -> state payload for an upstream stage."""
    provenance: dict[str, dict[str, Any]] = {}
    for input_dir in _normalize_paths(input_dirs):
        state_path = Path(input_dir) / state_filename
        if state_path.is_file():
            provenance[input_dir] = _read_upstream_state_for_provenance(
                state_path,
                state_filename=state_filename,
            )
    return provenance


def read_prepare_provenance(input_shar_dirs: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """Best-effort map of resolved input SHAR dir -> prepare state payload.

    Only dirs that carry our prepare-state file participate. External SHAR roots
    without that file still remain safe because the resolved input path itself is
    fingerprinted separately.
    """
    return read_stage_provenance(input_shar_dirs, state_filename=PREPARE_STATE_FILE)


def build_tokenize_resume_fingerprint(
    spec: DatasetSpec,
    *,
    input_shar_dirs: Sequence[str | Path],
    prepare_provenance: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resume fingerprint for tokenize, including resolved upstream inputs."""
    assert spec.tokenize is not None
    fingerprint = dict(spec.tokenize.fingerprint_payload())
    fingerprint["resolved_input_shar_dirs"] = _normalize_paths(input_shar_dirs)
    fingerprint["input_prepare_state_by_dir"] = dict(prepare_provenance or {})
    return fingerprint


def build_interleave_resume_fingerprint(
    interleave: InterleaveProductSpec,
    *,
    cache_dir: str | Path,
    tokenizer_path: str | Path,
    tokenize_provenance: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resume fingerprint for interleave, including resolved derived inputs."""
    fingerprint = dict(interleave.fingerprint_payload())
    fingerprint["resolved_cache_dir"] = normalize_optional_path(cache_dir)
    fingerprint["resolved_tokenizer_path"] = normalize_optional_path(tokenizer_path)
    fingerprint["input_tokenize_state_by_cache_dir"] = dict(tokenize_provenance or {})
    return fingerprint


def _normalize_paths(paths: Sequence[str | Path]) -> list[str]:
    return sorted(normalize_optional_path(path) for path in paths)


def _read_upstream_state_for_provenance(
    state_path: Path,
    *,
    state_filename: str,
) -> dict[str, Any]:
    """Read an upstream state file for fingerprinting.

    Stage ownership stays strict: any upstream state file that exists must use
    the current typed/versioned state contract. External SHAR roots without a
    state file are still allowed; they are fingerprinted by resolved path only.
    """
    return read_prepare_state(
        state_path, expected_version=state_version_for_filename(state_filename)
    )
