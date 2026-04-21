"""Resolved stage-plan primitives for the unified control plane.

The key idea: commands and runners should operate on the same resolved
object, not re-derive defaults / paths / fingerprints in multiple places.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any, Callable

from audio_tokenization.prepare.runtime import read_prepare_state
from audio_tokenization.stages._resume import diff_fingerprint


def _noop_preflight() -> None:
    return None


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


@dataclass
class ResolvedStagePlan:
    stage: str
    enabled: bool
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    effective: dict[str, Any]
    fingerprint: dict[str, Any]
    output_dir: Path | None
    state_path: Path | None
    success_marker: Path | None
    reason: str | None = None
    preflight: Callable[[], None] = field(default=_noop_preflight, repr=False)
    execute: Callable[[bool], dict[str, Any]] = field(default=lambda _resume: {}, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "enabled": self.enabled,
            "reason": self.reason,
            "inputs": _serialize(self.inputs),
            "outputs": _serialize(self.outputs),
            "effective": _serialize(self.effective),
            "fingerprint": _serialize(self.fingerprint),
            "paths": {
                "output_dir": _serialize(self.output_dir),
                "state_path": _serialize(self.state_path),
                "success_marker": _serialize(self.success_marker),
            },
        }


def disabled_stage_plan(*, stage: str, reason: str) -> ResolvedStagePlan:
    return ResolvedStagePlan(
        stage=stage,
        enabled=False,
        reason=reason,
        inputs={},
        outputs={},
        effective={},
        fingerprint={},
        output_dir=None,
        state_path=None,
        success_marker=None,
        execute=lambda _resume: {"skipped": True, "reason": reason},
    )


def inspect_stage_plan(plan: ResolvedStagePlan) -> dict[str, Any]:
    payload = plan.as_dict()
    if not plan.enabled:
        payload["status"] = "disabled"
        payload["action"] = "skip"
        return payload

    preflight_error = None
    try:
        plan.preflight()
    except Exception as exc:  # pragma: no cover - exercised via callers
        preflight_error = f"{type(exc).__name__}: {exc}"

    output_dir = plan.output_dir
    state_path = plan.state_path
    success_marker = plan.success_marker

    if preflight_error is not None:
        payload["status"] = "blocked"
        payload["action"] = "blocked"
        payload["error"] = preflight_error
        return payload

    if output_dir is None:
        payload["status"] = "disabled"
        payload["action"] = "skip"
        return payload

    if success_marker is not None and success_marker.is_file() and state_path is not None and state_path.is_file():
        on_disk = read_prepare_state(state_path)
        drift = diff_fingerprint(plan.fingerprint, on_disk)
        if drift:
            payload["status"] = "dirty"
            payload["action"] = "rebuild"
            payload["drift"] = _serialize(
                {k: {"expected": exp, "actual": act} for k, (exp, act) in drift.items()}
            )
        else:
            payload["status"] = "ready"
            payload["action"] = "skip"
        return payload

    if output_dir.exists():
        payload["status"] = "partial"
        payload["action"] = "rebuild"
    else:
        payload["status"] = "missing"
        payload["action"] = "build"
    return payload


def clean_stage_plan(plan: ResolvedStagePlan) -> dict[str, Any]:
    if not plan.enabled or plan.output_dir is None:
        return {"skipped": True, "reason": plan.reason or f"{plan.stage}.disabled"}

    if not plan.output_dir.exists():
        return {"skipped": True, "reason": f"{plan.stage}.output_absent", "output_dir": str(plan.output_dir)}

    if plan.output_dir.is_dir():
        shutil.rmtree(plan.output_dir)
    else:
        plan.output_dir.unlink()
    return {"removed": True, "output_dir": str(plan.output_dir)}
