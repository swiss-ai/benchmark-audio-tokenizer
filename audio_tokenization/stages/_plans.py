"""Resolved stage-plan primitives for the unified control plane.

The key idea: commands and runners should operate on the same resolved
object, not re-derive defaults / paths / fingerprints in multiple places.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any, Callable


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
    success_marker: Path | None
    reason: str | None = None
    state_path: Path | None = None  # deprecated; kept for caller compatibility during refactor
    preflight: Callable[[], None] = field(default=_noop_preflight, repr=False)
    execute: Callable[[bool], dict[str, Any]] = field(default=lambda _overwrite: {}, repr=False)

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
                "success_marker": _serialize(self.success_marker),
            },
        }


def _disabled_execute_raises(_overwrite):
    raise AssertionError(
        "disabled_stage_plan.execute is not reachable from run_*; "
        "use inspect_stage_plan for inspection paths."
    )


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
        success_marker=None,
        state_path=None,
        execute=_disabled_execute_raises,
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

    if success_marker is not None and success_marker.is_file():
        payload["status"] = "ready"
        payload["action"] = "skip"
    elif output_dir.exists():
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
