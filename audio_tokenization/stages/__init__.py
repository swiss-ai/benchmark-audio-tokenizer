"""Stage adapters for the unified audio-pipeline entrypoint.

Per-function stage semantics:

* ``run_stages``: stage is a required single name. Multi-stage is not supported
  because ``convert`` (CPU/IO-bound), ``tokenize`` (GPU-distributed) and
  ``materialize`` (CPU-only) have different cluster resource profiles and
  cannot share a Slurm allocation — each gets its own job.
* ``plan_stages`` / ``status_stages``: stage is a single name or ``None``
  (inspects all three).
* ``clean_stages``: stage is a required single name (destructive — never
  defaulted).
"""

from __future__ import annotations

from typing import Any, Callable

from audio_tokenization.config.schema import DatasetSpec
from ._plans import ResolvedStagePlan, clean_stage_plan, inspect_stage_plan
from .convert import resolve_convert_plan, run_convert
from .materialize import resolve_materialize_plan, run_materialize
from .tokenize import resolve_tokenize_plan, run_tokenize


_STAGES: tuple[str, ...] = ("convert", "tokenize", "materialize")


_STAGE_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "convert": run_convert,
    "tokenize": run_tokenize,
    "materialize": run_materialize,
}
_STAGE_RESOLVE: dict[str, Callable[[DatasetSpec], ResolvedStagePlan]] = {
    "convert": resolve_convert_plan,
    "tokenize": resolve_tokenize_plan,
    "materialize": resolve_materialize_plan,
}


def _check_known(stage: str) -> str:
    if stage not in _STAGES:
        raise ValueError(f"Unknown stage {stage!r}; valid: {list(_STAGES)}")
    return stage


def _require_single_stage(stage: str | None, *, command: str) -> str:
    if not stage:
        raise ValueError(
            f"{command} requires stage=<convert|tokenize|materialize> (got {stage!r})"
        )
    return _check_known(stage)


def _inspection_stages(stage: str | None) -> tuple[str, ...]:
    if not stage:
        return _STAGES
    return (_check_known(stage),)


def run_stages(
    spec: DatasetSpec,
    *,
    stage: str | None,
    overwrite: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run exactly one pipeline stage for *spec*."""
    name = _require_single_stage(stage, command="run")
    return {name: _STAGE_DISPATCH[name](spec, overwrite=overwrite)}


def plan_stages(
    spec: DatasetSpec,
    *,
    stage: str | None = None,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for name in _inspection_stages(stage):
        try:
            payload[name] = inspect_stage_plan(_STAGE_RESOLVE[name](spec))
        except Exception as exc:
            payload[name] = {
                "stage": name,
                "enabled": True,
                "status": "blocked",
                "action": "blocked",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return payload


def status_stages(
    spec: DatasetSpec,
    *,
    stage: str | None = None,
) -> dict[str, dict[str, Any]]:
    return plan_stages(spec, stage=stage)


def clean_stages(
    spec: DatasetSpec,
    *,
    stage: str | None,
) -> dict[str, dict[str, Any]]:
    name = _require_single_stage(stage, command="clean")
    plan = _STAGE_RESOLVE[name](spec)
    return {name: clean_stage_plan(plan)}


__all__ = [
    "ResolvedStagePlan",
    "plan_stages",
    "status_stages",
    "clean_stages",
    "run_convert",
    "run_tokenize",
    "run_materialize",
    "run_stages",
]
