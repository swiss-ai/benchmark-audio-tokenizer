"""Stage adapters for the unified audio-pipeline entrypoint."""

from __future__ import annotations

from typing import Any, Callable, Literal

from audio_tokenization.config.schema import DatasetSpec
from ._plans import ResolvedStagePlan, clean_stage_plan, inspect_stage_plan
from .convert import resolve_convert_plan, run_convert
from .materialize import resolve_materialize_plan, run_materialize
from .tokenize import resolve_tokenize_plan, run_tokenize


Stage = Literal["convert", "tokenize", "materialize", "all"]
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


def _requested_stages(stage: Stage) -> tuple[str, ...]:
    if stage not in _STAGES and stage != "all":
        raise ValueError(
            f"Unknown stage {stage!r}; expected one of "
            f"{_STAGES + ('all',)}"
        )
    return _STAGES if stage == "all" else (stage,)


def resolve_stage_plans(
    spec: DatasetSpec,
    *,
    stage: Stage = "all",
) -> dict[str, ResolvedStagePlan]:
    requested = _requested_stages(stage)
    return {s: _STAGE_RESOLVE[s](spec) for s in requested}


def run_stages(
    spec: DatasetSpec,
    *,
    stage: Stage = "all",
    resume: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run one or all pipeline stages for *spec*.

    ``stage="all"`` runs convert → tokenize → materialize in order. Each
    stage may raise if its input prerequisite is missing — the caller
    must address that (e.g. run ``stage=convert`` first), not skip
    silently.
    """
    requested = _requested_stages(stage)
    return {s: _STAGE_DISPATCH[s](spec, resume=resume) for s in requested}


def plan_stages(
    spec: DatasetSpec,
    *,
    stage: Stage = "all",
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for name in _requested_stages(stage):
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
    stage: Stage = "all",
) -> dict[str, dict[str, Any]]:
    return plan_stages(spec, stage=stage)


def clean_stages(
    spec: DatasetSpec,
    *,
    stage: Stage = "all",
) -> dict[str, dict[str, Any]]:
    plans = resolve_stage_plans(spec, stage=stage)
    order = tuple(reversed(list(plans))) if stage == "all" else tuple(plans)
    return {name: clean_stage_plan(plans[name]) for name in order}


__all__ = [
    "Stage",
    "ResolvedStagePlan",
    "resolve_stage_plans",
    "plan_stages",
    "status_stages",
    "clean_stages",
    "run_convert",
    "run_tokenize",
    "run_materialize",
    "run_stages",
]
