"""Resolved-plan convert stage adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audio_tokenization.config.schema import DatasetSpec, PrepareSpec
from audio_tokenization.contracts.artifacts import SHAR_INDEX_FILENAME, SUCCESS_MARKER_FILE
from audio_tokenization.prepare.runtime import (
    get_prepare_runner,
    preflight_prepare_spec,
    resolve_num_workers,
    resolve_prepare_inputs,
)
from audio_tokenization.stages._plans import (
    ResolvedStagePlan,
    disabled_stage_plan,
)
from audio_tokenization.stages._stage_runner import (
    MANIFEST_FILE,
    check_stage_output,
    run_stage,
)


logger = logging.getLogger(__name__)


def resolve_convert_plan(spec: DatasetSpec) -> ResolvedStagePlan:
    if spec.convert is None or not spec.convert.enabled:
        return disabled_stage_plan(stage="convert", reason="convert.disabled")

    prepare_spec = _resolve_prepare_spec(spec.convert)
    resolved_inputs, input_summary = resolve_prepare_inputs(prepare_spec)
    output_dir = Path(prepare_spec.output.shar_dir)
    success_marker = output_dir / SUCCESS_MARKER_FILE

    return ResolvedStagePlan(
        stage="convert",
        enabled=True,
        reason=None,
        inputs=input_summary,
        outputs={
            "shar_dir": str(output_dir),
            "manifest": str(output_dir / MANIFEST_FILE),
            "success_marker": str(success_marker),
        },
        effective=_effective_convert_values(prepare_spec, resolved_inputs),
        fingerprint=prepare_spec.fingerprint_payload(),
        output_dir=output_dir,
        success_marker=success_marker,
        preflight=lambda: preflight_prepare_spec(prepare_spec, resolved_inputs=resolved_inputs),
        execute=lambda overwrite: _execute_convert_plan(
            prepare_spec,
            resolved_inputs=resolved_inputs,
            overwrite=overwrite,
        ),
    )


def run_convert(spec: DatasetSpec, *, overwrite: bool = False) -> dict[str, Any]:
    if spec.convert is None:
        raise ValueError(
            "stage=convert requested but DatasetSpec has no convert section. "
            "Add outputs.shar_dir to the dataset YAML to enable this stage."
        )
    if not spec.convert.enabled:
        return {"skipped": True, "reason": "convert.disabled"}

    # Skip BEFORE plan resolution so a completed convert can be reused on
    # nodes that no longer have the raw inputs mounted or the prepare-time
    # runtime deps (ffmpeg, polars, ...) installed. Plan resolution eagerly
    # globs raw inputs and preflight loads the text tokenizer; both fail
    # loudly in those environments and would block legitimate skips.
    shar_dir = Path(spec.convert.output.shar_dir)
    skipped = check_stage_output(stage="convert", output_dir=shar_dir, overwrite=overwrite)
    if skipped is not None:
        _ensure_convert_shar_manifest(spec.convert)
        return skipped

    plan = resolve_convert_plan(spec)
    return plan.execute(overwrite)


def _resolve_prepare_spec(spec: PrepareSpec) -> PrepareSpec:
    # The lhotse_recipe runner does ``range(num_workers)`` directly, so it
    # needs a concrete int when None comes through. Hardcoded to 64 instead of
    # the SLURM-aware resolver so the plan stays stable across nodes with
    # different ``SLURM_CPUS_PER_TASK``. Other families resolve None inside the
    # runner because their worker count only affects execution.
    if spec.family != "lhotse_recipe" or spec.output.num_workers is not None:
        return spec
    return spec.model_copy(
        update={"output": spec.output.model_copy(update={"num_workers": 64})},
    )


def _effective_convert_values(spec: PrepareSpec, resolved_inputs: list[str]) -> dict[str, Any]:
    # Empty input list (lhotse_recipe) means no fan-out cap; only SLURM/CPU applies.
    num_inputs = len(resolved_inputs) or None
    return {
        "family": spec.family,
        "effective_num_workers": resolve_num_workers(spec.output.num_workers, num_inputs=num_inputs),
        "shard_size": spec.output.shard_size,
        "target_sr": spec.output.target_sr,
        "shar_format": spec.output.shar_format,
        "text_tokenizer": spec.output.text_tokenizer,
        "resampling_backend": spec.output.resampling_backend,
        "resolved_input_count": len(resolved_inputs),
    }


def _execute_convert_plan(
    spec: PrepareSpec,
    *,
    resolved_inputs: list[str] | None = None,
    overwrite: bool,
) -> dict[str, Any]:
    shar_dir = Path(spec.output.shar_dir)
    runner_module = get_prepare_runner(spec)

    def _work() -> dict[str, Any]:
        result = runner_module.run(spec, resolved_inputs=resolved_inputs) or {}
        _ensure_convert_shar_manifest(spec)
        return result

    return run_stage(
        stage="convert",
        output_dir=shar_dir,
        fingerprint=spec.fingerprint_payload(),
        work=_work,
        overwrite=overwrite,
        logger=logger,
        preflight=lambda: runner_module.preflight(spec, resolved_inputs=resolved_inputs),
    )


def _ensure_convert_shar_manifest(spec: PrepareSpec) -> None:
    """Backfill the durable SHAR manifest after convert success or skip.

    Prepare runners are still family-specific, so the stage adapter owns the
    cross-family handoff contract: every completed SHAR root should expose
    ``_shar_work_manifest.json`` for tokenization planning, whether the run was
    fresh or skipped because ``_SUCCESS`` already existed.
    """

    from audio_tokenization.pipelines.lhotse.planning import (
        SHAR_WORK_MANIFEST_FILE,
        write_shar_work_manifest,
    )

    shar_dir = Path(spec.output.shar_dir)
    manifest_path = shar_dir / SHAR_WORK_MANIFEST_FILE
    if manifest_path.is_file():
        return
    manifest = write_shar_work_manifest(
        shar_dir,
        index_name=_convert_shar_index_filename(spec),
    )
    logger.info(
        "Wrote SHAR work manifest at %s: work_units=%s hours=%.2f",
        manifest_path,
        len(manifest.work_units),
        sum(unit.duration_sec for unit in manifest.work_units) / 3600.0,
    )


def _convert_shar_index_filename(spec: PrepareSpec) -> str:
    if spec.family == "lhotse_recipe":
        return spec.input.shar_index_filename
    return SHAR_INDEX_FILENAME
