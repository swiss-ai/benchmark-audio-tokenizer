#!/usr/bin/env python3
"""Unified entrypoint for the audio pipeline.

Examples::

    python -m audio_tokenization run dataset=infore2 stage=convert
    python -m audio_tokenization plan dataset=infore2 stage=all
    python -m audio_tokenization status dataset=infore2 stage=tokenize
    python -m audio_tokenization clean dataset=infore2 stage=materialize
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from audio_tokenization.config import load_dataset_spec
from audio_tokenization.stages import clean_stages, plan_stages, run_stages, status_stages


logger = logging.getLogger(__name__)

Command = Literal["run", "plan", "status", "clean"]
_COMMANDS: tuple[str, ...] = ("run", "plan", "status", "clean")


def _split_command(argv: list[str]) -> tuple[Command, list[str]]:
    if argv and argv[0] in _COMMANDS:
        return argv[0], argv[1:]
    return "run", argv


def _compose_pipeline_cfg(overrides: list[str]):
    config_dir = Path(__file__).resolve().parent / "configs" / "pipeline"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config", overrides=overrides)


def _execute_command(command: Command, cfg) -> dict[str, Any]:
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info("Pipeline config:\n%s", OmegaConf.to_yaml(cfg))

    spec = load_dataset_spec(cfg.dataset)
    stage = cfg.get("stage", "all")
    runtime = cfg.get("runtime") or {}
    resume = bool(runtime.get("resume", True))

    if command == "run":
        stages = run_stages(spec, stage=stage, resume=resume)
    elif command == "plan":
        stages = plan_stages(spec, stage=stage)
    elif command == "status":
        stages = status_stages(spec, stage=stage)
    elif command == "clean":
        stages = clean_stages(spec, stage=stage)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command {command!r}")

    return {
        "command": command,
        "dataset": spec.name,
        "stage": stage,
        "resume": resume,
        "stages": stages,
    }


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command, overrides = _split_command(argv)
    result = _execute_command(command, _compose_pipeline_cfg(overrides))
    if command != "run":
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
