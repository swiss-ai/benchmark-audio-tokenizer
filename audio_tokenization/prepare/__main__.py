#!/usr/bin/env python3
"""Config-driven prepare entrypoint."""

from __future__ import annotations

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from audio_tokenization.config import load_dataset_spec
from audio_tokenization.stages import run_prepare


logger = logging.getLogger(__name__)


def run_from_cfg(cfg: DictConfig):
    spec = load_dataset_spec(cfg.dataset)
    return run_prepare(spec)


@hydra.main(version_base=None, config_path="../configs/pipeline", config_name="config")
def main(cfg: DictConfig):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(f"Pipeline config:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.get("stage", "prepare") != "prepare":
        raise ValueError(
            f"audio_tokenization.prepare only supports stage='prepare'; got {cfg.get('stage')!r}"
        )
    return run_from_cfg(cfg)


if __name__ == "__main__":
    main()
