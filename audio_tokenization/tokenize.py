#!/usr/bin/env python3
"""Main entry point for audio tokenization pipeline.

Usage:
    python -m audio_tokenization.tokenize
    python -m audio_tokenization.tokenize dataset=audioset_lhotse
"""

# Avoid thread oversubscription with many dataloader workers
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _validate_mode_from_dataset_folder(cfg: DictConfig) -> None:
    """Fail fast if dataset folder semantics conflict with selected mode.

    If dataset is chosen from a nested Hydra group like ``dataset=audio_text/foo``,
    we treat the first segment (``audio_text`` / ``audio_only``) as the expected
    mode and verify it matches ``cfg.mode``.
    """
    mode = cfg.get("mode", "audio_only")

    dataset_choice = None
    try:
        dataset_choice = HydraConfig.get().runtime.choices.get("dataset")
    except Exception:
        # Hydra runtime info may be unavailable in some test harnesses.
        return

    if not dataset_choice or "/" not in dataset_choice:
        return

    folder_mode = dataset_choice.split("/", 1)[0]
    if folder_mode not in {"audio_only", "audio_text"}:
        return

    if mode != folder_mode:
        raise ValueError(
            f"Mode mismatch: dataset={dataset_choice!r} implies mode={folder_mode!r} "
            f"from folder name, but cfg.mode={mode!r}. "
            f"Set mode={folder_mode} or pick a dataset under {mode}/."
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print config only on rank 0
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    _validate_mode_from_dataset_folder(cfg)

    tokenizer_cfg = cfg.get("tokenizer", {})
    tokenizer_path = tokenizer_cfg.get("path") if tokenizer_cfg else None
    tokenizer_sampling_rate = tokenizer_cfg.get("sampling_rate") if tokenizer_cfg else None
    torch_compile = tokenizer_cfg.get("torch_compile", True) if tokenizer_cfg else True
    trim_last_tokens = tokenizer_cfg.get("trim_last_tokens", 5) if tokenizer_cfg else 5
    if tokenizer_path is None:
        tokenizer_path = cfg.get("tokenizer_path")

    min_sample_rate = cfg.get("min_sample_rate")

    from audio_tokenization.pipelines.lhotse import run_lhotse_pipeline

    # Build flat config dict for the unified Lhotse pipeline (DDP, no Ray)
    pipeline_cfg = {
        "tokenizer_path": tokenizer_path,
        "target_sample_rate": tokenizer_sampling_rate,
        "torch_compile": torch_compile,
        "trim_last_tokens": trim_last_tokens,
        "output_dir": cfg.output_dir,
        "mode": cfg.get("mode", "audio_only"),
        "resume": cfg.get("resume", False),
        "min_duration": cfg.dataset.get("min_duration", cfg.filter.min_duration),
        "max_duration": cfg.dataset.get("max_duration", cfg.filter.max_duration),
        "min_sample_rate": cfg.dataset.get("min_sample_rate", cfg.filter.min_sample_rate),
        "min_rms_db": cfg.dataset.get("min_rms_db", cfg.filter.min_rms_db),
        "normalize_rms_db": cfg.dataset.get("normalize_rms_db", cfg.filter.normalize_rms_db),
        # Shar data (pre-built by prepare_hf_to_shar / prepare_wds_to_shar)
        "shar_dir": list(cfg.dataset.shar_dir) if isinstance(cfg.dataset.shar_dir, (list, ListConfig)) else cfg.dataset.shar_dir,
        "shar_index_filename": cfg.dataset.get("shar_index_filename", "shar_index.json"),
        # Dynamic bucketing sampler
        "max_batch_duration": cfg.dataset.get("max_batch_duration", cfg.dataloader.max_batch_duration),
        "max_batch_cuts": cfg.dataset.get("max_batch_cuts"),
        "num_buckets": cfg.dataset.get("num_buckets", 20),
        "bucket_buffer_size": cfg.dataset.get("bucket_buffer_size", 20000),
        "sampler_shuffle": cfg.dataset.get("sampler_shuffle", True),
        "sampler_seed": cfg.dataset.get("sampler_seed", 42),
        "quadratic_duration": cfg.dataset.get("quadratic_duration"),
        # DataLoader prefetching
        "num_workers": cfg.dataset.get("num_workers", cfg.dataloader.num_workers),
        "prefetch_factor": cfg.dataset.get("prefetch_factor", cfg.dataloader.prefetch_factor),
        # Checkpointing
        "checkpoint_interval_batches": cfg.dataset.get("checkpoint_interval_batches", cfg.dataloader.checkpoint_interval_batches),
        # Output subdirectory name (overrides auto-inferred name)
        "output_name": cfg.dataset.get("output_name"),
        # audio_text mode
        "clip_id_parser": cfg.dataset.get("clip_id_parser", "generic"),
        "dataset_name": cfg.dataset.get("dataset_name", cfg.dataset.get("output_name", "")),
        "audio_text_format": cfg.get("audio_text_format", "interleaved"),
        "audio_text_task": cfg.get("audio_text_task", "transcribe"),
        # W&B
        "wandb": OmegaConf.to_container(cfg.get("wandb", {}), resolve=True),
    }

    result = run_lhotse_pipeline(pipeline_cfg)

    logger.info("Pipeline completed!")
    logger.info(f"Total processed: {result.get('samples_processed', 0)}")
    logger.info(f"Total tokens: {result.get('tokens_generated', 0)}")
    logger.info(f"Output directory: {result.get('output_dir', cfg.output_dir)}")

    return result


if __name__ == "__main__":
    main()
