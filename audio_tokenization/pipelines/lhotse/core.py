#!/usr/bin/env python3
"""Shared tokenization loop infrastructure.

Architecture (3 files per mode):
    core.py       -- shared: setup, loop skeleton, run_lhotse_pipeline entry point
    audio_only.py -- AudioOnlyHandler (Megatron indexed dataset output)
    audio_text.py -- AudioTextHandler (Parquet cache output)

Launch examples::

    # Single node, 4 GPUs
    srun --ntasks-per-node=4 --gpus-per-node=4 \\
        python -m audio_tokenization.tokenize dataset=peoples_speech_lhotse

    # Multi-node SLURM -- srun spawns all ranks directly (no torchrun, no NCCL)
    srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --kill-on-bad-exit=0 \\
        python -m audio_tokenization.tokenize dataset=peoples_speech_lhotse

    # Resume from checkpoint
    srun --ntasks-per-node=4 --gpus-per-node=4 \\
        python -m audio_tokenization.tokenize dataset=peoples_speech_lhotse resume=true
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch

from .checkpoint import (
    WorkerStats,
    _get_rss_gb,
    is_cuda_oom,
    load_checkpoint,
    save_checkpoint,
    SimpleWandbLogger,
)
from .data import build_cutset

logger = logging.getLogger(__name__)


def _build_sampler_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build DynamicBucketingSampler kwargs from config."""
    sampler_kwargs = dict(
        max_duration=cfg.get("max_batch_duration", 1500.0),
        num_buckets=cfg.get("num_buckets", 20),
        buffer_size=cfg.get("bucket_buffer_size", 20000),
        shuffle=cfg.get("sampler_shuffle", True),
        seed=cfg.get("sampler_seed", 42),
        world_size=1,
        rank=0,
        drop_last=False,
    )

    max_cuts = cfg.get("max_batch_cuts")
    if max_cuts is not None:
        sampler_kwargs["max_cuts"] = max_cuts

    quadratic_duration = cfg.get("quadratic_duration")
    if quadratic_duration is not None:
        sampler_kwargs["quadratic_duration"] = quadratic_duration

    duration_bins = cfg.get("duration_bins")
    if duration_bins is not None:
        sampler_kwargs["duration_bins"] = list(duration_bins)

    num_cuts_for_bins_estimate = cfg.get("num_cuts_for_bins_estimate")
    if num_cuts_for_bins_estimate is not None:
        sampler_kwargs["num_cuts_for_bins_estimate"] = int(num_cuts_for_bins_estimate)

    if "sampler_concurrent" in cfg:
        sampler_kwargs["concurrent"] = bool(cfg.get("sampler_concurrent"))

    if "sampler_sync_buckets" in cfg:
        sampler_kwargs["sync_buckets"] = bool(cfg.get("sampler_sync_buckets"))

    return sampler_kwargs


def _format_writer_state(writer_state: Any) -> str:
    if isinstance(writer_state, dict):
        items = ", ".join(f"{k}={v}" for k, v in sorted(writer_state.items()))
        return "{" + items + "}"
    return str(writer_state)


def _normalize_batch(batch: dict, target_db: float, device: str = "cpu") -> dict:
    """Peak-normalize audio in a batch dict (works for all handler modes).

    Moves audio to *device* before normalizing for GPU-accelerated peak
    computation.  Matches WavTokenizer training: SOX ``norm`` to *target_db* dBFS.
    """
    from audio_tokenization.utils.prepare_data.audio_ops import normalize_batch_peak

    if "audio" in batch:
        batch["audio"] = normalize_batch_peak(batch["audio"].to(device, non_blocking=True), target_db)
    elif "inputs" in batch:
        batch["inputs"] = normalize_batch_peak(batch["inputs"].to(device, non_blocking=True), target_db)
    return batch


# ---------------------------------------------------------------------------
# Main tokenization loop (per-rank)
# ---------------------------------------------------------------------------


def tokenize_loop(rank: int, world_size: int, cfg: Dict[str, Any], handler) -> Dict[str, Any]:
    """Main per-rank tokenization loop.

    Steps:
        1. Load prepared Shar CutSet -- see ``data.py``
        2. Create ``DynamicBucketingSampler`` with global bucketing
        3. Optionally resume from checkpoint
        4. Wrap in dataset + ``DataLoader`` for CPU/GPU overlap
        5. Loop over prefetched batches, tokenize on GPU, write output
        6. Periodically checkpoint (sampler state + chunk boundary)
    """
    from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler

    output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Clean up stale .tmp files from killed runs (e.g. OOM kill).
    for tmp in Path(output_dir).glob(f"rank_{rank:04d}_*.tmp"):
        logger.warning(f"[rank {rank}] Removing stale temp file: {tmp.name}")
        tmp.unlink()

    cumulative_stats = WorkerStats()

    # ------------------------------------------------------------------
    # 1. Build CutSet (prepared Shar load + filters/resample safety-net)
    # ------------------------------------------------------------------
    cuts = build_cutset(cfg, rank, world_size, stats=cumulative_stats)

    # ------------------------------------------------------------------
    # 2. Dynamic bucketing sampler -- each rank's CutSet is already split
    #    at the shard level (see data.py), so the sampler uses
    #    world_size=1 to avoid the O(world_size) strided distribution.
    # ------------------------------------------------------------------
    sampler_kwargs = _build_sampler_kwargs(cfg)
    max_duration = sampler_kwargs["max_duration"]
    num_buckets = sampler_kwargs["num_buckets"]
    buffer_size = sampler_kwargs["buffer_size"]
    sampler = DynamicBucketingSampler(cuts, **sampler_kwargs)

    # ------------------------------------------------------------------
    # 3. Resume from checkpoint -- sampler.load_state_dict() restores
    #    sampler state via metadata bookkeeping (no audio decoding), so
    #    recovery is typically fast.
    # ------------------------------------------------------------------
    resume = cfg.get("resume", False)
    start_writer_state: Any = 0

    if resume:
        ckpt = load_checkpoint(output_dir, rank)
        if ckpt is not None:
            ckpt_ws = ckpt.get("world_size")
            if ckpt_ws is not None and ckpt_ws != world_size:
                logger.warning(
                    f"[rank {rank}] Checkpoint world_size ({ckpt_ws}) != current "
                    f"world_size ({world_size}). Shard assignment changed — "
                    f"ignoring checkpoint, starting from scratch."
                )
                ckpt = None
        if ckpt is not None:
            sampler.load_state_dict(ckpt["sampler_state"])
            if "writer_state" in ckpt:
                start_writer_state = ckpt["writer_state"]
            else:
                start_writer_state = ckpt["chunk_id"] + 1
            prev = ckpt.get("stats", {})
            cumulative_stats.samples_processed = prev.get("samples_processed", 0)
            cumulative_stats.tokens_generated = prev.get("tokens_generated", 0)
            cumulative_stats.text_tokens_generated = prev.get("text_tokens_generated", 0)
            cumulative_stats.errors = prev.get("errors", 0)
            cumulative_stats.samples_skipped = prev.get("samples_skipped", 0)
            cumulative_stats.rms_skipped = prev.get("rms_skipped", 0)
            logger.info(
                f"[rank {rank}] Resumed from writer_state={_format_writer_state(start_writer_state)}, "
                f"samples={cumulative_stats.samples_processed}"
            )

    # ------------------------------------------------------------------
    # 4. DataLoader with prefetching -- worker subprocesses decode audio
    #    in parallel while the main thread runs GPU tokenization.
    # ------------------------------------------------------------------
    max_workers = os.cpu_count() // max(torch.cuda.device_count(), 1)
    num_workers = min(cfg.get("num_workers", 4), max_workers)
    prefetch_factor = cfg.get("prefetch_factor", 4)
    dataloader_timeout = cfg.get("dataloader_timeout", 300)  # 5 min default
    worker_init_fn = None
    if num_workers > 0:
        from lhotse.dataset.dataloading import make_worker_init_fn

        worker_init_fn = make_worker_init_fn(
            rank=rank,
            world_size=world_size,
            seed=cfg.get("sampler_seed", 42),
        )

    dataset = handler.create_dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        timeout=dataloader_timeout if num_workers > 0 else 0,
        worker_init_fn=worker_init_fn,
    )

    # ------------------------------------------------------------------
    # 5. Create tokenizer on GPU
    # ------------------------------------------------------------------
    from audio_tokenization.vokenizers import create_tokenizer

    device = f"cuda:{cfg.get('local_rank', 0)}"
    tokenizer_path = cfg["tokenizer_path"]
    mode = cfg.get("mode", "audio_only")
    torch_compile = cfg.get("torch_compile", True)
    target_sr = int(cfg.get("target_sample_rate", 24000))
    trim_last_tokens = cfg.get("trim_last_tokens", 5)

    tokenizer = create_tokenizer(
        omni_tokenizer_path=tokenizer_path,
        mode=mode,
        device=device,
        torch_compile=torch_compile,
        trim_last_tokens=trim_last_tokens,
    )

    # ------------------------------------------------------------------
    # 6. W&B logger (rank 0 only)
    # ------------------------------------------------------------------
    wandb_logger = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False) and rank == 0:
        # Auto-generate wandb run name: {task}/{dataset}_dur{min}-{max}
        _wandb_name = wandb_cfg.get("name")
        if not _wandb_name:
            _output_name = cfg.get("output_name", "unknown")
            _min_d = cfg.get("min_duration")
            _max_d = cfg.get("max_duration")
            _dur_tag = ""
            if _min_d is not None or _max_d is not None:
                _fmt = lambda v: str(int(v)) if v is not None and float(v).is_integer() else str(v).replace(".", "p") if v is not None else ""
                _dur_tag = f"_dur{_fmt(_min_d) or 'min'}-{_fmt(_max_d) or 'max'}"
            _wandb_name = f"{_build_output_subdir(cfg)}{_dur_tag}"

        wandb_logger = SimpleWandbLogger(
            project=wandb_cfg.get("project", "audio-tokenization"),
            entity=wandb_cfg.get("entity"),
            name=_wandb_name,
            tags=wandb_cfg.get("tags", []),
            config={
                "rank": rank,
                "world_size": world_size,
                "max_batch_duration": max_duration,
                "min_duration": cfg.get("min_duration"),
                "max_duration": cfg.get("max_duration"),
                "num_buckets": num_buckets,
                "buffer_size": buffer_size,
                "target_sample_rate": target_sr,
                **{k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
            },
            log_interval_seconds=wandb_cfg.get("log_interval_seconds", 10.0),
        )

    # ------------------------------------------------------------------
    # 7. Main loop -- tokenize batches, write output, checkpoint
    # ------------------------------------------------------------------
    checkpoint_interval = cfg.get("checkpoint_interval_batches", 500)
    writer_state = start_writer_state
    batch_count = 0

    handler.setup_writer(output_dir, rank, writer_state, tokenizer)

    stats = cumulative_stats
    total_audio_seconds = 0.0

    logger.info(
        f"[rank {rank}] Starting tokenization loop "
        f"(writer_state={_format_writer_state(writer_state)}, checkpoint_interval={checkpoint_interval})"
    )

    consecutive_errors = 0
    max_consecutive_errors = cfg.get("max_consecutive_errors", 50)
    _loop_error = None

    normalize_rms_db = cfg.get("normalize_rms_db")
    if normalize_rms_db is not None:
        normalize_rms_db = float(normalize_rms_db)
        logger.info(f"[rank {rank}] Peak normalization enabled: target {normalize_rms_db} dBFS")

    import time as _time

    _t_start = _t_encode_start = _t_encode_end = None
    if wandb_logger is not None:
        _t_start = torch.cuda.Event(enable_timing=True)
        _t_encode_start = torch.cuda.Event(enable_timing=True)
        _t_encode_end = torch.cuda.Event(enable_timing=True)

    _batch_ready_time = _time.monotonic()

    _pbar = None
    if rank == 0:
        from tqdm import tqdm
        _pbar = tqdm(desc="tokenize", unit=" batches", dynamic_ncols=True)

    try:
        for batch in dataloader:
            _dataloader_wait_ms = (_time.monotonic() - _batch_ready_time) * 1000

            # Decide whether to capture per-batch timing (only when W&B will flush).
            _time_this = (
                wandb_logger is not None and wandb_logger.should_log_now()
            )
            if _time_this:
                _t_start.record()

            try:
                _host_process_start = _time.monotonic()
                # Normalize audio volume before tokenization (all modes).
                if normalize_rms_db is not None:
                    batch = _normalize_batch(batch, normalize_rms_db, device)

                if _time_this:
                    _t_encode_start.record()

                batch_audio_secs = handler.process_batch(
                    batch, tokenizer, stats, target_sr, device,
                )

                if _time_this:
                    _t_encode_end.record()

                total_audio_seconds += batch_audio_secs
                _process_batch_wall_ms = (_time.monotonic() - _host_process_start) * 1000
                consecutive_errors = 0  # reset on batch success

            except Exception as batch_err:
                stats.errors += 1
                consecutive_errors += 1

                # CUDA OOM: free the failed allocation so the next batch can succeed.
                if is_cuda_oom(batch_err):
                    torch.cuda.empty_cache()
                    logger.warning(
                        f"[rank {rank}] CUDA OOM on batch {batch_count}, freed cache "
                        f"({consecutive_errors}/{max_consecutive_errors})"
                    )
                else:
                    logger.warning(
                        f"[rank {rank}] Batch error ({consecutive_errors}/{max_consecutive_errors}): "
                        f"{batch_err}"
                    )

                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"[rank {rank}] {max_consecutive_errors} consecutive batch errors, aborting"
                    ) from batch_err
                continue

            batch_count += 1

            if _pbar is not None:
                elapsed = _time.time() - stats.start_time
                tok_s = stats.tokens_generated / elapsed if elapsed > 0 else 0
                _pbar.set_postfix_str(
                    f"{stats.samples_processed} samples, {tok_s:.0f} tok/s, {stats.errors} err"
                )
                _pbar.update(1)

            # W&B log (rate-limited by interval inside logger)
            if wandb_logger is not None:
                metrics = None
                if _time_this:
                    _t_encode_end.synchronize()
                    tokenize_wall_ms = _t_start.elapsed_time(_t_encode_end)
                    tokenize_gpu_ms = _t_encode_start.elapsed_time(_t_encode_end)
                    metrics = {
                        "timing/dataloader_wait_ms": _dataloader_wait_ms,
                        "timing/tokenize_wall_ms": tokenize_wall_ms,
                        "timing/tokenize_gpu_ms": tokenize_gpu_ms,
                        "timing/process_batch_wall_ms": _process_batch_wall_ms,
                        "memory/rss_gb": _get_rss_gb(),
                        "memory/cuda_alloc_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                        "memory/cuda_reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
                    }
                wandb_logger.log(
                    samples=stats.samples_processed,
                    tokens=stats.tokens_generated,
                    errors=stats.errors,
                    skipped=stats.samples_skipped,
                    batch_audio_seconds=total_audio_seconds,
                    text_tokens=stats.text_tokens_generated,
                    metrics=metrics,
                )

            # Periodic checkpoint: finalize current chunk, save state, open next
            if batch_count % checkpoint_interval == 0 and handler.chunk_samples > 0:
                writer_state = handler.checkpoint_writer()
                logger.info(
                    f"[rank {rank}] Checkpointed writer state {_format_writer_state(writer_state)} "
                    f"({stats.tokens_generated} total tokens)"
                )

                save_checkpoint(
                    output_dir,
                    rank,
                    sampler_state=sampler.state_dict(),
                    writer_state=writer_state,
                    stats=stats.to_dict(),
                    world_size=world_size,
                )

            _batch_ready_time = _time.monotonic()

    except Exception as e:
        logger.error(f"[rank {rank}] Fatal error in tokenization loop: {e}", exc_info=True)
        stats.errors += 1
        _loop_error = e

    if _pbar is not None:
        _pbar.close()

    # ------------------------------------------------------------------
    # 8. Finalize last chunk (always save progress, even on failure)
    # ------------------------------------------------------------------
    handler.finalize_writer()

    save_checkpoint(
        output_dir,
        rank,
        sampler_state=sampler.state_dict(),
        writer_state=handler.get_writer_state(),
        stats=stats.to_dict(),
        world_size=world_size,
    )

    result = stats.finalize()
    result["rank"] = rank
    result["chunks_written"] = handler.chunks_written

    if wandb_logger is not None:
        wandb_logger.finish()

    text_tok_msg = ""
    if result.get("text_tokens_generated", 0) > 0:
        text_tok_msg = f", {result['text_tokens_generated']} text tokens"
    rms_msg = ""
    if result.get("rms_skipped", 0) > 0:
        rms_msg = f", {result['rms_skipped']} rms_skipped"
    logger.info(
        f"[rank {rank}] Done: {result['samples_processed']} samples, "
        f"{result['tokens_generated']} audio tokens{text_tok_msg}, "
        f"{result['errors']} errors{rms_msg}, {result['elapsed_time']:.1f}s"
    )

    result["output_dir"] = output_dir

    try:
        from .stats_reducer import write_rank_stats, maybe_write_stats_summary

        write_rank_stats(output_dir, result)
        summary = maybe_write_stats_summary(output_dir, expected_ranks=world_size)
        if summary is not None:
            logger.info(
                f"[rank {rank}] Stats summary: {summary['samples_processed']:,} samples, "
                f"{summary['audio_tokens']:,} audio tokens, "
                f"{summary['text_tokens']:,} text tokens across {summary['num_ranks']} ranks"
            )
    except Exception:
        logger.warning(f"[rank {rank}] Failed to write stats JSON", exc_info=True)

    # Re-raise after saving progress so the exit code signals failure.
    # This MUST come after all cleanup (checkpoint, metadata, wandb) so
    # partial work is never silently lost.
    if _loop_error is not None:
        raise RuntimeError(
            f"[rank {rank}] Tokenization loop failed after processing "
            f"{result['samples_processed']} samples"
        ) from _loop_error

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_output_subdir(cfg: Dict[str, Any]) -> str:
    """Build a dataset-specific subdirectory path.

    Layout::

        audio_only:              audio_only/{output_name}
        audio_text (direct):     {task}/{output_name}
        audio_text (interleaved): interleave_cache/{output_name}

    Where task is: transcribe, translate, annotate.
    Interleave stage 2 (pattern build) writes to interleave/{output_name}
    separately via build_interleaved.
    """
    output_name = cfg.get("output_name")
    if not output_name:
        raise ValueError("'output_name' is required in the dataset config.")

    mode = cfg.get("mode", "audio_only")

    if mode == "audio_text":
        fmt = cfg.get("audio_text_format", "direct")
        if fmt == "interleaved":
            return str(Path("interleave_cache") / output_name)
        task = cfg.get("audio_text_task", "transcribe")
        return str(Path(task) / output_name)

    return str(Path("audio_only") / output_name)



def run_lhotse_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the Lhotse tokenization pipeline.

    Expects pre-built Shar data (via ``prepare_hf_to_shar`` or
    ``prepare_wds_to_shar``).  Loads the Shar CutSet, tokenizes on GPU,
    and writes micro-shards with DDP checkpointing.
    """
    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK.
    # srun (without torchrun) sets SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID.
    num_gpus = cfg.get("num_gpus")
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    # Cross-check: if num_gpus is set, verify it matches the env-derived world_size.
    if num_gpus is not None:
        num_gpus = int(num_gpus)
        if world_size == 1 and num_gpus > 1:
            # No env vars set (local run) — use num_gpus as world_size.
            world_size = num_gpus
            logger.warning(
                f"No SLURM/torchrun env vars detected, using num_gpus={num_gpus} as world_size"
            )
        elif world_size != num_gpus:
            raise RuntimeError(
                f"num_gpus={num_gpus} from config does not match "
                f"world_size={world_size} from environment. "
                f"Check SLURM --ntasks-per-node * --nodes matches num_gpus."
            )

    # Safety: infer LOCAL_RANK from global rank + GPUs per node if env vars
    # are missing (e.g. bare srun without torchrun on multi-GPU nodes).
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" not in os.environ:
        gpus_per_node = torch.cuda.device_count()
        if gpus_per_node > 0:
            local_rank = rank % gpus_per_node
            logger.warning(
                f"[rank {rank}] LOCAL_RANK not set, inferred {local_rank} "
                f"from rank % {gpus_per_node} GPUs"
            )

    # Only rank 0 logs at INFO; others at WARNING to avoid 160x noise.
    if rank != 0:
        logging.getLogger("audio_tokenization").setLevel(logging.WARNING)
        logging.getLogger("lhotse").setLevel(logging.WARNING)

    cfg["rank"] = rank
    cfg["world_size"] = world_size
    cfg["local_rank"] = local_rank

    # Namespace tokenization output to avoid checkpoint collisions.
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / _build_output_subdir(cfg))
    torch.cuda.set_device(local_rank)

    logger.info(
        f"[rank {rank}/{world_size}] starting (local_rank={local_rank}, "
        f"no NCCL — each rank is independent)"
    )

    mode = cfg.get("mode", "audio_only")
    if mode == "audio_only":
        from .audio_only import AudioOnlyHandler
        handler = AudioOnlyHandler(cfg)
    elif mode == "audio_text":
        from .audio_text import AudioTextHandler
        handler = AudioTextHandler(cfg)
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    return tokenize_loop(rank, world_size, cfg, handler)
