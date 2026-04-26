#!/usr/bin/env python3
"""Shared tokenization loop infrastructure.

Architecture (3 files per mode):
    core.py       -- shared: setup, loop skeleton, run_lhotse_pipeline entry point
    audio_only.py -- AudioOnlyHandler (Megatron indexed dataset output)
    audio_text.py -- AudioTextHandler (Parquet cache output)

Launch through the unified stage graph::

    srun --ntasks-per-node=4 --gpus-per-node=4 --kill-on-bad-exit=0 \\
        python -m audio_tokenization run dataset=stage1_suno_s1 stage=tokenize
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence, assert_never

import torch
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.output_layout import (
    build_tokenize_output_subdir,
    resolve_tokenize_output_name,
)
from audio_tokenization.utils.io import cleanup_tmp_files

from .checkpoint import (
    WorkerStats,
    _get_rss_gb,
    is_cuda_oom,
    SimpleWandbLogger,
)
from .data import build_cutset

logger = logging.getLogger(__name__)


def _format_duration_tag_value(value: Any) -> str:
    if value is None:
        return ""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(value).replace(".", "p")


def _build_sampler_kwargs(spec: TokenizeSpec) -> dict[str, Any]:
    """Build DynamicBucketingSampler kwargs from config."""
    dataloader = spec.dataloader
    sampler_kwargs = dict(
        max_duration=dataloader.max_batch_duration,
        num_buckets=dataloader.num_buckets,
        buffer_size=dataloader.bucket_buffer_size,
        shuffle=dataloader.sampler_shuffle,
        seed=dataloader.sampler_seed,
        world_size=1,
        rank=0,
        drop_last=False,
    )

    if dataloader.max_batch_cuts is not None:
        sampler_kwargs["max_cuts"] = dataloader.max_batch_cuts

    if dataloader.quadratic_duration is not None:
        sampler_kwargs["quadratic_duration"] = dataloader.quadratic_duration

    return sampler_kwargs


def _cutset_len(cuts) -> int | None:
    try:
        return len(cuts)
    except TypeError:
        return None


def _cap_sampler_buckets_to_cut_count(
    sampler_kwargs: dict[str, Any],
    *,
    cut_count: int | None,
    rank: int,
) -> dict[str, Any]:
    """Keep Lhotse bucketing valid for tiny rank-local assignments."""
    if cut_count is None or cut_count <= 0:
        return sampler_kwargs
    num_buckets = int(sampler_kwargs["num_buckets"])
    if num_buckets <= cut_count:
        return sampler_kwargs
    sampler_kwargs = dict(sampler_kwargs)
    sampler_kwargs["num_buckets"] = cut_count
    logger.warning(
        "[rank %s] Capping num_buckets from %s to %s because only %s cuts "
        "are assigned to this rank.",
        rank,
        num_buckets,
        cut_count,
        cut_count,
    )
    return sampler_kwargs


def _flatten_wandb_config(prefix: str, payload: Any) -> dict[str, Any]:
    """Flatten resolved config into W&B config parameters, not metrics."""
    if isinstance(payload, dict):
        flattened: dict[str, Any] = {}
        for key, value in payload.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_wandb_config(child_key, value))
        return flattened
    return {prefix: payload}


def _build_wandb_config(
    spec: TokenizeSpec,
    *,
    dataset_name: str,
    input_shar_dirs: Sequence[str],
    final_output_dir: Path,
    rank: int,
    world_size: int,
    local_rank: int,
    assigned_cut_count: int | None,
    sampler_kwargs: dict[str, Any],
    effective_num_workers: int,
    effective_prefetch_factor: int | None,
    max_workers_per_rank: int,
    dataloader_timeout: int,
    output_name: str,
) -> dict[str, Any]:
    """Build static W&B run config from resolved Hydra/spec values."""
    resolved = {
        "dataset_name": dataset_name,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "input_shar_dirs": list(input_shar_dirs),
        "final_output_dir": str(final_output_dir),
        "assigned_cut_count": assigned_cut_count,
        "output_name": output_name,
        "tokenize": spec.model_dump(mode="json"),
        "effective": {
            "dataloader": {
                "num_workers": effective_num_workers,
                "prefetch_factor": effective_prefetch_factor,
                "timeout": dataloader_timeout,
                "persistent_workers": effective_num_workers > 0,
                "pin_memory": True,
                "max_workers_per_rank": max_workers_per_rank,
            },
            "sampler": dict(sampler_kwargs),
        },
    }
    config = _flatten_wandb_config("", resolved)
    # Preserve legacy top-level keys used by older W&B dashboards.
    config.update(
        {
            "max_batch_duration": sampler_kwargs["max_duration"],
            "min_duration": spec.filter.min_duration,
            "max_duration": spec.filter.max_duration,
            "num_buckets": sampler_kwargs["num_buckets"],
            "buffer_size": sampler_kwargs["buffer_size"],
            "target_sample_rate": spec.tokenizer.sampling_rate,
            "mode": spec.mode,
            "audio_text_format": spec.audio_text_format,
            "audio_text_task": spec.audio_text_task,
        }
    )
    return config


def _build_wandb_tags(
    configured_tags: Sequence[str],
    *,
    dataset_name: str,
    spec: TokenizeSpec,
) -> list[str]:
    """Stable W&B tags for filtering tokenization runs."""
    automatic = [
        f"dataset:{dataset_name}",
        "stage:tokenize",
        f"mode:{spec.mode}",
        f"format:{spec.audio_text_format}",
        f"task:{spec.audio_text_task}",
    ]
    tags: list[str] = []
    for tag in [*configured_tags, *automatic]:
        if tag not in tags:
            tags.append(tag)
    return tags


class _WorkUnitBatchSampler:
    """Yield dynamic batches independently per SHAR work unit.

    Lhotse SHAR is a streaming format: skipping globally planned batches would
    still read skipped audio bytes from tar streams. Keeping bucketing scoped to
    one work unit preserves sequential I/O locality and makes batch padding for
    each cut independent of the number of launched ranks.
    """

    def __init__(
        self,
        work_units: list[tuple[str, Any, int | None]],
        *,
        sampler_kwargs: dict[str, Any],
        rank: int,
    ) -> None:
        self.work_units = work_units
        self.sampler_kwargs = sampler_kwargs
        self.rank = rank
        self._completed_work_units = 0

    def __iter__(self):
        self._completed_work_units = 0
        for work_unit_id, cuts, cut_count in self.work_units:
            kwargs = _cap_sampler_buckets_to_cut_count(
                dict(self.sampler_kwargs),
                cut_count=cut_count,
                rank=self.rank,
            )
            logger.info(
                "[rank %s] Sampling work unit %s with %s estimated cuts",
                self.rank,
                work_unit_id,
                cut_count,
            )
            sampler = DynamicBucketingSampler(cuts, **kwargs)
            for batch in sampler:
                yield batch
            self._completed_work_units += 1


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
    from audio_tokenization.prepare.audio_ops import normalize_batch_peak

    if "audio" in batch:
        batch["audio"] = normalize_batch_peak(batch["audio"].to(device, non_blocking=True), target_db)
    elif "inputs" in batch:
        batch["inputs"] = normalize_batch_peak(batch["inputs"].to(device, non_blocking=True), target_db)
    return batch


# ---------------------------------------------------------------------------
# Main tokenization loop (per-rank)
# ---------------------------------------------------------------------------


def tokenize_loop(
    spec: TokenizeSpec,
    *,
    dataset_name: str,
    input_shar_dirs: Sequence[str],
    planned_shar_fields: dict[str, list[str]],
    planned_work_units: list[dict[str, Any]] | None = None,
    rank: int,
    world_size: int,
    local_rank: int,
    final_output_dir: Path,
    handler,
    assigned_cut_count: int | None = None,
) -> dict[str, Any]:
    """Main per-rank tokenization loop.

    Steps:
        1. Load prepared Shar CutSet -- see ``data.py``
        2. Create ``DynamicBucketingSampler`` with global bucketing
        3. Optionally resume from checkpoint
        4. Wrap in dataset + ``DataLoader`` for CPU/GPU overlap
        5. Loop over prefetched batches, tokenize on GPU, write output
        6. Periodically checkpoint (sampler state + chunk boundary)
    """
    output_dir = str(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale .tmp files from killed runs (e.g. OOM kill).
    cleanup_tmp_files(
        final_output_dir,
        f"rank_{rank:04d}_*.tmp",
        logger=logger,
        label=f"rank {rank} stale temp file",
    )

    cumulative_stats = WorkerStats()

    # ------------------------------------------------------------------
    # 1. Build CutSet (prepared Shar load + filters/resample safety-net)
    # ------------------------------------------------------------------
    if planned_work_units:
        work_unit_cutsets = []
        for work_unit in planned_work_units:
            cuts = build_cutset(
                spec,
                input_shar_dirs=input_shar_dirs,
                planned_shar_fields=work_unit["fields"],
                rank=rank,
                world_size=1,
                stats=cumulative_stats,
            )
            work_unit_cutsets.append(
                (
                    str(work_unit["work_unit_id"]),
                    cuts,
                    int(work_unit["cut_count"]),
                )
            )
        cut_count = (
            assigned_cut_count
            if assigned_cut_count is not None
            else sum(count for _wid, _cuts, count in work_unit_cutsets)
        )
    else:
        cuts = build_cutset(
            spec,
            input_shar_dirs=input_shar_dirs,
            planned_shar_fields=planned_shar_fields,
            rank=rank,
            world_size=world_size,
            stats=cumulative_stats,
        )
        work_unit_cutsets = None
        cut_count = assigned_cut_count if assigned_cut_count is not None else _cutset_len(cuts)
    if cut_count == 0:
        logger.warning(
            "[rank %s/%s] no cuts assigned after shard-level distribution; "
            "writing empty rank stats and exiting cleanly.",
            rank,
            world_size,
        )
        result = cumulative_stats.finalize()
        result["rank"] = rank
        result["chunks_written"] = 0
        result["output_dir"] = output_dir
        result["success"] = True
        from .stats_reducer import (
            maybe_publish_terminal_artifacts,
            write_rank_stats,
        )

        write_rank_stats(output_dir, result)
        # Terminal publication failure must surface in exit code; don't swallow.
        maybe_publish_terminal_artifacts(output_dir, expected_ranks=world_size)
        return result

    # ------------------------------------------------------------------
    # 2. Dynamic bucketing sampler -- each rank's CutSet is already split
    #    at the shard level (see data.py), so the sampler uses
    #    world_size=1 to avoid the O(world_size) strided distribution.
    # ------------------------------------------------------------------
    sampler_kwargs = _build_sampler_kwargs(spec)
    if work_unit_cutsets is None:
        sampler_kwargs = _cap_sampler_buckets_to_cut_count(
            sampler_kwargs,
            cut_count=cut_count,
            rank=rank,
        )
    if work_unit_cutsets is not None:
        sampler = _WorkUnitBatchSampler(
            work_unit_cutsets,
            sampler_kwargs=sampler_kwargs,
            rank=rank,
        )
    else:
        sampler = DynamicBucketingSampler(cuts, **sampler_kwargs)

    # ------------------------------------------------------------------
    # 3. DataLoader with prefetching -- worker subprocesses decode audio
    #    in parallel while the main thread runs GPU tokenization.
    # ------------------------------------------------------------------
    max_workers = os.cpu_count() // max(torch.cuda.device_count(), 1)
    num_workers = min(int(spec.dataloader.num_workers), max_workers)
    prefetch_factor = spec.dataloader.prefetch_factor
    effective_prefetch_factor = prefetch_factor if num_workers > 0 else None
    dataloader_timeout = 300  # 5 min default
    worker_init_fn = None
    if num_workers > 0:
        from lhotse.dataset.dataloading import make_worker_init_fn

        worker_init_fn = make_worker_init_fn(
            rank=rank,
            world_size=world_size,
            seed=spec.dataloader.sampler_seed,
        )

    dataset = handler.create_dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=effective_prefetch_factor,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        timeout=dataloader_timeout if num_workers > 0 else 0,
        worker_init_fn=worker_init_fn,
    )

    # ------------------------------------------------------------------
    # 4. Create tokenizer on GPU
    # ------------------------------------------------------------------
    from audio_tokenization.vokenizers import create_tokenizer

    device = f"cuda:{local_rank}"
    tokenizer_path = spec.tokenizer.path
    mode = spec.mode
    torch_compile = spec.tokenizer.torch_compile
    target_sr = int(spec.tokenizer.sampling_rate)
    trim_last_tokens = spec.tokenizer.trim_last_tokens

    tokenizer = create_tokenizer(
        omni_tokenizer_path=tokenizer_path,
        mode=mode,
        device=device,
        torch_compile=torch_compile,
        trim_last_tokens=trim_last_tokens,
    )

    # ------------------------------------------------------------------
    # 5. W&B logger (rank 0 only)
    # ------------------------------------------------------------------
    wandb_logger = None
    wandb_cfg = spec.wandb
    if wandb_cfg.get("enabled", False) and rank == 0:
        # Auto-generate wandb run name: {task}/{dataset}_dur{min}-{max}
        _wandb_name = wandb_cfg.get("name")
        _output_name = resolve_tokenize_output_name(spec, dataset_name=dataset_name)
        if not _wandb_name:
            _min_d = spec.filter.min_duration
            _max_d = spec.filter.max_duration
            _dur_tag = ""
            if _min_d is not None or _max_d is not None:
                _dur_tag = (
                    f"_dur{_format_duration_tag_value(_min_d) or 'min'}-"
                    f"{_format_duration_tag_value(_max_d) or 'max'}"
                )
            _wandb_name = f"{build_tokenize_output_subdir(spec, dataset_name=dataset_name)}{_dur_tag}"

        wandb_logger = SimpleWandbLogger(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            name=_wandb_name,
            tags=_build_wandb_tags(
                wandb_cfg["tags"],
                dataset_name=dataset_name,
                spec=spec,
            ),
            config=_build_wandb_config(
                spec,
                dataset_name=dataset_name,
                input_shar_dirs=input_shar_dirs,
                final_output_dir=final_output_dir,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                assigned_cut_count=assigned_cut_count,
                sampler_kwargs=sampler_kwargs,
                effective_num_workers=num_workers,
                effective_prefetch_factor=effective_prefetch_factor,
                max_workers_per_rank=max_workers,
                dataloader_timeout=dataloader_timeout if num_workers > 0 else 0,
                output_name=_output_name,
            ),
            log_interval_seconds=wandb_cfg["log_interval_seconds"],
        )

    # ------------------------------------------------------------------
    # 6. Main loop -- tokenize batches and rotate output chunks
    # ------------------------------------------------------------------
    checkpoint_interval = spec.dataloader.checkpoint_interval_batches
    writer_state: Any = 0
    batch_count = 0

    handler.setup_writer(output_dir, rank, writer_state, tokenizer)

    stats = cumulative_stats
    total_audio_seconds = 0.0

    logger.info(
        f"[rank {rank}] Starting tokenization loop "
        f"(writer_state={_format_writer_state(writer_state)}, checkpoint_interval={checkpoint_interval})"
    )

    consecutive_errors = 0
    max_consecutive_errors = 50
    _loop_error = None

    normalize_peak_db = spec.filter.normalize_peak_db
    if normalize_peak_db is not None:
        normalize_peak_db = float(normalize_peak_db)
        logger.info(f"[rank {rank}] Peak normalization enabled: target {normalize_peak_db} dBFS")

    _t_start = _t_encode_start = _t_encode_end = None
    if wandb_logger is not None:
        _t_start = torch.cuda.Event(enable_timing=True)
        _t_encode_start = torch.cuda.Event(enable_timing=True)
        _t_encode_end = torch.cuda.Event(enable_timing=True)

    _batch_ready_time = time.monotonic()

    _pbar = None
    if rank == 0:
        from tqdm import tqdm
        _pbar = tqdm(desc="tokenize", unit=" batches", dynamic_ncols=True)

    try:
        for batch in dataloader:
            _dataloader_wait_ms = (time.monotonic() - _batch_ready_time) * 1000

            # Decide whether to capture per-batch timing (only when W&B will flush).
            _time_this = (
                wandb_logger is not None and wandb_logger.should_log_now()
            )
            if _time_this:
                _t_start.record()

            try:
                _host_process_start = time.monotonic()
                # Normalize audio volume before tokenization (all modes).
                if normalize_peak_db is not None:
                    batch = _normalize_batch(batch, normalize_peak_db, device)

                if _time_this:
                    _t_encode_start.record()

                batch_audio_secs = handler.process_batch(
                    batch, tokenizer, stats, target_sr, device,
                )

                if _time_this:
                    _t_encode_end.record()

                total_audio_seconds += batch_audio_secs
                _process_batch_wall_ms = (time.monotonic() - _host_process_start) * 1000
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
                elapsed = time.time() - stats.start_time
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

            # Periodic chunk rotation: finalize current chunk and open next.
            if batch_count % checkpoint_interval == 0 and handler.chunk_samples > 0:
                writer_state = handler.checkpoint_writer()
                logger.info(
                    f"[rank {rank}] Rotated writer state {_format_writer_state(writer_state)} "
                    f"({stats.tokens_generated} total tokens)"
                )

            _batch_ready_time = time.monotonic()

    except Exception as e:
        logger.error(f"[rank {rank}] Fatal error in tokenization loop: {e}", exc_info=True)
        stats.errors += 1
        _loop_error = e

    if _pbar is not None:
        _pbar.close()

    # ------------------------------------------------------------------
    # 7. Finalize last chunk.
    # ------------------------------------------------------------------
    handler.finalize_writer()

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
    result["success"] = _loop_error is None

    from .stats_reducer import (
        maybe_publish_terminal_artifacts,
        write_rank_stats,
    )

    # Rank stats are the distributed success signal. If they cannot be written,
    # no rank can safely publish _SUCCESS, so fail the rank loudly.
    write_rank_stats(output_dir, result)
    summary = maybe_publish_terminal_artifacts(output_dir, expected_ranks=world_size)
    if summary is not None:
        logger.info(
            f"[rank {rank}] Stats summary: {summary['samples_processed']:,} samples, "
            f"{summary['audio_tokens']:,} audio tokens, "
            f"{summary['text_tokens']:,} text tokens across {summary['num_ranks']} ranks"
        )

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


def run_lhotse_pipeline(
    spec: TokenizeSpec,
    *,
    dataset_name: str,
    input_shar_dirs: Sequence[str],
    planned_shar_fields: dict[str, list[str]],
    planned_work_units: list[dict[str, Any]] | None = None,
    rank: int,
    world_size: int,
    local_rank: int,
    final_output_dir: Path,
    assigned_cut_count: int | None = None,
) -> dict[str, Any]:
    """Entry point for the Lhotse tokenization pipeline.

    Expects pre-built Shar data (via ``prepare_hf_to_shar`` or
    ``prepare_wds_to_shar``).  Loads the Shar CutSet, tokenizes on GPU,
    and writes micro-shards with DDP checkpointing.
    """
    if planned_shar_fields is None:
        raise RuntimeError(
            "Tokenization requires a stage-created SHAR assignment. Use "
            "`python -m audio_tokenization run ... stage=tokenize`; direct "
            "`run_lhotse_pipeline` calls must pass planned_shar_fields."
        )

    # Only rank 0 logs at INFO; others at WARNING to avoid 160x noise.
    if rank != 0:
        logging.getLogger("audio_tokenization").setLevel(logging.WARNING)
        logging.getLogger("lhotse").setLevel(logging.WARNING)

    torch.cuda.set_device(local_rank)

    logger.info(
        f"[rank {rank}/{world_size}] starting (local_rank={local_rank}, "
        f"no NCCL — each rank is independent)"
    )

    if spec.mode == "audio_only":
        from .audio_only import AudioOnlyHandler
        handler = AudioOnlyHandler(spec)
    elif spec.mode == "audio_text":
        from .audio_text import AudioTextHandler
        handler = AudioTextHandler(spec, dataset_name=dataset_name)
    else:
        assert_never(spec.mode)

    return tokenize_loop(
        spec,
        dataset_name=dataset_name,
        input_shar_dirs=input_shar_dirs,
        planned_shar_fields=planned_shar_fields,
        planned_work_units=planned_work_units,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        final_output_dir=final_output_dir,
        assigned_cut_count=assigned_cut_count,
        handler=handler,
    )
