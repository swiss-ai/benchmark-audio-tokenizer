"""Checkpointing, micro-shard I/O, stats tracking, and W&B logging.

Design decisions:
- **Micro-shard chunking**: Each rank writes independent chunks named
  ``rank_XXXX_chunk_YYYY.{bin,idx}``.  Chunks are written to ``.tmp``
  files first and atomically renamed on finalize — no partial files on crash.
- **Sampler-state checkpointing**: Lhotse's ``DynamicBucketingSampler``
  supports ``state_dict()`` / ``load_state_dict()``. On resume the sampler
  restores progress through metadata bookkeeping (no audio decoding), so
  recovery is typically fast.
- **WorkerStats** is an inline dataclass (no Ray dependency from base.py).
- **SimpleWandbLogger**: Plain Python class (rank 0 only), rate-limited
  by a configurable interval.  No Ray actor overhead.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from audio_tokenization.pipelines.shard_io import finalize_shard_writer
from audio_tokenization.utils.indexed_dataset import DType, IndexedDatasetBuilder

logger = logging.getLogger(__name__)

# Re-export finalize_shard_writer so pipeline.py only needs to import from here.
__all__ = [
    "WorkerStats",
    "open_chunk_writer",
    "save_checkpoint",
    "load_checkpoint",
    "SimpleWandbLogger",
    "finalize_shard_writer",
    "is_cuda_oom",
    "_get_rss_gb",
]


def _get_rss_gb() -> float:
    """Return current process RSS in GiB by reading /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)  # kB -> GiB
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# CUDA OOM detection
# ---------------------------------------------------------------------------


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True if *exc* indicates a CUDA out-of-memory error.

    Checks both the dedicated ``torch.cuda.OutOfMemoryError`` (PyTorch ≥ 2.0)
    and the older ``RuntimeError("CUDA out of memory")`` pattern.
    """
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(exc, cuda_oom_type):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "cuda out of memory" in msg or "out of memory" in msg
    return False


# ---------------------------------------------------------------------------
# Per-rank statistics
# ---------------------------------------------------------------------------


@dataclass
class WorkerStats:
    """Cumulative statistics tracked per rank (no Ray dependency)."""

    samples_processed: int = 0
    tokens_generated: int = 0
    text_tokens_generated: int = 0
    errors: int = 0
    samples_skipped: int = 0
    rms_skipped: int = 0
    no_text_skipped: int = 0
    duration_skipped: int = 0
    frequency_skipped: int = 0
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "samples_processed": self.samples_processed,
            "tokens_generated": self.tokens_generated,
            "errors": self.errors,
            "samples_skipped": self.samples_skipped,
            "rms_skipped": self.rms_skipped,
            "no_text_skipped": self.no_text_skipped,
            "duration_skipped": self.duration_skipped,
            "frequency_skipped": self.frequency_skipped,
            "elapsed_time": self.elapsed_time,
            "throughput": self.throughput,
        }
        if self.text_tokens_generated > 0:
            d["text_tokens_generated"] = self.text_tokens_generated
        return d

    def finalize(self) -> Dict[str, Any]:
        """Compute elapsed time and throughput, return final stats dict."""
        self.elapsed_time = time.time() - self.start_time
        self.throughput = (
            self.tokens_generated / self.elapsed_time if self.elapsed_time > 0 else 0
        )
        return self.to_dict()


# ---------------------------------------------------------------------------
# Micro-shard chunk writer
# ---------------------------------------------------------------------------


def open_chunk_writer(
    output_dir: str,
    rank: int,
    chunk_id: int,
    vocab_size: int,
) -> Tuple[IndexedDatasetBuilder, str, str, str, str]:
    """Open a Megatron IndexedDatasetBuilder for a micro-shard chunk.

    Naming: ``rank_XXXX_chunk_YYYY.{bin,idx}``
    Writes to ``.tmp`` suffix; call ``finalize_shard_writer()`` to atomically
    rename to the final paths.

    Returns:
        (builder, tmp_bin_path, tmp_idx_path, final_bin_path, final_idx_path)
    """
    output_prefix = Path(output_dir) / f"rank_{rank:04d}_chunk_{chunk_id:04d}"
    bin_path = str(output_prefix) + ".bin"
    idx_path = str(output_prefix) + ".idx"
    tmp_bin_path = bin_path + ".tmp"
    tmp_idx_path = idx_path + ".tmp"
    dtype = DType.optimal_dtype(vocab_size)
    builder = IndexedDatasetBuilder(tmp_bin_path, dtype=dtype)
    return builder, tmp_bin_path, tmp_idx_path, bin_path, idx_path


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def _checkpoint_path(output_dir: str, rank: int) -> Path:
    return Path(output_dir) / f"rank_{rank:04d}_checkpoint.pt"


def save_checkpoint(
    output_dir: str,
    rank: int,
    sampler_state: Dict[str, Any],
    writer_state: Any,
    stats: Dict[str, Any],
    world_size: int = 1,
) -> None:
    """Atomically save checkpoint via ``.tmp`` + ``os.replace()``."""
    ckpt_path = _checkpoint_path(output_dir, rank)
    tmp_path = str(ckpt_path) + ".tmp"
    payload = {
        "sampler_state": sampler_state,
        "writer_state": writer_state,
        "stats": stats,
        "world_size": world_size,
    }
    if isinstance(writer_state, int):
        payload["chunk_id"] = writer_state
    torch.save(payload, tmp_path)
    os.replace(tmp_path, str(ckpt_path))
    logger.debug(f"[rank {rank}] Saved checkpoint writer_state={writer_state!r}")


def load_checkpoint(output_dir: str, rank: int) -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists, else return None."""
    ckpt_path = _checkpoint_path(output_dir, rank)
    if not ckpt_path.exists():
        return None
    logger.info(f"[rank {rank}] Loading checkpoint from {ckpt_path}")
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# W&B logger (rank 0 only)
# ---------------------------------------------------------------------------


class SimpleWandbLogger:
    """Lightweight W&B logger for rank 0.

    Logs running totals + throughput at a configurable interval.
    Calls are rate-limited: ``log()`` is a no-op unless ``log_interval_seconds``
    has elapsed since the last flush (or ``force=True``).
    """

    def __init__(
        self,
        project: str = "audio-tokenization",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
        log_interval_seconds: float = 10.0,
    ):
        import wandb

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags or [],
            config=config or {},
            resume="allow",
        )
        self._interval = max(1.0, log_interval_seconds)
        self._last_flush = time.time()
        self._start_time = time.time()
        self._step = 0

    def should_log_now(self) -> bool:
        """Return True if the flush interval has elapsed since the last log."""
        return (time.time() - self._last_flush) >= self._interval

    def log(
        self,
        samples: int,
        tokens: int,
        errors: int,
        skipped: int,
        batch_audio_seconds: float = 0.0,
        text_tokens: int = 0,
        force: bool = False,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log absolute totals if the flush interval has elapsed."""
        now = time.time()
        if not force and now - self._last_flush < self._interval:
            return
        import wandb

        elapsed = now - self._start_time
        payload = {
            "samples_processed": samples,
            "audio_tokens_generated": tokens,
            "tokens_per_second": (tokens + text_tokens) / elapsed if elapsed > 0 else 0,
            "errors": errors,
            "samples_skipped": skipped,
            "samples_per_second": samples / elapsed if elapsed > 0 else 0,
            "audio_tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            "elapsed_seconds": elapsed,
            "batch_audio_seconds": batch_audio_seconds,
        }
        if text_tokens > 0:
            payload["text_tokens_generated"] = text_tokens
            payload["text_tokens_per_second"] = text_tokens / elapsed if elapsed > 0 else 0
        if metrics:
            payload.update(metrics)
        wandb.log(payload, step=self._step)
        self._step += 1
        self._last_flush = now

    def log_final(self, metrics: Dict[str, Any]) -> None:
        import wandb

        wandb.log({f"final/{k}": v for k, v in metrics.items()})

    def finish(self) -> None:
        import wandb

        wandb.finish()

