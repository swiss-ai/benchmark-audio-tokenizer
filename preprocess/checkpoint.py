import logging
import time
from pathlib import Path
import os

import torch
import wandb

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA OOM Helper
# ---------------------------------------------------------------------------

def is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutofMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "cuda out of memory" in msg or "out of memory" in msg
    return False

# ---------------------------------------------------------------------------
# Per-rank statistics
# ---------------------------------------------------------------------------

@dataclass
class PreprocessStats:
    """Cumulative statistics tracked per rank."""

    samples_processed: int = 0
    samples_skipped: int = 0
    errors: int = 0
    total_audio_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    throughput: float = 0.0
    text_tokens_generated = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples_processed": self.samples_processed,
            "samples_skipped": self.samples_skipped,
            "errors": self.errors,
            "total_audio_seconds": self.total_audio_seconds,
            "elapsed_time": self.elapsed_time,
            "throughput": self.throughput,
            "text_tokens_generated": self.text_tokens_generated
        }
    
    def finalize(self) -> Dict[str, Any]:
        self.elapsed_time = time.time() - self.start_time
        self.throughput = self.total_audio_seconds/self.elapsed_time if self.elapsed_time > 0 else 0
        return self.to_dict()

# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _ckpt_path(output_dir: str, rank: int) -> Path:
    return Path(output_dir) / f"rank_{rank:04d}_checkpoint.pt"

def save_checkpoint(
        output_dir: str,
        rank: int,
        *,
        shard_idx: int,
        sampler_state: Dict[str, Any],
        cuts_written: int,
        stats: Dict[str, Any],
        world_size: int = 1
    ) -> None:
    """Atomically save checkpoints"""

    path = _ckpt_path(output_dir, rank)
    tmp = str(path) + ".tmp"
    torch.save(
        {   
            "shard_idx": shard_idx,
            "sampler_state": sampler_state,
            "cuts_written": cuts_written,
            "stats": stats,
            "world_size": world_size
        },
        tmp
    )
    os.replace(tmp, str(path))
    logger.debug(f"[rank {rank}] Checkpoint saved (shard={shard_idx}, cuts={cuts_written})")

def load_checkpoint(
        output_dir: str,
        rank: int,
        world_size: int
    ) -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists and world_size matches."""

    path = _ckpt_path(output_dir, rank)
    if not path.exists():
        return None
    
    ckpt = torch.load(str(path), map_location="cpu",weights_only=False)
    ckpt_ws = ckpt.get("world_size")
    if ckpt_ws is not None and ckpt_ws != world_size:
        logger.warning(
            f"[rank {rank}] Checkpoint world size {ckpt_ws} != {world_size}"
        )
        return None
    logger.info(f"[rank {rank}] Loaded checkpoint (chunk_id={ckpt['shard_idx']})")
    return ckpt

# ---------------------------------------------------------------------------
# W&B logger (rank 0 only)
# ---------------------------------------------------------------------------

class SimpleWandbLogger:
    """Lightweight W&B logger for rank 0.

    Logs running totals, throughput and GPU stats at a configurable interval.
    """ 

    def __init__(
            self,
            project: str = "audio-preprocessing",
            entity: Optional[str] = None,
            name: Optional[str] = None,
            tags: Optional[list] = None,
            config: Optional[dict] = None,
            log_interval_seconds: float = 10.0,
            ):
        

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
    
    def log(self, stats: "PreprocessStats", force: bool = False) -> None:
        """Log benchmark stats + GPU metrics if the flush interval has elapsed."""
        now = time.time()
        if not force and now - self._last_flush < self._interval:
            return

        elapsed = now - self._start_time

        data = {
            "samples_processed": stats.samples_processed,
            "errors": stats.errors,
            "total_audio_seconds": stats.total_audio_seconds,
            "samples_per_second": stats.samples_processed / elapsed if elapsed > 0 else 0,
            "throughput": stats.total_audio_seconds / elapsed if elapsed > 0 else 0,
            "elapsed_seconds": elapsed,
            "text_tokens_generated": stats.text_tokens_generated
        }

        wandb.log(data, step=self._step)
        self._step += 1
        self._last_flush = now
    
    def log_final(self, metrics: Dict[str, Any]) -> None:
        wandb.log({f"final/{k}": v for k,v in metrics.items()})
    
    def finish(self) -> None:
        wandb.finish()