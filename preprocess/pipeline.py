import logging
from typing import Dict, Any
import os
from pathlib import Path
import json

import torch

from .outputs.base import BaseOutputWriter
# from .outputs.shar import SharOutputWriter
from .outputs.cuts import CutsOutputWriter

from .checkpoint import (
    PreprocessStats,
    load_checkpoint,
    SimpleWandbLogger,
    is_cuda_oom,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def _load_shard_index(cfg: Dict[str, Any]):
    """Load the shar_index.json and return the fields dict."""
    from audio_tokenization.pipelines.lhotse.data import SHAR_INDEX_FILENAME

    shar_dir = cfg.get("shar_dir")
    if not shar_dir:
        raise ValueError("shar_dir is required")
    
    shar_dirs = shar_dir if isinstance(shar_dir, (list, tuple)) else [shar_dir]
    index_name = cfg.get("shar_index_filename", SHAR_INDEX_FILENAME)

    merged_fields: dict[str, list[str]] = {}
    for sd in shar_dirs:
        shar_path = Path(sd)
        index_path = shar_path / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing shar index: {index_path}")
        
        with open(index_path) as f:
            fields = json.load(f).get("fields", {})
        
        for field, paths in fields.items():
            resolved = []
            for p in paths:
                pp = Path(p)
                if not pp.is_absolute():
                    pp = shar_path / pp
                resolved.append(str(pp))
            merged_fields.setdefault(field, []).extend(resolved)
        
        merged_fields = {k: sorted(v) for k, v in merged_fields.items()}
    return merged_fields

def _build_cutset_for_shard(shard: dict[str, str], cfg: Dict[str, Any]):
    """Load a CutSet from a single shard's cuts + recording paths."""
    from lhotse import CutSet
    from audio_tokenization.pipelines.lhotse.data import _set_resampling_backend

    _set_resampling_backend(cfg.get("rank", 0))

    fields = {k: [v] for k, v in shard.items()}
    cuts = CutSet.from_shar(fields=fields, split_for_dataloading=False, shuffle_shards=False)

    # Drop low sample-rate audio before resampling (e.g., 8kHz -> 24kHz = garbage).
    min_sr = cfg.get("min_sample_rate")
    if min_sr is not None:
        min_sr = int(min_sr)
        cuts = cuts.filter(
            lambda cut: getattr(cut, "sampling_rate", None) is not None
            and cut.sampling_rate >= min_sr
        )
    
    # Lazy safety-net resample (no-op when SR already matches).
    target_sr = cfg.get("target_sample_rate")
    if target_sr:
        cuts = cuts.resample(int(target_sr))

    min_dur = cfg.get("min_duration")
    max_dur = cfg.get("max_duration")
    if min_dur is not None or max_dur is not None:
        def _dur_filter(cut) -> bool:
            d = cut.duration
            if min_dur is not None and d < min_dur:
                return False
            if max_dur is not None and d > max_dur:
                return False
            return True

        cuts = cuts.filter(_dur_filter)

    return cuts

def _assign_shards_to_rank(fields: dict[str, list[str]], 
                           rank: int,
                           world_size: int
                           ) -> list[dict[str, str]]:
    """Round-robin assign shards to ranks.
 
    Returns a list of dicts, one per shard assigned to this rank:
        [{"cuts": "/path/cuts_000000.jsonl.gz",
          "recording": "/path/recording_000000.tar"}, ...]
    """
    cuts_paths = fields.get("cuts", [])
    # Build per-shard field mapping.
    # Fields are aligned by index: cuts[i] <-> recording[i] <-> ...      
    all_shards = []
    for i in range(len(cuts_paths)):
        shard = {}
        for field, paths in fields.items():
            if i < len(paths):
                shard[field] = paths[i]
        all_shards.append(shard)    
    
    # Round robin assignment
    return [all_shards[i] for i in range(rank, len(all_shards), world_size)]

def _detect_shard_language(batch, cfg, target_sr):
    """Detect shard language via majority vote over the first batch."""
    from collections import Counter

    whisper = cfg.get("_whisperASR")
    if whisper is None:
        return

    audios = batch["inputs"]                          # (B, T)
    cuts_list = batch["supervisions"]["cut"]
    audio_lens = torch.tensor(
        [c.num_samples for c in cuts_list], dtype=torch.int64
    )

    n = min(audios.shape[0], cfg.get("language_detection_samples", audios.shape[0]))

    results = whisper.detect_language(
        audios[:n], audio_lens[:n], sr=target_sr
    )

    lang_list = [r["language"] for r in results]
    vote_counts = Counter(lang_list)
    best_lang, _ = vote_counts.most_common(1)[0]
    cfg["_shard_language"] = best_lang
    logger.info(
        f"Detected shard language: {best_lang}"
    )

# ------------------------------------------------------------------
# Preprocess Loop per shard
# ------------------------------------------------------------------

def preprocess_loop(rank: int, 
                    world_size: int, 
                    cfg: Dict[str, Any], 
                    writer: BaseOutputWriter,
                    device: str
                    ) -> Dict[str, Any]:
    """Per-rank preprocess loop.

    Steps:
        1. Load SHAR CutSet
        2. DynamicBucketingSampler for batching
        3. DataLoader with CPU prefetch workers for audio preprocessing
        4. Batch transcription
        5. Per-sample metric computation on CPU
        6. Write results via output writer
        7. Periodically checkpoint
    """

    from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler

    output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Assign shards to current rank.
    # ------------------------------------------------------------------
    fields = _load_shard_index(cfg)
    my_shards = _assign_shards_to_rank(fields, rank, world_size)
    total_shards = len(fields.get("cuts", []))
    logger.info(
        f"[rank {rank}] Assigned {len(my_shards)}/{total_shards}"
    )

    if not my_shards:
        logger.warning(f"[rank {rank}] No shards assigned, exiting")
        return PreprocessStats().finalize()

    # ------------------------------------------------------------------
    # 2. Resume from checkpoint
    # ------------------------------------------------------------------
    start_shard_idx = 0
    stats = PreprocessStats()
    resume_cuts_written = 0
    
    if cfg.get("resume", False):
        ckpt = load_checkpoint(output_dir, rank, world_size)
        if ckpt is not None:
            start_shard_idx = ckpt["shard_idx"]
            resume_cuts_written = ckpt["cuts_written"]
            prev = ckpt.get("stats", {})
            stats.samples_processed = prev.get("samples_processed", 0)
            stats.samples_skipped = prev.get("samples_skipped", 0)
            stats.errors = prev.get("errors", 0)
            stats.total_audio_seconds = prev.get("total_audio_seconds", 0.0)
            stats.text_tokens_generated = prev.get("text_tokens_generated", 0.0)
            logger.info(
                f"[rank {rank}] Resuming from shard {start_shard_idx},"
                f"cuts_written={resume_cuts_written}"
            )

    # ------------------------------------------------------------------
    # 3. W&B logger (rank 0 only)
    # ------------------------------------------------------------------
    wandb_logger = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False) and rank == 0:
        wandb_logger = SimpleWandbLogger(
                    project=wandb_cfg.get("project", "audio-preprocessing"),
                    entity=wandb_cfg.get("entity"),
                    name=wandb_cfg.get("name"),
                    tags=wandb_cfg.get("tags", []),
                    config={
                        "rank": rank,
                        "world_size": world_size,
                        "max_batch_duration": cfg.get("max_batch_duration", 1500.0),
                        "output_format": cfg.get("output_format", "cuts_only"),
                        "target_sample_rate": cfg.get("target_sample_rate"),
                        "include_text_tokens": cfg.get("include_text_tokens", False),
                        **{k:v for k, v in cfg.items() if isinstance(v, (int, float, str, bool)) and not k.startswith("_")},    
                    },
                    log_interval_seconds=wandb_cfg.get("log_interval_seconds", 10.0),
                )
    
    # ------------------------------------------------------------------
    # 4. DataLoader with prefetch workers + Main Loop
    # ------------------------------------------------------------------

    # Prefetch Worker
    max_workers = os.cpu_count() // max(torch.cuda.device_count(), 1)
    num_workers = min(cfg.get("num_workers", 4), max_workers)
    prefetch_factor = cfg.get("prefetch_factor", 4)

    target_sr = int(cfg.get("target_sample_rate", 16000))

    # Dataset configuration
    from lhotse.dataset import K2SpeechRecognitionDataset
    from lhotse.dataset.input_strategies import AudioSamples
    dataset = K2SpeechRecognitionDataset(
        return_cuts=True,
        input_strategy=AudioSamples()
    )
    
    # Checkpoint configuration
    checkpoint_interval = cfg.get("checkpoint_interval_batches", 500)
    max_consecutive_errors = cfg.get("max_consecutive_errors", 50)
    _loop_error = None

    # Sampler Arguments for dynamic batching
    sampler_kwargs = dict(
            max_duration=cfg.get("max_batch_duration", 1500.0),
            max_cuts=cfg.get("max_batch_cuts"),
            num_buckets=cfg.get("num_buckets", 20),
            buffer_size=cfg.get("bucket_buffer_size", 20000),
            shuffle=cfg.get("sampler_shuffle", True),
            seed=cfg.get("sampler_seed", 42),
            world_size=1,
            rank=0,
            drop_last=False,
        )
    if cfg.get("quadratic_duration") is not None:
        sampler_kwargs["quadratic_duration"] = cfg["quadratic_duration"]

    # Main Loop
    try:
        for shard_local_idx in range(start_shard_idx, len(my_shards)):
            shard = my_shards[shard_local_idx]
            cuts_path = shard["cuts"]

            logger.info(
                f"[rank {rank}] Processing shard {shard_local_idx + 1}/{len(my_shards)}: {cuts_path}"
            )

            # Build CutSet for this shard
            cuts = _build_cutset_for_shard(shard, cfg)

            # Build Sampler for this shard
            sampler = DynamicBucketingSampler(cuts, **sampler_kwargs)

            # Resume sampler within this shard
            if shard_local_idx == start_shard_idx and resume_cuts_written > 0:
                ckpt = load_checkpoint(output_dir, rank, world_size)
                if ckpt is not None and ckpt.get("sampler_state"):
                    sampler.load_state_dict(ckpt["sampler_state"])
            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=None,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=False,
                pin_memory=True,
                timeout=cfg.get("dataloader_timeout", 300) if num_workers > 0 else 0,
            )


            # Open writer for this shard
            rc = resume_cuts_written if shard_local_idx == start_shard_idx else 0
            
            shard_source_dir = Path(cuts_path).parent.name
            shard_output_dir = Path(cfg.get("output_dir_data", cfg["output_dir"])) / shard_source_dir
            shard_output_dir.mkdir(parents=True, exist_ok=True)
            writer.output_dir = str(shard_output_dir)
            writer.open(cuts_path, resume_count=rc)

            batch_count = 0
            consecutive_errors = 0

            cfg.pop("_shard_language", None)
            _shard_lang_detected = False
            for batch in dataloader:
                try:
                    if not _shard_lang_detected:
                        _detect_shard_language(batch, cfg, target_sr)
                        _shard_lang_detected = True

                        if cfg.get("_shard_language") in cfg.get("NVIDIA_LANGUAGES", []):
                            from .models.parakeetwrapper import ParakeetWrapper

                            cfg["model"] = ParakeetWrapper(
                                rank=rank,
                                device=device,
                                model_id=cfg.get("nvidia_model", "nvidia/parakeet-tdt-0.6b-v3"),
                                cache_dir=cfg.get("cache_dir")
                            )

                    _process_batch(
                        batch, writer, stats, target_sr, device, cfg
                    )
                    consecutive_errors = 0
                except Exception as batch_err:
                    stats.errors += 1
                    consecutive_errors += 1
                    if is_cuda_oom(batch_err):
                        torch.cuda.empty_cache()
                        logger.warning(
                            f"[rank {rank}] CUDA OOM on batch {batch_count} ({consecutive_errors}/{max_consecutive_errors})"
                        )
                    else:
                        logger.warning(
                            f"[rank {rank}] batch error ({consecutive_errors}/{max_consecutive_errors}): {batch_err}",
                            exc_info=True,
                        )
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"[rank {rank}] {max_consecutive_errors} consecutive errors"
                        ) from batch_err
                    continue
                batch_count += 1

                if wandb_logger is not None:
                    wandb_logger.log(stats)
                
                if batch_count % checkpoint_interval == 0:
                    save_checkpoint(
                        output_dir,
                        rank,
                        shard_idx=shard_local_idx,
                        sampler_state=sampler.state_dict(),
                        cuts_written=writer.cuts_written,
                        stats=stats.to_dict(),
                        world_size=world_size
                    )
            
            # Shard complete - rename .tmp to to .jsonl.gz  format
            writer.finalize(rank)

            # Mark this shard one by updating shard_idx to shard_idx + 1
            save_checkpoint(
                output_dir,
                rank,
                shard_idx=shard_local_idx + 1,
                sampler_state={},
                cuts_written=0,
                stats=stats.to_dict(),
                world_size=world_size
            )

            # Reset parameters
            resume_cuts_written = 0

    except Exception as e:
        logger.error("[rank %d] fatal: %s", rank, e, exc_info=True)
        stats.errors += 1
        writer.close()
        _loop_error = e
    
    result = stats.finalize()
    result["rank"] = rank

    if wandb_logger is not None:
        wandb_logger.log(stats, force=True)
        wandb_logger.log_final(result)
        wandb_logger.finish()

    logger.info(
        f"[rank {rank}] Done: {result['samples_processed']} samples, "
        f"{result['errors']} errors, {result['elapsed_time']:.1f}s"
    )

    if _loop_error is not None:
        raise RuntimeError(
            f"[rank {rank}] Pipeline failed after {result['samples_processed']} samples"
        ) from _loop_error

    return result


# ------------------------------------------------------------------
# Preprocess per batch
# ------------------------------------------------------------------

def _process_batch(batch: Dict[str, Any], 
                   writer: BaseOutputWriter, 
                   stats: PreprocessStats, 
                   target_sr: int, 
                   device: str,
                   cfg: Dict[str, Any]):
    
    audios = batch["inputs"]
    cuts_list = batch["supervisions"]["cut"]
    audio_lens = torch.tensor(
        [c.num_samples for c in cuts_list], dtype=torch.int64
    ).to(device)

    B = audios.shape[0]

    whisper = cfg.get("_whisperASR")
    model = cfg.get("model", whisper)
    shard_language = cfg.get("_shard_language")
    transcription = None
    cer_metrics = None
    if model is not None and cfg.get("transcribe", True):
        transcription = model.transcribe_batch(
            audios, audio_lens, sr=target_sr, language=shard_language
        )  

        if cfg.get("calculate_cer", True):
            refs = [cut.supervisions[0].text for cut in cuts_list]
            cer_metrics = whisper.calculate_cer(transcriptions=transcription, 
                                                references=refs)
    
    for i in range(B):
        cut = cuts_list[i]

        new_text = transcription[i] if transcription is not None else None
        text = cut.supervisions[0].text if len(cut.supervisions) > 0 else None
        
        text_tokens = None
        if cfg.get("include_text_tokens", False):
            tokenize_text = new_text if transcription is not None and cfg.get("use_transcription", False) else text
            
            text_tokens = None
            tokenize_fn = cfg.get("_text_tokenizer_fn")
            if tokenize_fn is not None:
                text_tokens = tokenize_fn.encode(
                    tokenize_text, add_special_tokens=False
                ).ids
                stats.text_tokens_generated += len(text_tokens)
        
        duration = int(audio_lens[i]) / target_sr
        extra = {"duration": duration}
        if shard_language is not None:
            extra["language"] = shard_language
        if cer_metrics is not None:
            extra["CER"] = cer_metrics[i]

        cut_dict = cut.to_dict()
        cut_dict.setdefault("custom", {})

        writer.write_cut(
            cut_dict=cut_dict,
            text=new_text,
            extra=extra,
            text_tokens=text_tokens
        )

        stats.samples_processed += 1
        stats.total_audio_seconds += duration

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for preprocessing pipeline.

    Resolves rank/world_size from environment (SLURM or torchrun),
    instantiates the tokenizer and output writer, then runs the loop.
    """
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCAL_ID", 0)))

    if "LOCAL_RANK" not in os.environ and "SLURM_LOCAL_ID" not in os.environ:
        gpus_per_node = torch.cuda.device_count()
        if gpus_per_node > 0:
            local_rank = rank % gpus_per_node
    
    # Only rank 0 logs at INFO
    if rank != 0:
        # logging.getLogger("preprocess").setLevel(logging.WARNING)
        logging.getLogger("audio_tokenization").setLevel(logging.WARNING)
        logging.getLogger("lhotse").setLevel(logging.WARNING)

    cfg["rank"] = rank
    cfg["world_size"] = world_size
    cfg["local_rank"] = local_rank

    torch.cuda.set_device(local_rank)
    device = f"cuda:{cfg.get('local_rank', 0)}"

    logger.info(
        f"[rank {rank}/{world_size}] starting (local_rank={local_rank})"
    )

    # Loading text tokenizer for including text tokens
    if cfg.get("include_text_tokens", False):
        from audio_tokenization.utils.prepare_data.common import (
            load_text_tokenizer
        )

        text_tok_path = cfg.get("text_tokenizer_path")
        if text_tok_path is None:
            raise ValueError(
                "text_tokenizer_path is required when include_text_tokens=true"
            )
        cfg["_text_tokenizer_fn"] = load_text_tokenizer(text_tok_path)
        logger.info(f"[rank {rank}] text tokenization enabled: {text_tok_path}")

    # Load ASR models
    from .models.whisperwrapper import WhisperWrapper

    whisper_model_id = cfg.get("whisper_model_id", "openai/whisper-large-v3")
    cfg["_whisperASR"] = WhisperWrapper(
                                    rank,
                                    device=device,
                                    model_id=whisper_model_id,
                                    cache_dir=cfg.get("cache_dir"),
                                    )

    # Output
    writer_type = cfg.get("output_format", "cuts_only")
    output_dir_data = cfg.get("output_dir_data", None)
    shard_size = cfg.get("output_shard_size", 1000)
    if writer_type == "shar":
        # writer = SharOutputWriter(
        #     shard_size=shard_size,
        #     output_dir=output_dir_data
        # )
        raise ValueError(f"SharOutputWriter needs to be implemented")
    elif writer_type == "cuts_only":
        writer = CutsOutputWriter(
            shard_size=shard_size,
            output_dir=output_dir_data
        )
    else:
        raise ValueError(f"Unsupported output format: {writer_type}")
    
    return preprocess_loop(rank, world_size, cfg, writer, device)
