#!/usr/bin/env python3
"""Convert any Lhotse-supported dataset to Lhotse Shar format.

Uses Lhotse's built-in recipes (100+ datasets) to create manifests, then
spawns N parallel workers to convert to Shar. Each worker:
  1. Takes its interleaved partition of the CutSet
  2. Writes to ``part-{rank:05d}/`` via ``CutSet.to_shar()``

After all workers finish, builds a merged ``shar_index.json``.

Lhotse recipes return manifests in two shapes:
  - Flat:   {split: {"recordings": ..., "supervisions": ...}}
  - Nested: {language: {split: {"recordings": ..., "supervisions": ...}}}
Use ``--language`` to navigate the nested case.

Usage:
    # Common Voice zh-CN unverified (nested by language)
    python -m audio_tokenization.prepare.prepare_lhotse_recipe_to_shar \
        --recipe commonvoice \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24 \
        --split other \
        --language zh-CN \
        --target_sample_rate 24000 \
        --num_workers 64

    # Common Voice es train with explicit output dir name
    python -m audio_tokenization.prepare.prepare_lhotse_recipe_to_shar \
        --recipe commonvoice \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24 \
        --split train \
        --language es \
        --shar_base_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/commonvoice \
        --shar_output_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/commonvoice/es_train

    # LibriSpeech (flat)
    python -m audio_tokenization.prepare.prepare_lhotse_recipe_to_shar \
        --recipe librispeech \
        --corpus_dir /path/to/LibriSpeech \
        --split train-clean-360 \
        --num_workers 32

    # VoxPopuli (recipe-specific kwargs still needed for non-standard params)
    python -m audio_tokenization.prepare.prepare_lhotse_recipe_to_shar \
        --recipe voxpopuli \
        --corpus_dir /path/to/voxpopuli \
        --split train \
        --language en \
        --recipe_kwargs '{"task": "asr"}' \
        --num_workers 32

    # Thorsten-DE (single-split dataset, use --split all)
    PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:$PYTHONPATH \
    python -m audio_tokenization.prepare.prepare_lhotse_recipe_to_shar \
        --recipe thorsten_de \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/thorsten-de \
        --split all \
        --shar_base_dir /iopsstor/scratch/cscs/xyixuan/audio-datasets \
        --text_tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --num_workers 64

    # AISHELL-1 (run once per split: train, dev, test)
    PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:$PYTHONPATH \
    python -m audio_tokenization.prepare.prepare_lhotse_recipe_to_shar \
        --recipe aishell \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/aishell/aishell1 \
        --split train \
        --shar_base_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2 \
        --target_sample_rate 24000 \
        --text_tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --shar_shard_size 5000 \
        --num_workers 64

    # AISHELL-3 (run once per split: train, test)
    PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:$PYTHONPATH \
    python -m audio_tokenization.prepare.prepare_lhotse_recipe_to_shar \
        --recipe aishell3 \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/aishell/aishell3 \
        --split train \
        --shar_base_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2 \
        --target_sample_rate 24000 \
        --text_tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --shar_shard_size 5000 \
        --num_workers 64

    # AISHELL-4 (run once per split: train_L, train_M, train_S, test; requires: pip install textgrid)
    PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:$PYTHONPATH \
    python -m audio_tokenization.prepare.prepare_lhotse_recipe_to_shar \
        --recipe aishell4 \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/aishell/aishell4 \
        --split train_L \
        --shar_base_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2 \
        --target_sample_rate 24000 \
        --text_tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --shar_shard_size 5000 \
        --num_workers 64

    # HUI-Audio-Corpus-German (clean subset)
    PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:$PYTHONPATH \
    python -m audio_tokenization.prepare.prepare_lhotse_recipe_to_shar \
        --recipe hui_audio_corpus_german \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/hui-audio-corpus-german \
        --split clean \
        --shar_base_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2 \
        --target_sample_rate 24000 \
        --text_tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --shar_shard_size 5000 \
        --num_workers 64
"""

import argparse
import importlib
import json
import logging
import tempfile
from multiprocessing import Process
from pathlib import Path
from typing import Optional

from audio_tokenization.prepare.audio_ops import make_rms_filter_fn, to_mono
from audio_tokenization.prepare.constants import SUCCESS_MARKER_FILE
from audio_tokenization.prepare.identity import assign_universal_ids
from audio_tokenization.prepare.runtime import (
    build_shar_index_from_parts,
    mark_partition_success,
    setup_partition_dir,
    validate_prepare_runtime,
    write_prepare_state_for_spec,
)
from audio_tokenization.prepare.text_ops import (
    load_text_tokenizer,
    make_text_tokenize_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PART_SUCCESS_MARKER = SUCCESS_MARKER_FILE


# ---------------------------------------------------------------------------
# Recipe helpers
# ---------------------------------------------------------------------------

def get_recipe_fn(recipe_name: str):
    """Import and return ``prepare_{recipe_name}`` from ``lhotse.recipes``."""
    module = importlib.import_module(f"lhotse.recipes.{recipe_name}")
    fn_name = f"prepare_{recipe_name}"
    if not hasattr(module, fn_name):
        raise AttributeError(f"lhotse.recipes.{recipe_name} has no function '{fn_name}'")
    return getattr(module, fn_name)


def extract_manifests(manifests: dict, split: str, language: Optional[str] = None) -> dict:
    """Navigate a Lhotse recipe output to get the manifests for a specific split.

    Handles both flat ({split: ...}) and nested ({language: {split: ...}}) layouts.
    """
    if language and language in manifests:
        manifests = manifests[language]

    if split not in manifests:
        available = list(manifests.keys())
        raise KeyError(f"Split '{split}' not found. Available: {available}")

    return manifests[split]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def convert_worker(
    rank: int,
    my_cuts: list,
    *,
    shar_dir: Path,
    shar_format: str,
    shar_shard_size: int,
    min_sample_rate: int | None,
    target_sample_rate: int | None,
    text_tokenizer=None,
    stats_dir: Path | None = None,
):
    """Convert one partition of cuts to Shar format."""
    from lhotse import CutSet

    output_dir = shar_dir / f"part-{rank:05d}"
    if setup_partition_dir(
        output_dir,
        success_marker_name=PART_SUCCESS_MARKER,
        reuse_log=f"[worker {rank}] Reusing completed Shar in {output_dir}",
        reset_log=f"[worker {rank}] Removing partial Shar output in {output_dir}",
        logger=logger,
    ):
        return

    logger.info(f"[worker {rank}] Processing {len(my_cuts)} cuts")

    if len(my_cuts) == 0:
        logger.warning(f"[worker {rank}] Empty partition, skipping")
        mark_partition_success(output_dir, success_marker_name=PART_SUCCESS_MARKER)
        return

    cuts = CutSet.from_cuts(my_cuts)
    cuts = cuts.map(to_mono)

    if min_sample_rate:
        cuts = cuts.filter(lambda c: c.sampling_rate >= min_sample_rate)

    if text_tokenizer is not None:
        cuts = cuts.map(make_text_tokenize_fn(text_tokenizer))

    if target_sample_rate:
        cuts = cuts.resample(target_sample_rate)

    compute_rms, keep_loud = make_rms_filter_fn()
    cuts = cuts.map(compute_rms).filter(keep_loud)

    # Collect stats lazily as cuts flow through to_shar
    stats = {"num_cuts": 0, "total_duration": 0.0, "num_text_tokens": 0}

    def _collect_stats(cut):
        stats["num_cuts"] += 1
        stats["total_duration"] += cut.duration
        stats["num_text_tokens"] += len((cut.custom or {}).get("text_tokens", []))
        return cut

    cuts = cuts.map(_collect_stats)

    cuts.to_shar(
        output_dir=str(output_dir),
        fields={"recording": shar_format},
        shard_size=shar_shard_size,
        num_jobs=1,
        verbose=(rank == 0),
    )

    if stats_dir is not None:
        (stats_dir / f"part-{rank:05d}.json").write_text(json.dumps(stats))
    mark_partition_success(output_dir, success_marker_name=PART_SUCCESS_MARKER)
    logger.info(f"[worker {rank}] Done → {output_dir}")


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_shar_index(shar_root: Path, index_filename: str, world_size: int):
    part_dirs = [shar_root / f"part-{rank:05d}" for rank in range(world_size)]
    index_path, cuts_count = build_shar_index_from_parts(
        shar_root=shar_root,
        part_dirs=part_dirs,
        index_filename=index_filename,
        success_marker_name=PART_SUCCESS_MARKER,
    )
    logger.info(f"Wrote merged index: {index_path} ({cuts_count} cut shards)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only parallel Lhotse recipe → Shar conversion",
    )

    # Recipe source
    parser.add_argument("--recipe", required=True,
                        help="Lhotse recipe name (e.g. commonvoice, librispeech, voxpopuli)")
    parser.add_argument("--corpus_dir", type=Path, required=True,
                        help="Path to the extracted corpus (passed to prepare_*)")
    parser.add_argument("--split", required=True,
                        help="Dataset split (e.g. train, dev, test, other, validated)")
    parser.add_argument("--language", default=None,
                        help="Language key for recipes that return nested dicts (e.g. zh-CN, en)")
    parser.add_argument("--recipe_kwargs", default="{}",
                        help='Extra kwargs for the recipe as JSON (e.g. \'{"splits": ["other"]}\')')

    # Shar output
    parser.add_argument("--shar_base_dir", type=Path,
                        default=Path("/iopsstor/scratch/cscs/xyixuan/audio-datasets"))
    parser.add_argument(
        "--shar_output_dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit output directory. If set, this is used directly "
            "and --shar_base_dir + derived naming is skipped."
        ),
    )
    parser.add_argument("--shar_shard_size", type=int, default=1000)
    parser.add_argument("--shar_format", default="flac")
    parser.add_argument("--shar_index_filename", default="shar_index.json")

    # Audio processing
    parser.add_argument("--target_sample_rate", type=int, default=None)
    parser.add_argument("--min_sample_rate", type=int, default=None,
                        help="Drop cuts with sample rate below this threshold")

    # Cut segmentation
    parser.add_argument("--trim_to_supervisions", action="store_true", default=False,
                        help="Segment cuts to supervision boundaries (one cut per supervision). "
                             "Use for datasets with long recordings and many supervisions (e.g. meetings).")

    # Text tokenization
    parser.add_argument("--text_tokenizer", type=str, default=None,
                        help="Path to tokenizer.json for pre-tokenizing supervision text")

    # Parallelism
    parser.add_argument("--num_workers", type=int, default=64)

    return parser


def run(spec):
    """Execute lhotse_recipe prepare for a typed PrepareSpec."""
    num_workers = spec.output.num_workers if spec.output.num_workers is not None else 64
    if spec.output.num_workers is None:
        spec = spec.model_copy(
            update={
                "output": spec.output.model_copy(update={"num_workers": num_workers}),
            }
        )
    i, o, m = spec.input, spec.output, spec.metadata
    shar_dir = Path(o.shar_dir)

    extra_kwargs = json.loads(i.recipe_kwargs)

    validate_prepare_runtime(
        resampling_backend=None,
        require_ffmpeg=False,
        text_tokenizer_path=o.text_tokenizer,
    )

    shar_dir.mkdir(parents=True, exist_ok=True)
    write_prepare_state_for_spec(spec)

    # Step 1: Run Lhotse recipe to build manifests
    logger.info(f"Running lhotse recipe: prepare_{i.recipe}")
    recipe_fn = get_recipe_fn(i.recipe)

    manifest_dir = shar_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    recipe_kwargs = {"corpus_dir": Path(i.corpus_dir), "output_dir": manifest_dir, **extra_kwargs}
    recipe_kwargs.setdefault("num_jobs", num_workers)
    recipe_kwargs.setdefault("splits", [i.split])
    if m.language:
        recipe_kwargs.setdefault("languages", [m.language])

    manifests = recipe_fn(**recipe_kwargs)

    # Step 2: Extract the right split and build CutSet
    split_manifests = extract_manifests(manifests, i.split, m.language)

    from lhotse import CutSet
    cuts = CutSet.from_manifests(
        recordings=split_manifests["recordings"],
        supervisions=split_manifests.get("supervisions"),
    )

    if i.trim_to_supervisions:
        cuts = cuts.trim_to_supervisions(keep_overlapping=False)
        logger.info("Trimming cuts to supervision boundaries (one per supervision)")

    # Drop cuts whose supervisions have no text
    cuts = cuts.filter(
        lambda c: any(s.text and s.text.strip() for s in (c.supervisions or []))
    )

    cuts_list = list(cuts)
    cuts_list = assign_universal_ids(cuts_list, store_clip_start=True)
    logger.info(f"Built CutSet with {len(cuts_list)} cuts from {i.recipe}/{i.split}")
    logger.info(f"Converting to Shar → {shar_dir}")
    logger.info(f"Using {num_workers} parallel workers")

    text_tokenizer = load_text_tokenizer(o.text_tokenizer)

    with tempfile.TemporaryDirectory(prefix="shar_stats_") as stats_dir:
        stats_dir = Path(stats_dir)
        procs = [
            Process(
                target=convert_worker,
                args=(rank, cuts_list[rank::num_workers]),
                kwargs=dict(
                    shar_dir=shar_dir,
                    shar_format=o.shar_format,
                    shar_shard_size=o.shard_size,
                    min_sample_rate=i.min_sample_rate,
                    target_sample_rate=o.target_sr,
                    text_tokenizer=text_tokenizer,
                    stats_dir=stats_dir,
                ),
            )
            for rank in range(num_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        failed = [rank for rank, p in enumerate(procs) if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Workers {failed} failed")

        build_shar_index(shar_dir, i.shar_index_filename, num_workers)

        from audio_tokenization.prepare.validate_shar import validate_shar_directory
        counts = validate_shar_directory(shar_dir, index_filename=i.shar_index_filename)
        logger.info(
            "Validated SHAR: %d cuts across %d shards",
            sum(counts.values()), len(counts),
        )

        mark_partition_success(shar_dir, success_marker_name=PART_SUCCESS_MARKER)

        total = {"num_cuts": 0, "total_duration": 0.0, "num_text_tokens": 0}
        for stats_path in sorted(stats_dir.glob("*.json")):
            ws = json.loads(stats_path.read_text())
            for k in total:
                total[k] += ws.get(k, 0)

        n = total["num_cuts"] or 1
        hours = total["total_duration"] / 3600
        avg_dur = total["total_duration"] / n
        summary = f"Summary: {total['num_cuts']:,} cuts, {hours:.1f} hours (avg {avg_dur:.1f}s/cut)"
        if total["num_text_tokens"]:
            avg_tok = total["num_text_tokens"] / n
            summary += f", {total['num_text_tokens']:,} text tokens (avg {avg_tok:.1f}/cut)"
        logger.info(summary)

    logger.info("All done!")


def _args_to_spec(args):
    """Translate flat argparse Namespace → typed PrepareSpec.

    The legacy CLI's ``--shar_output_dir`` / ``--shar_base_dir`` derivation
    happens here so the typed runner can assume a concrete ``shar_dir``.
    """
    from audio_tokenization.config.schema import PrepareSpec

    if args.shar_output_dir is not None:
        shar_dir = args.shar_output_dir
    else:
        parts = [args.recipe]
        if args.language:
            parts.append(args.language)
        parts.append(args.split)
        shar_dir = args.shar_base_dir / "_".join(parts)

    return PrepareSpec.from_mapping({
        "family": "lhotse_recipe",
        "input": {
            "recipe": args.recipe,
            "corpus_dir": str(args.corpus_dir),
            "split": args.split,
            "recipe_kwargs": args.recipe_kwargs,
            "min_sample_rate": args.min_sample_rate,
            "trim_to_supervisions": args.trim_to_supervisions,
            "shar_index_filename": args.shar_index_filename,
        },
        "output": {
            "shar_dir": str(shar_dir),
            "shard_size": args.shar_shard_size,
            "shar_format": args.shar_format,
            "target_sr": args.target_sample_rate,
            "text_tokenizer": args.text_tokenizer,
            "num_workers": args.num_workers,
        },
        "metadata": {
            "language": args.language,
        },
    })


def main(argv=None):
    return run(_args_to_spec(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
