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
    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
        --recipe commonvoice \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24 \
        --split other \
        --language zh-CN \
        --target_sample_rate 24000 \
        --num_workers 64

    # Common Voice es train with explicit output dir name
    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
        --recipe commonvoice \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24 \
        --split train \
        --language es \
        --shar_base_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/commonvoice \
        --shar_output_dir /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/commonvoice/es_train

    # LibriSpeech (flat)
    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
        --recipe librispeech \
        --corpus_dir /path/to/LibriSpeech \
        --split train-clean-360 \
        --num_workers 32

    # VoxPopuli (recipe-specific kwargs still needed for non-standard params)
    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
        --recipe voxpopuli \
        --corpus_dir /path/to/voxpopuli \
        --split train \
        --language en \
        --recipe_kwargs '{"task": "asr"}' \
        --num_workers 32

    # Thorsten-DE (single-split dataset, use --split all)
    PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:$PYTHONPATH \
    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
        --recipe thorsten_de \
        --corpus_dir /capstor/store/cscs/swissai/infra01/audio-datasets/raw/thorsten-de \
        --split all \
        --shar_base_dir /iopsstor/scratch/cscs/xyixuan/audio-datasets \
        --text_tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json \
        --num_workers 64

    # AISHELL-1 (run once per split: train, dev, test)
    PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:$PYTHONPATH \
    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
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
    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
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
    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
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
    python -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
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

from audio_tokenization.utils.prepare_data.common import (
    PREPARE_STATE_FILE,
    SUCCESS_MARKER_FILE,
    build_shar_index_from_parts,
    load_text_tokenizer,
    make_rms_filter_fn,
    make_text_tokenize_fn,
    mark_partition_success,
    normalize_optional_path,
    setup_partition_dir,
    to_mono,
    validate_or_write_prepare_state,
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

def convert_worker(rank: int, my_cuts: list, args,
                   text_tokenizer=None, stats_dir: Path | None = None):
    """Convert one partition of cuts to Shar format."""
    from lhotse import CutSet

    output_dir = args.shar_dir / f"part-{rank:05d}"
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

    if args.min_sample_rate:
        cuts = cuts.filter(lambda c: c.sampling_rate >= args.min_sample_rate)

    if text_tokenizer is not None:
        cuts = cuts.map(make_text_tokenize_fn(text_tokenizer))

    if args.target_sample_rate:
        cuts = cuts.resample(args.target_sample_rate)

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
        fields={"recording": args.shar_format},
        shard_size=args.shar_shard_size,
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

def _validate_or_write_prepare_state(args) -> None:
    state_path = args.shar_dir / PREPARE_STATE_FILE
    expected = {
        "recipe": args.recipe,
        "corpus_dir": str(args.corpus_dir),
        "split": args.split,
        "language": args.language,
        "recipe_kwargs": args.recipe_kwargs,
        "text_tokenizer": normalize_optional_path(args.text_tokenizer),
        "num_workers": int(args.num_workers),
    }
    wrote = validate_or_write_prepare_state(
        state_path,
        expected=expected,
        invariant_keys=("recipe", "corpus_dir", "split", "language", "recipe_kwargs", "text_tokenizer", "num_workers"),
        guidance=(
            "Use the same --recipe, --corpus_dir, --split, --language, --recipe_kwargs, "
            f"--text_tokenizer, and --num_workers to resume, or remove {args.shar_dir} "
            "and restart from scratch."
        ),
    )
    if wrote:
        logger.info(f"Wrote prepare state: {state_path}")


def main():
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

    args = parser.parse_args()
    extra_kwargs = json.loads(args.recipe_kwargs)

    if args.shar_output_dir is not None:
        args.shar_dir = args.shar_output_dir
    else:
        # Derive output directory: commonvoice_zh-CN_other
        parts = [args.recipe]
        if args.language:
            parts.append(args.language)
        parts.append(args.split)
        args.shar_dir = args.shar_base_dir / f"{'_'.join(parts)}"
    args.shar_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_prepare_state(args)

    # Step 1: Run Lhotse recipe to build manifests
    logger.info(f"Running lhotse recipe: prepare_{args.recipe}")
    recipe_fn = get_recipe_fn(args.recipe)

    manifest_dir = args.shar_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    recipe_kwargs = {"corpus_dir": args.corpus_dir, "output_dir": manifest_dir, **extra_kwargs}
    recipe_kwargs.setdefault("num_jobs", args.num_workers)
    recipe_kwargs.setdefault("splits", [args.split])
    if args.language:
        recipe_kwargs.setdefault("languages", [args.language])

    manifests = recipe_fn(**recipe_kwargs)

    # Step 2: Extract the right split and build CutSet
    split_manifests = extract_manifests(manifests, args.split, args.language)

    from lhotse import CutSet
    cuts = CutSet.from_manifests(
        recordings=split_manifests["recordings"],
        supervisions=split_manifests.get("supervisions"),
    )

    if args.trim_to_supervisions:
        cuts = cuts.trim_to_supervisions(keep_overlapping=False)
        logger.info("Trimming cuts to supervision boundaries (one per supervision)")

    # Drop cuts whose supervisions have no text
    cuts = cuts.filter(
        lambda c: any(s.text and s.text.strip() for s in (c.supervisions or []))
    )

    cuts_list = list(cuts)
    logger.info(f"Built CutSet with {len(cuts_list)} cuts from {args.recipe}/{args.split}")
    logger.info(f"Converting to Shar → {args.shar_dir}")
    logger.info(f"Using {args.num_workers} parallel workers")

    # Load text tokenizer before forking (shared via COW across workers)
    text_tokenizer = load_text_tokenizer(args.text_tokenizer)

    # Step 3: Spawn workers (stats written to a temp dir, cleaned up automatically)
    with tempfile.TemporaryDirectory(prefix="shar_stats_") as stats_dir:
        stats_dir = Path(stats_dir)
        procs = [
            Process(target=convert_worker,
                    args=(i, cuts_list[i::args.num_workers], args, text_tokenizer, stats_dir))
            for i in range(args.num_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Workers {failed} failed")

        # Merge all part-* into a single index
        build_shar_index(args.shar_dir, args.shar_index_filename, args.num_workers)
        mark_partition_success(args.shar_dir, success_marker_name=PART_SUCCESS_MARKER)

        # Aggregate worker stats
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
    # TemporaryDirectory cleaned up here

    logger.info("All done!")


if __name__ == "__main__":
    main()
