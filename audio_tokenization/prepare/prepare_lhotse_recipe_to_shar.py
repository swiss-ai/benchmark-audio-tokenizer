"""Convert any Lhotse-supported dataset to Lhotse Shar format.

Uses Lhotse's built-in recipes (100+ datasets) to materialize manifests, then
spawns N parallel workers to convert to Shar. Each worker takes its interleaved
partition of the CutSet and writes to ``part-{rank:05d}/``. After all workers
finish, builds a merged ``shar_index.json``.

Lhotse recipes return manifests in two shapes:
  - Flat:   {split: {"recordings": ..., "supervisions": ...}}
  - Nested: {language: {split: {"recordings": ..., "supervisions": ...}}}
Set ``metadata.language`` to navigate the nested case.

Invocation goes through the Hydra stage adapter:
``python -m audio_tokenization run dataset=<name> stage=convert`` with a
``configs/pipeline/dataset/<name>.yaml`` that picks the lhotse_recipe recipe.
"""

import importlib
import json
import logging
import tempfile
from multiprocessing import Process
from pathlib import Path
from typing import Optional

from audio_tokenization.contracts.artifacts import SHAR_INDEX_FILENAME
from audio_tokenization.prepare.audio_ops import make_rms_filter_fn, to_mono
from audio_tokenization.prepare.constants import PREPARE_SHAR_COMMIT_MODE
from audio_tokenization.prepare.identity import assign_interleave_metadata
from audio_tokenization.prepare.runtime import (
    build_shar_index_from_parts,
    mark_partition_success,
    setup_partition_dir,
    validate_prepare_runtime,
)
from audio_tokenization.prepare.text_ops import (
    load_text_tokenizer,
    make_text_tokenize_fn,
)
from audio_tokenization.utils.io import atomic_write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

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
    setup_partition_dir(output_dir, worker_id=rank, logger=logger)

    logger.info(f"[worker {rank}] Processing {len(my_cuts)} cuts")

    if len(my_cuts) == 0:
        logger.warning(f"[worker {rank}] Empty partition, skipping")
        mark_partition_success(output_dir)
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
        commit=PREPARE_SHAR_COMMIT_MODE,
    )

    if stats_dir is not None:
        atomic_write_json(stats_dir / f"part-{rank:05d}.json", stats, indent=None)
    mark_partition_success(output_dir)
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
    )
    logger.info(f"Wrote merged index: {index_path} ({cuts_count} cut shards)")


def resolve(spec) -> tuple[list[str], dict]:
    """Lhotse recipes materialize their own file list at runtime."""
    i = spec.input
    return [], {
        "family": spec.family,
        "recipe": i.recipe,
        "corpus_dir": i.corpus_dir,
        "split": i.split,
        "language": spec.metadata.language,
    }


def preflight(
    spec,
    *,
    runtime_validator=validate_prepare_runtime,
    resolved_inputs: list[str] | None = None,
) -> None:
    """Validate generic lhotse_recipe prepare prerequisites."""
    del resolved_inputs
    i, o = spec.input, spec.output
    corpus_dir = Path(i.corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Lhotse corpus dir not found: {corpus_dir}")
    json.loads(i.recipe_kwargs)
    runtime_validator(
        resampling_backend=None,
        require_ffmpeg=False,
        text_tokenizer_path=o.text_tokenizer,
    )


def run(spec, *, resolved_inputs: list[str] | None = None):
    """Execute lhotse_recipe prepare for a typed PrepareSpec."""
    del resolved_inputs
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

    shar_dir.mkdir(parents=True, exist_ok=True)

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
    cuts_list = assign_interleave_metadata(cuts_list, store_clip_start=True)
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

        # Stage-root _SUCCESS is owned exclusively by run_stage(stage="convert").
        # The lhotse_recipe runner only marks its per-worker partitions complete.

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
