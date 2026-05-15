"""Runtime and filesystem helpers for prepare scripts."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from audio_tokenization.contracts.artifacts import SHAR_INDEX_FILENAME, SUCCESS_MARKER_FILE
from audio_tokenization.prepare.constants import (
    PREPARE_SUMMARY_FILE,
    WORKER_STATS_FILE,
)
from audio_tokenization.utils.io import atomic_write_json, write_success_marker
from audio_tokenization.utils.stats import load_json_records, max_field, sum_counter_fields


logger = logging.getLogger(__name__)


_PREPARE_TOTAL_FIELDS = ("written", "skipped", "errors", "total_duration_sec")


def maybe_log_worker_progress(
    *,
    logger,
    worker_id: int,
    written: int,
    skipped: int,
    errors: int,
    t0: float,
    next_log_at: int,
    interval: int = 1000,
) -> int:
    """Log worker progress after crossing a write-count threshold.

    VAD conversion is one-to-many: one input recording can emit many output
    cuts. Exact modulo checks miss progress when ``written`` jumps past a
    threshold, so callers carry the next threshold explicitly.
    """
    if interval <= 0:
        raise ValueError("interval must be > 0")
    if written < next_log_at:
        return next_log_at

    elapsed = max(time.time() - t0, 1e-9)
    logger.info(
        "Worker %s: %d written, %d skipped, %d errors (%.1f samples/s)",
        worker_id,
        written,
        skipped,
        errors,
        written / elapsed,
    )
    return ((written // interval) + 1) * interval


def validate_prepare_runtime(
    *,
    resampling_backend: str | None = None,
    require_ffmpeg: bool = False,
    text_tokenizer_path: str | Path | None = None,
) -> None:
    """Fail fast on runtime prerequisites before worker startup."""
    from audio_tokenization.prepare.text_ops import load_text_tokenizer

    require_lhotse_shar_features()
    init_worker_process(resampling_backend)

    if text_tokenizer_path is not None:
        load_text_tokenizer(text_tokenizer_path)

    if require_ffmpeg and shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for audio-bytes decode fallback but was not found on PATH. "
            "Set PATH/LD_LIBRARY_PATH like the dataset SLURM script, or add a pinned ffmpeg "
            "runtime before starting prepare_parquet_to_shar."
        )


def require_lhotse_shar_features() -> None:
    """Fail early when the runtime is not the dev Lhotse required by prepare.

    Convert now relies on two SHAR APIs that are intentionally owned by our
    Lhotse fork: metadata-only tar iteration for validation, and atomic SHAR
    commit for durable writes. Falling back to older packaged Lhotse would
    silently weaken the stage contract, so make the dependency explicit.
    """
    missing: list[str] = []
    try:
        from lhotse import CutSet
        import lhotse.shar as shar
    except ImportError as exc:
        raise RuntimeError(_dev_lhotse_error(["lhotse import"])) from exc

    if not hasattr(shar, "iterate_tarfile_pairwise_metadata"):
        missing.append("lhotse.shar.iterate_tarfile_pairwise_metadata")

    shar_writer = getattr(shar, "SharWriter", None)
    if shar_writer is None or not _accepts_keyword(shar_writer, "commit"):
        missing.append("lhotse.shar.SharWriter(commit=...)")

    to_shar = getattr(CutSet, "to_shar", None)
    if to_shar is None or not _accepts_keyword(to_shar, "commit"):
        missing.append("lhotse.CutSet.to_shar(commit=...)")

    if missing:
        raise RuntimeError(_dev_lhotse_error(missing))


def _accepts_keyword(fn: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if keyword in signature.parameters:
        return True
    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _dev_lhotse_error(missing: Sequence[str]) -> str:
    missing_msg = ", ".join(missing)
    return (
        "audio-tokenization prepare requires dev Lhotse with SHAR atomic commit "
        f"support; missing: {missing_msg}. Source "
        "`scripts/utils/source_lhotse_runtime.sh` before running convert, or set "
        "`LHOTSE_DIR` to a checkout that provides these APIs."
    )


def get_prepare_runner(spec):
    """Import the runner module declared by the prepare input spec."""
    return importlib.import_module(spec.input.RUNNER_MODULE)


def resolve_prepare_inputs(spec) -> tuple[list[str], dict[str, Any]]:
    """Resolve raw prepare inputs through the family runner."""
    return get_prepare_runner(spec).resolve(spec)


def coerce_resolved_inputs(spec, resolved_inputs: list[str] | None) -> list[str]:
    """Use the stage-supplied input list when provided; otherwise resolve once.

    Lets CLI-only runners (no upstream resolver) and stage-driven runs share one
    code path while guaranteeing the actual glob runs exactly once per invocation.
    """
    if resolved_inputs is not None:
        return list(resolved_inputs)
    return resolve_prepare_inputs(spec)[0]


def preflight_prepare_spec(
    spec,
    *,
    runtime_validator: Callable[..., None] = validate_prepare_runtime,
    resolved_inputs: list[str] | None = None,
) -> None:
    """Run generic prepare preflight through the family runner."""
    return get_prepare_runner(spec).preflight(
        spec,
        runtime_validator=runtime_validator,
        resolved_inputs=resolved_inputs,
    )


def setup_partition_dir(
    part_dir: Path,
    *,
    worker_id: int | None = None,
    logger=None,
) -> None:
    """Wipe and recreate a partition directory for fresh writing.

    The stage adapter owns skip/overwrite semantics; partition writers are
    invoked only when the stage has already decided to (re)build.

    Logs the wipe with worker_id context when both *worker_id* and *logger*
    are provided. Callers that don't have a worker_id (or don't care about
    the log) can omit either.
    """
    if part_dir.is_dir():
        if logger is not None and worker_id is not None:
            logger.warning(
                "Worker %s: removing partial output in %s", worker_id, part_dir,
            )
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)


def mark_partition_success(part_dir: Path) -> None:
    """Atomically mark a per-worker partition as complete.

    Distinct contract from stage-root ``run_stage`` completion: this signals
    that *this worker's slice* is done, not that the whole stage is.
    """
    write_success_marker(part_dir)


def resolve_num_workers(requested: int | None, *, num_inputs: int | None = None) -> int:
    """Resolve a worker count from the user request and the available resources.

    - ``requested`` is None → use ``SLURM_CPUS_PER_TASK`` if running under
      SLURM, else ``os.cpu_count()``.
    - Result is clamped to ``num_inputs`` when provided so the pool never
      spawns more processes than there is work for.
    - Result is at least 1.

    Why this exists: defaulting to "one worker per input shard" silently OOMs
    on large datasets (e.g. SRG Apertus has 1361 parquet shards; each lhotse-
    loaded forkserver worker is ~1+ GB). Defaulting to "the resources you
    actually allocated" is the only safe behaviour.
    """
    if requested is None:
        cap = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    else:
        cap = int(requested)
    if num_inputs is not None:
        cap = min(cap, num_inputs)
    return max(1, cap)


def build_shar_index_from_parts(
    *,
    shar_root: Path,
    part_dirs: Iterable[Path],
    index_filename: str,
) -> tuple[Path, int]:
    """Build a merged ``shar_index.json`` from expected partition directories."""
    fields = defaultdict(list)
    shar_root = shar_root.resolve()

    for part_dir in part_dirs:
        if not part_dir.is_dir():
            raise FileNotFoundError(f"Missing partition directory: {part_dir}")

        success_marker = part_dir / SUCCESS_MARKER_FILE
        if not success_marker.is_file():
            raise RuntimeError(
                f"Missing completion marker in {part_dir}. "
                "Partial partition detected; rerun the stage with overwrite=True."
            )

        for p in sorted(part_dir.iterdir()):
            if not p.is_file() or p.name == SUCCESS_MARKER_FILE:
                continue
            abs_p = p.resolve()
            try:
                index_path = str(abs_p.relative_to(shar_root))
            except ValueError as e:
                raise RuntimeError(
                    f"Index entry is outside shar_root and cannot be made relative: {abs_p} "
                    f"(shar_root={shar_root})"
                ) from e
            field = p.name.split(".")[0]
            if field == "cuts" and p.suffix == ".gz":
                fields["cuts"].append(index_path)
            elif p.suffix in (".tar", ".gz"):
                fields[field].append(index_path)

    if not fields.get("cuts"):
        raise FileNotFoundError(f"No Shar cuts found under {shar_root}")

    payload = {
        "version": 1,
        "fields": {k: sorted(v) for k, v in fields.items()},
    }
    index_path = shar_root / index_filename
    atomic_write_json(index_path, payload)
    return index_path, len(fields["cuts"])


def audio_md5(path: str) -> str:
    """MD5 of decoded audio waveform (float32 PCM, not raw file bytes)."""
    import soundfile as sf

    data, _ = sf.read(path, dtype="float32")
    return hashlib.md5(data.tobytes()).hexdigest()


def build_audio_index(audio_root: Path, pattern: str = "**/*.ogg") -> dict[str, str]:
    """Map lowercased file stems to full paths (recursive glob)."""
    return {p.stem.lower(): str(p) for p in audio_root.glob(pattern)}


def distribute_round_robin(items: Sequence, num_workers: int) -> list[list]:
    """Distribute items across workers in round-robin order."""
    buckets: list[list] = [[] for _ in range(num_workers)]
    for i, item in enumerate(items):
        buckets[i % num_workers].append(item)
    return buckets


def build_shar_index(
    shar_root: Path,
    num_workers: int,
    index_filename: str = SHAR_INDEX_FILENAME,
    worker_dir_fmt: str = "worker_{:02d}",
) -> None:
    """Build a merged ``shar_index.json`` from all ``worker_*`` directories."""
    worker_dirs = [shar_root / worker_dir_fmt.format(wid) for wid in range(num_workers)]
    build_shar_index_for_worker_dirs(
        shar_root,
        worker_dirs,
        index_filename=index_filename,
    )


def build_shar_index_for_worker_dirs(
    shar_root: Path,
    worker_dirs: Sequence[Path],
    index_filename: str = SHAR_INDEX_FILENAME,
) -> None:
    """Build a merged ``shar_index.json`` from completed worker directories."""
    index_path, cuts_count = build_shar_index_from_parts(
        shar_root=shar_root,
        part_dirs=worker_dirs,
        index_filename=index_filename,
    )
    logger.info(f"Wrote merged index: {index_path} ({cuts_count} cut shards)")


def init_worker_process(resampling_backend: str | None = None) -> None:
    """Per-process initialisation for pool workers."""
    from lhotse.audio.resampling_backend import (
        available_resampling_backends,
        set_current_resampling_backend,
    )

    backend = resampling_backend or os.environ.get(
        "LHOTSE_RESAMPLING_BACKEND", "soxr"
    )
    if backend == "torchaudio":
        backend = "default"
    if backend not in available_resampling_backends():
        raise RuntimeError(
            f"Resampling backend {backend!r} not available. "
            f"Installed: {available_resampling_backends()}"
        )
    set_current_resampling_backend(backend)


def write_worker_result(
    *,
    worker_id: int,
    worker_dir: Path,
    written: int,
    skipped: int,
    errors: int,
    total_duration_sec: float,
    runtime_counts: Counter,
    t0: float,
    extra_stats: dict | None = None,
) -> dict:
    """Log completion, persist worker stats, mark success, and return result dict."""
    import time as _time

    elapsed = _time.time() - t0
    logger.info(
        f"Worker {worker_id} done in {elapsed:.1f}s: "
        f"{written} written, {skipped} skipped, {errors} errors"
    )

    worker_stats: dict = {
        "worker_id": worker_id,
        "elapsed_sec": elapsed,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "total_duration_sec": total_duration_sec,
        "runtime_counts": dict(runtime_counts),
    }
    if extra_stats:
        worker_stats.update(extra_stats)

    worker_stats_path = worker_dir / WORKER_STATS_FILE
    atomic_write_json(worker_stats_path, worker_stats, sort_keys=False)

    mark_partition_success(worker_dir)

    result: dict = {
        "worker_id": worker_id,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "elapsed": elapsed,
        "total_duration_sec": total_duration_sec,
        "worker_stats": worker_stats,
    }
    if extra_stats:
        result.update(extra_stats)
    return result


def build_prepare_summary(
    results: list[dict],
    *,
    elapsed_sec: float,
    num_workers: int,
) -> dict:
    """Aggregate prepare worker results into the durable summary schema."""
    totals = {
        key: sum((result.get(key) or 0) for result in results)
        for key in _PREPARE_TOTAL_FIELDS
    }
    reason_counts = sum_counter_fields(
        results,
        "reason_counts",
        ("worker_stats", "reason_counts"),
    )
    runtime_counts = sum_counter_fields(results, ("worker_stats", "runtime_counts"))

    return {
        "version": 1,
        "num_workers": num_workers,
        "elapsed_sec": elapsed_sec,
        "total_written": int(totals["written"]),
        "total_skipped": int(totals["skipped"]),
        "total_errors": int(totals["errors"]),
        "total_duration_sec": float(totals["total_duration_sec"]),
        "runtime_counts": runtime_counts,
        "reason_counts": reason_counts,
        "results": results,
    }


def build_prepare_rollup(summaries: list[dict]) -> dict:
    """Aggregate node-level prepare summaries for operator reporting."""
    return {
        "num_nodes": len(summaries),
        "total_written": int(sum(s.get("total_written", 0) for s in summaries)),
        "total_skipped": int(sum(s.get("total_skipped", 0) for s in summaries)),
        "total_errors": int(sum(s.get("total_errors", 0) for s in summaries)),
        "total_duration_sec": float(sum(s.get("total_duration_sec", 0.0) for s in summaries)),
        "max_elapsed_sec": max_field(summaries, "elapsed_sec"),
        "reason_counts": sum_counter_fields(summaries, "reason_counts"),
        "runtime_counts": sum_counter_fields(summaries, "runtime_counts"),
    }


def run_pool_and_finalize(
    worker_fn,
    worker_args: list,
    shar_dir: Path,
    num_workers: int,
    mp_start_method: str = "forkserver",
) -> list[dict]:
    """Run *worker_fn* in a multiprocessing pool, aggregate stats, write summary & index."""
    import multiprocessing as _mp
    import time as _time

    if not worker_args:
        raise ValueError("worker_args must be non-empty")

    available_methods = _mp.get_all_start_methods()
    if mp_start_method not in available_methods:
        raise ValueError(
            f"Unsupported multiprocessing start method: {mp_start_method!r}. "
            f"Available methods: {available_methods}"
        )

    logger.info(
        "Starting worker pool with start_method=%s, processes=%d",
        mp_start_method,
        len(worker_args),
    )
    t0 = _time.time()
    ctx = _mp.get_context(mp_start_method)
    results = []
    with ctx.Pool(processes=len(worker_args)) as pool:
        for result in pool.imap_unordered(worker_fn, worker_args):
            results.append(result)
            logger.info(
                "Worker %s finished (%d/%d)",
                result.get("worker_id", "?"),
                len(results),
                len(worker_args),
            )

    elapsed = _time.time() - t0
    summary = build_prepare_summary(
        results,
        elapsed_sec=elapsed,
        num_workers=num_workers,
    )
    total_written = summary["total_written"]
    total_skipped = summary["total_skipped"]
    total_errors = summary["total_errors"]
    total_duration_sec = summary["total_duration_sec"]
    total_reason_counts = summary["reason_counts"]
    total_runtime_counts = summary["runtime_counts"]

    logger.info(
        f"All workers done in {elapsed:.1f}s — "
        f"{total_written} samples, {total_skipped} skipped, {total_errors} errors, "
        f"{total_duration_sec / 3600.0:.1f} hours written"
    )
    if total_reason_counts:
        logger.info(f"VAD reasons (global): {dict(total_reason_counts)}")
    if total_runtime_counts:
        logger.info(f"Runtime counters (global): {dict(total_runtime_counts)}")

    summary_path = Path(shar_dir) / PREPARE_SUMMARY_FILE
    atomic_write_json(summary_path, summary, sort_keys=False)
    logger.info(f"Wrote prepare summary: {summary_path}")

    worker_dirs = [
        Path(shar_dir) / f"worker_{int(result['worker_id']):02d}"
        for result in sorted(results, key=lambda r: int(r["worker_id"]))
    ]
    build_shar_index_for_worker_dirs(Path(shar_dir), worker_dirs)

    from audio_tokenization.prepare.validate_shar import (
        validate_shar_directory,
    )
    counts = validate_shar_directory(Path(shar_dir))
    logger.info(
        "Validated SHAR: %d cuts across %d shards", sum(counts.values()), len(counts)
    )
    for shard_name, n in counts.items():
        logger.debug("  %s: %d cuts", shard_name, n)

    # Stage-root _SUCCESS is owned exclusively by run_stage(stage="convert").
    # Prepare runners only mark per-worker partitions complete.
    logger.info("All done!")
    return results


def run_aggregate(shar_root: Path) -> None:
    """Read prepare_summary.json from all node_*/ dirs, sum totals, and print."""
    node_dirs = sorted(shar_root.glob("node_*"))
    if not node_dirs:
        single = shar_root / PREPARE_SUMMARY_FILE
        if single.is_file():
            node_dirs = [shar_root]
        else:
            raise FileNotFoundError(
                f"No node_*/ dirs (or {PREPARE_SUMMARY_FILE}) found under {shar_root}"
            )

    summary_paths = []
    for nd in node_dirs:
        sp = nd / PREPARE_SUMMARY_FILE
        if not sp.is_file():
            logger.warning(f"Missing {sp}, skipping")
            continue
        summary_paths.append(sp)
    summaries = load_json_records(summary_paths, logger=logger)

    if not summaries:
        raise FileNotFoundError(f"No {PREPARE_SUMMARY_FILE} found in any node dir")

    rollup = build_prepare_rollup(summaries)
    total_hours = rollup["total_duration_sec"] / 3600.0

    print()
    print(f"=== Aggregate stats from {len(summaries)} node(s) under {shar_root} ===")
    print(f"  Samples written:  {rollup['total_written']:>12d}")
    print(f"  Samples skipped:  {rollup['total_skipped']:>12d}")
    print(f"  Errors:           {rollup['total_errors']:>12d}")
    print(f"  Total hours:      {total_hours:>12.1f}")
    print(f"  Max wall-time:    {rollup['max_elapsed_sec']:>12.1f}s")
    if rollup["reason_counts"]:
        print(f"  VAD reasons:      {rollup['reason_counts']}")
    if rollup["runtime_counts"]:
        print(f"  Runtime counters: {rollup['runtime_counts']}")
    print()
