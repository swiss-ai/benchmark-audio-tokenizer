"""Runtime and filesystem helpers for prepare_data."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from audio_tokenization.prepare.constants import (
    CURRENT_PREPARE_STATE_VERSION,
    PREPARE_STATE_FILE,
    PREPARE_SUMMARY_FILE,
    SUCCESS_MARKER_FILE,
    WORKER_ASSIGNMENT_FILE,
    WORKER_STATS_FILE,
)


logger = logging.getLogger(__name__)


def validate_prepare_runtime(
    *,
    resampling_backend: str | None = None,
    require_ffmpeg: bool = False,
    text_tokenizer_path: str | Path | None = None,
) -> None:
    """Fail fast on runtime prerequisites before worker startup."""
    from audio_tokenization.prepare.text_ops import load_text_tokenizer

    init_worker_process(resampling_backend)

    if text_tokenizer_path is not None:
        load_text_tokenizer(text_tokenizer_path)

    if require_ffmpeg and shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for audio-bytes decode fallback but was not found on PATH. "
            "Set PATH/LD_LIBRARY_PATH like the dataset SLURM script, or add a pinned ffmpeg "
            "runtime before starting prepare_parquet_to_shar."
        )


def setup_partition_dir(
    part_dir: Path,
    *,
    success_marker_name: str = SUCCESS_MARKER_FILE,
    reuse_log: str | None = None,
    reset_log: str | None = None,
    logger=None,
) -> bool:
    """Prepare a partition directory for resume-safe writing."""
    success_marker = part_dir / success_marker_name
    if success_marker.is_file():
        if logger and reuse_log:
            logger.info(reuse_log)
        return True

    if part_dir.is_dir():
        if logger and reset_log:
            logger.warning(reset_log)
        shutil.rmtree(part_dir)

    part_dir.mkdir(parents=True, exist_ok=True)
    return False


def mark_partition_success(
    part_dir: Path,
    *,
    success_marker_name: str = SUCCESS_MARKER_FILE,
) -> None:
    """Atomically mark a partition as fully prepared."""
    (part_dir / success_marker_name).write_text("ok\n")


def _migrate_prepare_state_v0_to_v1(payload: dict) -> dict:
    upgraded = dict(payload)
    upgraded["version"] = 1
    return upgraded


_PREPARE_STATE_MIGRATIONS = {
    (0, 1): _migrate_prepare_state_v0_to_v1,
}


def _atomic_write_state_json(state_path: Path, payload: dict) -> None:
    """Write ``payload`` to ``state_path`` atomically, with fsync for durability.

    Why fsync: cscs Lustre scratch can leave a zero-byte file after node
    failure if the data isn't flushed before the metadata rename. Losing the
    state file silently downgrades the next run to "first run" and re-prepares
    the entire dataset.
    """
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp_path, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, state_path)


def read_prepare_state(state_path: Path) -> dict:
    """Read a prepare state file, auto-upgrading v0 (unversioned) in place.

    Raises:
        FileNotFoundError: if ``state_path`` does not exist.
        RuntimeError: if the file has an unknown future version, a payload
            that isn't a dict, or a missing migration step.
    """
    if not state_path.is_file():
        raise FileNotFoundError(state_path)

    payload = json.loads(state_path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid prepare state format: {state_path}")

    version = payload.get("version", 0)
    if not isinstance(version, int):
        raise RuntimeError(
            f"Invalid prepare state at {state_path}: 'version' must be an int, "
            f"got {type(version).__name__}"
        )

    if version > CURRENT_PREPARE_STATE_VERSION:
        raise RuntimeError(
            f"Prepare state at {state_path} is version {version}, but this code "
            f"only knows how to read up to version {CURRENT_PREPARE_STATE_VERSION}. "
            "Upgrade audio_tokenization to a newer release, or delete the state "
            "file to start a fresh prepare run."
        )

    upgraded = False
    while version < CURRENT_PREPARE_STATE_VERSION:
        step = (version, version + 1)
        migrate = _PREPARE_STATE_MIGRATIONS.get(step)
        if migrate is None:
            raise RuntimeError(
                f"Missing prepare-state migration for {step}; this is a bug. "
                f"State file: {state_path}"
            )
        payload = migrate(payload)
        version += 1
        upgraded = True

    if upgraded:
        _atomic_write_state_json(state_path, payload)
        logger.info(
            "Upgraded %s to prepare-state v%d (in place).",
            state_path.name, CURRENT_PREPARE_STATE_VERSION,
        )

    return payload


def diff_fingerprint(
    expected: Mapping[str, object], on_disk: Mapping[str, object]
) -> dict[str, tuple[object, object]]:
    """{key: (expected, actual)} pairs that disagree.

    Ignores keys in *on_disk* that aren't part of the expected fingerprint
    (the ``version`` sentinel the state writer injects, plus any future
    additive on-disk fields).
    """
    drift: dict[str, tuple[object, object]] = {}
    for k, v in expected.items():
        if k not in on_disk:
            drift[k] = (v, "<missing>")
        elif on_disk[k] != v:
            drift[k] = (v, on_disk[k])
    return drift


def validate_or_write_prepare_state(
    state_path: Path,
    *,
    expected: Mapping[str, object],
    invariant_keys: Sequence[str],
    guidance: str,
) -> bool:
    """Persist first-run state or assert resume invariants on later runs.

    Backfill: if a key in ``invariant_keys`` is missing from the on-disk
    payload (e.g. expanded invariant set on a v0 file), the current value is
    accepted and persisted; subsequent runs then enforce strict equality.
    Returns ``True`` on first write, ``False`` on validated resume.
    """
    if state_path.is_file():
        payload = read_prepare_state(state_path)
        backfilled: list[str] = []

        for key in invariant_keys:
            if key not in payload:
                payload[key] = expected.get(key)
                backfilled.append(key)
                continue
            prev = payload[key]
            cur = expected.get(key)
            if prev != cur:
                raise AssertionError(
                    "Unsafe resume detected: persisted configuration changed.\n"
                    f"State file: {state_path}\n"
                    f"Key: {key}\n"
                    f"Existing value: {prev!r}\n"
                    f"Current value: {cur!r}\n"
                    f"{guidance}"
                )

        if backfilled:
            _atomic_write_state_json(state_path, payload)
            logger.info(
                "Backfilled prepare-state keys on first resume: %s",
                sorted(backfilled),
            )
        return False

    versioned = {"version": CURRENT_PREPARE_STATE_VERSION, **dict(expected)}
    _atomic_write_state_json(state_path, versioned)
    return True


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


def write_prepare_state_for_spec(spec) -> None:
    """Persist (or assert) the canonical PREPARE_STATE for a typed PrepareSpec.

    All five family runners share an identical state-file contract:
    state file at ``<shar_dir>/PREPARE_STATE_FILE``, expected payload =
    ``spec.fingerprint_payload()``, every key invariant. Hoisting here
    keeps the contract single-sourced.
    """
    shar_dir = Path(spec.output.shar_dir)
    state_path = shar_dir / PREPARE_STATE_FILE
    expected = spec.fingerprint_payload()
    wrote = validate_or_write_prepare_state(
        state_path,
        expected=expected,
        invariant_keys=tuple(expected.keys()),
        guidance=(
            "The run's configuration has drifted from a previous run that "
            f"wrote to {shar_dir}. Re-issue the original config to resume, "
            f"or remove {shar_dir} and restart from scratch."
        ),
    )
    if wrote:
        logger.info(f"Wrote prepare state: {state_path}")


def build_shar_index_from_parts(
    *,
    shar_root: Path,
    part_dirs: Iterable[Path],
    index_filename: str,
    success_marker_name: str = SUCCESS_MARKER_FILE,
) -> tuple[Path, int]:
    """Build a merged ``shar_index.json`` from expected partition directories."""
    fields = defaultdict(list)
    shar_root = shar_root.resolve()

    for part_dir in part_dirs:
        if not part_dir.is_dir():
            raise FileNotFoundError(f"Missing partition directory: {part_dir}")

        success_marker = part_dir / success_marker_name
        if not success_marker.is_file():
            raise RuntimeError(
                f"Missing completion marker in {part_dir}. "
                "Partial partition detected; resume is unsafe."
            )

        for p in sorted(part_dir.iterdir()):
            if not p.is_file() or p.name == success_marker_name:
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
    index_path.write_text(json.dumps(payload, indent=2))
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
    index_filename: str = "shar_index.json",
    worker_dir_fmt: str = "worker_{:02d}",
) -> None:
    """Build a merged ``shar_index.json`` from all ``worker_*`` directories."""
    worker_dirs = [shar_root / worker_dir_fmt.format(wid) for wid in range(num_workers)]
    index_path, cuts_count = build_shar_index_from_parts(
        shar_root=shar_root,
        part_dirs=worker_dirs,
        index_filename=index_filename,
        success_marker_name=SUCCESS_MARKER_FILE,
    )
    logger.info(f"Wrote merged index: {index_path} ({cuts_count} cut shards)")


def load_worker_assignment(
    shar_dir: Path,
    *,
    items_key: str = "resolved_items",
) -> dict | None:
    """Load a persisted worker assignment from ``_worker_assignment.json``."""
    path = shar_dir / WORKER_ASSIGNMENT_FILE
    if not path.is_file():
        return None

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid assignment file format: {path}")

    try:
        num_workers = int(payload["num_workers"])
        resolved = payload[items_key]
    except KeyError as e:
        raise RuntimeError(
            f"Invalid assignment file (missing key {e.args[0]}): {path}"
        ) from e

    if num_workers < 1:
        raise RuntimeError(f"Invalid num_workers in assignment file: {path}")
    if not isinstance(resolved, list):
        raise RuntimeError(f"Invalid {items_key} in assignment file: {path}")

    return {
        "path": path,
        "num_workers": num_workers,
        items_key: [str(p) for p in resolved],
    }


def write_worker_assignment(
    shar_dir: Path,
    num_workers: int,
    resolved_items: Sequence,
    *,
    items_key: str = "resolved_items",
) -> Path:
    """Persist worker assignment for resume safety."""
    path = shar_dir / WORKER_ASSIGNMENT_FILE
    payload = {
        "version": 1,
        "num_workers": int(num_workers),
        items_key: list(resolved_items),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def check_worker_reuse(worker_id: int, shar_dir: str | Path) -> dict | None:
    """Check if a worker partition is already complete; return reuse dict or None."""
    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    worker_stats_path = worker_dir / WORKER_STATS_FILE
    if setup_partition_dir(
        worker_dir,
        success_marker_name=SUCCESS_MARKER_FILE,
        reuse_log=f"Worker {worker_id}: reusing completed Shar in {worker_dir}",
        reset_log=f"Worker {worker_id}: removing partial output in {worker_dir}",
        logger=logger,
    ):
        reused_worker_stats: dict = {}
        if worker_stats_path.is_file():
            try:
                reused_worker_stats = json.loads(worker_stats_path.read_text())
            except Exception:
                reused_worker_stats = {}
        return {
            "worker_id": worker_id,
            "written": -1,
            "skipped": 0,
            "errors": 0,
            "elapsed": 0,
            "total_duration_sec": reused_worker_stats.get("total_duration_sec", 0.0),
            "reused": True,
            "worker_stats": reused_worker_stats,
        }
    return None


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
        "reused": False,
        "runtime_counts": dict(runtime_counts),
    }
    if extra_stats:
        worker_stats.update(extra_stats)

    worker_stats_path = worker_dir / WORKER_STATS_FILE
    worker_stats_path.write_text(json.dumps(worker_stats, indent=2) + "\n")

    mark_partition_success(worker_dir, success_marker_name=SUCCESS_MARKER_FILE)

    result: dict = {
        "worker_id": worker_id,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "elapsed": elapsed,
        "total_duration_sec": total_duration_sec,
        "reused": False,
        "worker_stats": worker_stats,
    }
    if extra_stats:
        result.update(extra_stats)
    return result


def ensure_worker_assignment(
    shar_dir: Path,
    resolved_items: Sequence,
    num_workers: int | None,
    items_key: str,
    item_noun: str,
) -> int:
    """Load or create a worker assignment; return the final ``num_workers``."""
    assignment = load_worker_assignment(shar_dir, items_key=items_key)
    if assignment is not None:
        if assignment[items_key] != list(resolved_items):
            raise RuntimeError(
                f"Existing worker assignment {item_noun} list does not match current resolved items. "
                f"Delete {shar_dir / WORKER_ASSIGNMENT_FILE} and worker_* directories to start fresh."
            )
        if num_workers is not None and int(num_workers) != assignment["num_workers"]:
            raise RuntimeError(
                f"Existing worker assignment requires num_workers={assignment['num_workers']}, "
                f"but got {num_workers}. Keep num_workers stable when resuming."
            )
        final = assignment["num_workers"]
        logger.info(f"Reusing worker assignment from {assignment['path']} (num_workers={final})")
        return final

    final = resolve_num_workers(num_workers, num_inputs=len(resolved_items))
    assignment_path = write_worker_assignment(
        shar_dir, final, resolved_items, items_key=items_key,
    )
    logger.info(f"Wrote worker assignment to {assignment_path}")
    return final


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
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.map(worker_fn, worker_args)

    elapsed = _time.time() - t0
    total_written = sum(r["written"] for r in results if r["written"] >= 0)
    total_skipped = sum(r["skipped"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_reused = sum(1 for r in results if r.get("reused"))
    total_duration_sec = sum(r.get("total_duration_sec", 0.0) for r in results)
    total_reason_counts: Counter = Counter()
    total_runtime_counts: Counter = Counter()
    for r in results:
        total_reason_counts.update(r.get("reason_counts", {}))
        total_runtime_counts.update(
            (r.get("worker_stats") or {}).get("runtime_counts", {})
        )

    logger.info(
        f"All workers done in {elapsed:.1f}s — "
        f"{total_written} samples, {total_skipped} skipped, {total_errors} errors, "
        f"{total_duration_sec / 3600.0:.1f} hours written"
    )
    if total_reason_counts:
        logger.info(f"VAD reasons (global): {dict(total_reason_counts)}")
    if total_runtime_counts:
        logger.info(f"Runtime counters (global): {dict(total_runtime_counts)}")

    summary = {
        "version": 1,
        "num_workers": num_workers,
        "workers_reused": total_reused,
        "elapsed_sec": elapsed,
        "total_written": total_written,
        "total_skipped": total_skipped,
        "total_errors": total_errors,
        "total_duration_sec": total_duration_sec,
        "runtime_counts": dict(total_runtime_counts),
        "reason_counts": dict(total_reason_counts),
        "results": results,
    }
    summary_path = Path(shar_dir) / PREPARE_SUMMARY_FILE
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(f"Wrote prepare summary: {summary_path}")

    build_shar_index(Path(shar_dir), num_workers=num_workers)

    from audio_tokenization.prepare.validate_shar import (
        validate_shar_directory,
    )
    counts = validate_shar_directory(Path(shar_dir))
    logger.info(
        "Validated SHAR: %d cuts across %d shards", sum(counts.values()), len(counts)
    )
    for shard_name, n in counts.items():
        logger.debug("  %s: %d cuts", shard_name, n)

    mark_partition_success(Path(shar_dir), success_marker_name=SUCCESS_MARKER_FILE)
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

    summaries = []
    for nd in node_dirs:
        sp = nd / PREPARE_SUMMARY_FILE
        if not sp.is_file():
            logger.warning(f"Missing {sp}, skipping")
            continue
        summaries.append(json.loads(sp.read_text()))

    if not summaries:
        raise FileNotFoundError(f"No {PREPARE_SUMMARY_FILE} found in any node dir")

    total_written = 0
    total_skipped = 0
    total_errors = 0
    total_duration_sec = 0.0
    total_elapsed_sec = 0.0
    agg_reason: Counter = Counter()
    agg_runtime: Counter = Counter()

    for s in summaries:
        total_written += s.get("total_written", 0)
        total_skipped += s.get("total_skipped", 0)
        total_errors += s.get("total_errors", 0)
        total_duration_sec += s.get("total_duration_sec", 0.0)
        total_elapsed_sec = max(total_elapsed_sec, s.get("elapsed_sec", 0.0))
        agg_reason.update(s.get("reason_counts", {}))
        agg_runtime.update(s.get("runtime_counts", {}))

    total_hours = total_duration_sec / 3600.0

    print()
    print(f"=== Aggregate stats from {len(summaries)} node(s) under {shar_root} ===")
    print(f"  Samples written:  {total_written:>12d}")
    print(f"  Samples skipped:  {total_skipped:>12d}")
    print(f"  Errors:           {total_errors:>12d}")
    print(f"  Total hours:      {total_hours:>12.1f}")
    print(f"  Max wall-time:    {total_elapsed_sec:>12.1f}s")
    if agg_reason:
        print(f"  VAD reasons:      {dict(agg_reason)}")
    if agg_runtime:
        print(f"  Runtime counters: {dict(agg_runtime)}")
    print()
