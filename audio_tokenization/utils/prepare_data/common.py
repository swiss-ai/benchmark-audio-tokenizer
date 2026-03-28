#!/usr/bin/env python3
"""Shared helpers for dataset preparation scripts (HF/WDS -> Shar)."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SUCCESS_MARKER_FILE = "_SUCCESS"
PREPARE_STATE_FILE = "_PREPARE_STATE.json"


def load_text_tokenizer(tokenizer_path: str | Path):
    """Load a Rust fast tokenizer from a tokenizer.json file.
    Returns None if tokenizer_path is None.
    """
    if tokenizer_path is None:
        return None
    import logging
    from tokenizers import Tokenizer
    path = Path(tokenizer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Text tokenizer not found: {path}")
    tok = Tokenizer.from_file(str(path))
    logging.getLogger(__name__).info(f"Text pre-tokenization enabled: {path}")
    return tok


def make_text_tokenize_fn(tokenizer):
    """Return a lhotse cut map function that tokenizes supervision text.
    Stores result as cut.custom["text_tokens"] (list[int]).
    """
    import logging
    _logger = logging.getLogger(__name__)

    def _tokenize_text(cut):
        texts = [s.text for s in (cut.supervisions or []) if s.text]
        if not texts:
            return cut
        if len(texts) > 1:
            _logger.debug(
                "Cut %s: merging %d supervision texts into one", cut.id, len(texts)
            )
        ids = tokenizer.encode(" ".join(texts), add_special_tokens=False).ids
        cut.custom = cut.custom or {}
        cut.custom["text_tokens"] = ids
        return cut
    return _tokenize_text


def normalize_batch_peak(audios, audio_lens, target_db: float = -3.0):
    """Peak-normalize each sample in a padded batch.

    Matches WavTokenizer's training preprocessing: SOX ``norm`` to a target
    peak dB below full scale.  Guarantees no clipping (peaks ≤ target_db dBFS).

    Args:
        audios: (B, T) float tensor on CPU.
        audio_lens: list of int, original sample counts per item.
        target_db: target peak in dBFS (default -3.0, matching WavTokenizer val).

    Returns:
        New tensor with each sample scaled so its peak matches *target_db* dBFS.
        Near-silence (peak < -100 dB) is left unchanged.
    """
    target_peak = 10 ** (target_db / 20.0)
    result = audios.clone()
    for i, length in enumerate(audio_lens):
        segment = result[i, :length]
        peak = segment.abs().max().item()
        if peak > 1e-10:  # skip near-silence
            result[i, :length] = segment * (target_peak / peak)
    return result


def rms_db(cut) -> float:
    """Compute RMS level in dB for a cut's audio.

    Returns a finite float (closer to 0 = louder).  Silent audio returns
    approximately -200 dB.
    """
    import numpy as np

    audio = cut.load_audio()          # (channels, samples)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20.0 * np.log10(rms + 1e-10)



def to_mono(cut, mono_downmix=True, stats=None):
    """Convert a multi-channel cut to mono.

    If ``mono_downmix`` is True, tries averaging channels first; falls back to
    channel 0 on decode errors.  If False, always takes channel 0.

    ``stats`` is an optional ``Counter`` for tracking fallback events.
    """
    if cut.num_channels <= 1:
        return cut
    if mono_downmix:
        try:
            result = cut.to_mono(mono_downmix=True)
            # Force-load to catch AudioLoadingError from broken headers now,
            # rather than letting it bubble up later in SharWriter.
            result.load_audio()
            return result
        except Exception:
            if stats is not None:
                stats["downmix_fallback_ch0"] += 1
            # Broken header — fall back to channel 0.
    result = cut.to_mono(mono_downmix=False)
    if isinstance(result, list):
        return result[0]  # Take first channel (channel 0)
    return result


def setup_partition_dir(
    part_dir: Path,
    *,
    success_marker_name: str = SUCCESS_MARKER_FILE,
    reuse_log: str | None = None,
    reset_log: str | None = None,
    logger=None,
) -> bool:
    """Prepare a partition directory for resume-safe writing.

    Returns:
        True if the partition is already complete and should be reused.
        False if caller should (re)process and write this partition.
    """
    success_marker = part_dir / success_marker_name
    if success_marker.is_file():
        if logger and reuse_log:
            logger.info(reuse_log)
        return True

    # If marker is missing, any leftover files are considered partial output.
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


def validate_or_write_prepare_state(
    state_path: Path,
    *,
    expected: Mapping[str, object],
    invariant_keys: Sequence[str],
    guidance: str,
) -> bool:
    """Persist first-run state or assert resume invariants on later runs."""
    if state_path.is_file():
        payload = json.loads(state_path.read_text())
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid prepare state format: {state_path}")

        for key in invariant_keys:
            prev = payload.get(key)
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
        return False

    state_path.write_text(json.dumps(dict(expected), indent=2) + "\n")
    return True


def normalize_optional_path(path: str | Path | None) -> str | None:
    """Normalize an optional path for stable resume-state comparisons."""
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


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


# ---------------------------------------------------------------------------
# Shared helpers for parallel prepare scripts (WDS, audio-dir, etc.)
# ---------------------------------------------------------------------------

WORKER_ASSIGNMENT_FILE = "_worker_assignment.json"
WORKER_STATS_FILE = "worker_stats.json"
PREPARE_SUMMARY_FILE = "prepare_summary.json"

logger = logging.getLogger(__name__)


def audio_md5(path: str) -> str:
    """MD5 of decoded audio waveform (float32 PCM, not raw file bytes)."""
    import soundfile as sf

    data, _ = sf.read(path, dtype="float32")
    return hashlib.md5(data.tobytes()).hexdigest()


def build_audio_index(audio_root: Path, pattern: str = "**/*.ogg") -> dict[str, str]:
    """Map lowercased file stems to full paths (recursive glob).

    Keys are lowercased for case-insensitive matching with
    ``canonical_sample_key`` used by the VAD / chunking pipeline.
    """
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
    """Build a merged ``shar_index.json`` from all ``worker_*`` directories.

    The index maps field names (``cuts``, ``recording``, ...) to sorted lists
    of SHAR-root-relative file paths (portable across root moves). Consumers
    should resolve relative entries against the SHAR root before loading.
    """
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
    """Load a persisted worker assignment from ``_worker_assignment.json``.

    *items_key* is the JSON key storing the list of input items
    (e.g. ``"resolved_shards"`` for WDS, ``"resolved_jsonls"`` for audio-dir).
    """
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
    """Check if a worker partition is already complete; return reuse dict or None.

    If the worker directory has a ``_SUCCESS`` marker, loads any persisted
    ``worker_stats.json`` and returns a result dict with ``"reused": True``
    (``written=-1`` signals "reused" to the aggregation logic).
    Otherwise returns ``None`` so the caller can proceed with processing.
    """
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
    """Per-process initialisation for pool workers.

    Notes:
      - ``sox`` resampling has shown allocator crashes (``free(): invalid pointer``)
        under high ``forkserver`` multiprocessing fan-out.
      - Keep the safer ``default`` backend unless explicitly overridden.
    """
    try:
        import os
        from lhotse.audio.resampling_backend import (
            available_resampling_backends,
            set_current_resampling_backend,
        )

        backend = resampling_backend or os.environ.get(
            "LHOTSE_RESAMPLING_BACKEND", "default"
        )
        if backend not in available_resampling_backends():
            logger.warning(
                "Unknown resampling backend %r; falling back to 'default'", backend
            )
            backend = "default"
        set_current_resampling_backend(backend)
    except Exception:
        pass


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

    final = min(num_workers or len(resolved_items), len(resolved_items))
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
    """Run *worker_fn* in a multiprocessing pool, aggregate stats, write summary & index.

    Returns the list of per-worker result dicts.
    """
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
