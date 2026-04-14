#!/usr/bin/env python3
"""Shared helpers for dataset preparation scripts (HF/WDS -> Shar)."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from audio_tokenization.utils.clip_id_parsers import available_clip_id_parsers


SUCCESS_MARKER_FILE = "_SUCCESS"

# Minimum RMS threshold (dB) for keeping audio during SHAR conversion.
# -50dB keeps quiet but audible speech; only drops near-silence.
MIN_RMS_DB = -50.0
PREPARE_STATE_FILE = "_PREPARE_STATE.json"
MetadataEntry = tuple[Optional[str], dict]
_MISSING = object()


def add_input_clip_id_parser_arg(parser) -> None:
    """Add a CLI option for parsing legacy input clip IDs at prepare time."""
    parser.add_argument(
        "--input-clip-id-parser",
        type=str,
        choices=available_clip_id_parsers(),
        default=None,
        help=(
            "Parse incoming sample IDs into (source_id, clip_num) before writing "
            "universal IDs. If unset, direct inputs become {raw_id}@000000 and "
            "chunked inputs use dense chunk numbering."
        ),
    )


def add_external_metadata_args(parser, *, include_custom_fields: bool = True) -> None:
    """Add shared CLI options for transcript/custom metadata overrides."""
    parser.add_argument(
        "--external-metadata",
        type=str,
        default=None,
        help=(
            "Path to external metadata file (.tsv, .jsonl/.jsonl.gz, or .csv). "
            "When set, entries are looked up by sample ID and can override text "
            "and provide additional custom fields."
        ),
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default="id",
        help="Key/column name for sample ID in external metadata (default: 'id')",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Key/column name for transcript text in external metadata (default: 'text')",
    )
    if include_custom_fields:
        parser.add_argument(
            "--custom-fields",
            type=str,
            nargs="*",
            default=None,
            help=(
                "Keys/columns to copy from external metadata into cut.custom "
                "(e.g. --custom-fields language speaker)"
            ),
        )


def resolve_input_source_and_clip_num(
    raw_id: object,
    *,
    chunk_idx: int = 0,
    input_clip_id_parser: Callable[[str], tuple[str, int]] | None = None,
) -> tuple[str, int]:
    """Resolve the output ``(source_id, clip_num)`` for a prepared cut."""
    source_id = str(raw_id)
    if input_clip_id_parser is None:
        return source_id, int(chunk_idx)
    if chunk_idx != 0:
        raise ValueError(
            "input_clip_id_parser cannot be combined with chunked outputs; "
            "the input ID already encodes clip numbering."
        )
    return input_clip_id_parser(source_id)


def _open_compressed(path: Path, mode: str = "rb"):
    """Open a file, transparently decompressing .gz or .zst."""
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, mode)
    if path.suffix == ".zst":
        import zstandard
        # zstandard.open("rb") doesn't support line iteration;
        # wrap in BufferedReader which does.
        if "b" in mode:
            raw = zstandard.open(path, mode)
            return io.BufferedReader(raw)
        return zstandard.open(path, mode)
    return open(path, mode)


def _strip_compression_suffix(path: Path) -> str:
    """Return the format suffix, ignoring .gz/.zst compression."""
    if path.suffix in (".gz", ".zst"):
        return Path(path.stem).suffix
    return path.suffix


def load_external_metadata(
    path: str,
    custom_fields: Optional[tuple[str, ...]] = None,
    *,
    id_field: str = "id",
    text_field: str = "text",
) -> dict[str, MetadataEntry]:
    """Load transcript metadata from an external file.

    Supported formats: ``.tsv``, ``.csv``, ``.jsonl``.
    Compression: ``.gz`` and ``.zst`` are handled transparently
    (e.g. ``metadata.jsonl.zst``).
    """
    p = Path(path)
    fmt = _strip_compression_suffix(p)
    result: dict[str, MetadataEntry] = {}

    if fmt == ".tsv":
        with _open_compressed(p, "rt") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    result[parts[0]] = (parts[1], {})

    elif fmt == ".jsonl":
        import orjson

        skipped = 0
        with _open_compressed(p, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except (orjson.JSONDecodeError, ValueError):
                    skipped += 1
                    continue
                custom = {k: obj[k] for k in (custom_fields or ()) if k in obj}
                result[str(obj[id_field])] = (obj.get(text_field), custom)
        if skipped:
            logging.getLogger(__name__).warning(
                "Skipped %d malformed lines in %s", skipped, p
            )

    elif fmt == ".csv":
        import csv

        with _open_compressed(p, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                custom = {k: row[k] for k in (custom_fields or ()) if k in row}
                result[str(row[id_field])] = (row.get(text_field), custom)

    else:
        raise ValueError(
            f"Unsupported external metadata format: {p.name} "
            "(expected .tsv, .csv, or .jsonl, optionally with .gz/.zst compression)"
        )

    logging.getLogger(__name__).info(
        "Loaded %d entries from external metadata: %s", len(result), p
    )
    return result


def lookup_external_metadata(
    metadata: Mapping[str, MetadataEntry],
    sample_id: str,
    *,
    stats: Counter | None = None,
    allow_extensions: Iterable[str] = (),
) -> MetadataEntry:
    """Resolve a sample from an external metadata map."""
    basename = sample_id.rsplit("/", 1)[-1] if "/" in sample_id else sample_id
    candidates = [sample_id, basename] if basename != sample_id else [sample_id]

    for key in candidates:
        if key in metadata:
            return metadata[key]

    for key in candidates:
        for ext in allow_extensions:
            candidate = key + ext
            if candidate in metadata:
                return metadata[candidate]

    if stats is not None:
        stats["external_meta_miss"] += 1
    return None, {}


def resolve_sample_text_and_custom(
    sample_id: str,
    *,
    default_text: str | None = None,
    default_custom: Mapping[str, object] | None = None,
    external_metadata: Mapping[str, MetadataEntry] | None = None,
    stats: Counter | None = None,
    allow_extensions: Iterable[str] = (),
) -> MetadataEntry:
    """Resolve text/custom for a sample, allowing external metadata overrides."""
    text = default_text
    custom = dict(default_custom or {})
    if not external_metadata:
        return text, custom

    ext_text, ext_custom = lookup_external_metadata(
        external_metadata,
        sample_id,
        stats=stats,
        allow_extensions=allow_extensions,
    )
    if ext_text is not None:
        text = ext_text
    if ext_custom:
        custom.update(ext_custom)
    return text, custom


def set_universal_cut_id(
    cut,
    source_id: str,
    clip_num: int,
    *,
    clip_start: float | None = None,
):
    """Rewrite ``cut.id`` and store canonical interleaving metadata.

    The conversion side is the source of truth for interleaving semantics:
    ``cut.custom`` carries ``source_id``/``clip_num``/``clip_start`` so
    downstream consumers do not need dataset-specific parsers.
    """
    if clip_num < 0:
        raise ValueError(f"clip_num must be >= 0, got {clip_num}")
    legacy_cut_id = getattr(cut, "id", None)
    cut.id = f"{source_id}@{clip_num:06d}"
    for supervision in getattr(cut, "supervisions", ()) or ():
        supervision.id = cut.id
    cut.custom = dict(cut.custom or {})
    cut.custom["source_id"] = source_id
    cut.custom["clip_num"] = int(clip_num)
    if clip_start is not None:
        cut.custom["clip_start"] = float(clip_start)
    if legacy_cut_id is not None and legacy_cut_id != cut.id:
        cut.custom["legacy_cut_id"] = legacy_cut_id
    return cut


def assign_universal_ids(
    cuts: list,
    store_clip_start: bool = True,
    max_gap_sec: float | None = None,
) -> list:
    """Rewrite cut IDs to the universal format: ``{recording_id}@{clip_num:06d}``.

    Groups cuts by ``recording_id``, sorts within each group by ``cut.start``
    (supervision start time), and assigns dense 0-based clip numbers.

    If *max_gap_sec* is set, a temporal gap larger than this between
    consecutive clips breaks the run: the ``source_id`` gets a suffix
    ``_R{run_idx}`` and ``clip_num`` resets to 0. This prevents the
    interleaving pipeline from combining clips that are far apart in time.

    If *store_clip_start* is True, stores ``cut.custom["clip_start"]``
    with the segment's start time in the source recording.
    """
    groups = defaultdict(list)
    for cut in cuts:
        groups[cut.recording_id].append(cut)

    result = []
    for rec_id, group in groups.items():
        group.sort(key=lambda c: (c.start, c.id))

        run_idx = 0
        clip_num = 0
        prev_end = None

        for cut in group:
            # Break the run if gap exceeds threshold
            if max_gap_sec is not None and prev_end is not None:
                gap = cut.start - prev_end
                if gap > max_gap_sec:
                    run_idx += 1
                    clip_num = 0

            source_id = f"{rec_id}_R{run_idx}" if run_idx > 0 else rec_id
            set_universal_cut_id(
                cut,
                source_id,
                clip_num,
                clip_start=cut.start if store_clip_start else None,
            )

            prev_end = cut.start + cut.duration
            clip_num += 1
            result.append(cut)
    return result


def _ffmpeg_cli_to_wav(audio_bytes: bytes, recording_id: str) -> bytes:
    """Transcode audio bytes to WAV via ffmpeg CLI subprocess.

    The CLI is more tolerant of malformed containers than the in-process
    library API — it skips corrupted frames instead of failing.
    """
    import subprocess

    result = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", "pipe:0",
         "-f", "wav", "-acodec", "pcm_s16le", "pipe:1"],
        input=audio_bytes, capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg CLI decode failed for {recording_id}: "
            f"{result.stderr.decode(errors='replace')[:200]}"
        )
    if not result.stdout:
        raise RuntimeError(f"ffmpeg CLI produced empty output for {recording_id}")
    return result.stdout


def build_recording_from_audio_bytes(
    audio_bytes: bytes,
    recording_id: str,
    *,
    runtime_counts: Counter | None = None,
):
    """Create a Lhotse Recording from encoded audio bytes.

    Tries the fast in-process path (``Recording.from_bytes``) first.
    Falls back to ffmpeg CLI subprocess for malformed files.
    """
    from lhotse import Recording

    recording_id = str(recording_id)
    if runtime_counts is not None:
        runtime_counts["recording_from_bytes"] += 1

    try:
        return Recording.from_bytes(data=audio_bytes, recording_id=recording_id)
    except (ValueError, RuntimeError, OSError):
        pass

    # Fallback: transcode to WAV via ffmpeg CLI, then parse the clean WAV
    if runtime_counts is not None:
        runtime_counts["ffmpeg_cli_fallback"] += 1
    wav_bytes = _ffmpeg_cli_to_wav(audio_bytes, recording_id)
    return Recording.from_bytes(data=wav_bytes, recording_id=recording_id)


def load_text_tokenizer(tokenizer_path: str | Path):
    """Load a Rust fast tokenizer from a tokenizer.json file.
    Returns None if tokenizer_path is None.
    """
    if tokenizer_path is None:
        return None
    from tokenizers import Tokenizer
    path = Path(tokenizer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Text tokenizer not found: {path}")
    tok = Tokenizer.from_file(str(path))
    logging.getLogger(__name__).info(f"Text pre-tokenization enabled: {path}")
    return tok


def make_text_tokenize_fn(tokenizer, extra_custom_columns=None):
    """Return a lhotse cut map function that tokenizes supervision text.

    Stores result as cut.custom["text_tokens"] (list[int]).
    If *extra_custom_columns* is provided, also tokenizes those
    cut.custom fields and stores as cut.custom["{col}_tokens"].
    """
    _logger = logging.getLogger(__name__)
    _extra = tuple(extra_custom_columns or ())

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
        for col in _extra:
            val = cut.custom.get(col)
            if val and isinstance(val, str):
                cut.custom[f"{col}_tokens"] = tokenizer.encode(
                    val, add_special_tokens=False
                ).ids
        return cut
    return _tokenize_text


def normalize_batch_peak(audios, target_db: float = -3.0):
    """Peak-normalize each sample in a padded batch (vectorized, no Python loop).

    Matches WavTokenizer's training preprocessing: SOX ``norm`` to a target
    peak dB below full scale.  Guarantees no clipping (peaks ≤ target_db dBFS).

    Assumes zero-padding (Lhotse's ``UnsupervisedWaveformDataset(collate=True)``
    and ``AudioSamples()`` both zero-pad by default).  Zero padding does not
    affect ``abs().max()`` and stays zero after scaling.

    Args:
        audios: (B, T) float tensor (CPU or GPU), zero-padded.
        target_db: target peak in dBFS (default -3.0, matching WavTokenizer val).

    Returns:
        New tensor with each sample scaled so its peak matches *target_db* dBFS.
        Near-silence (peak < -100 dB) is left unchanged.
    """
    target_peak = 10 ** (target_db / 20.0)
    peaks = audios.abs().max(dim=1).values.clamp(min=1e-10)
    scale = target_peak / peaks
    return audios * scale.unsqueeze(1)


def rms_db(cut) -> float:
    """Compute RMS level in dB for a cut's audio.

    Returns a finite float (closer to 0 = louder).  Silent audio returns
    approximately -200 dB.
    """
    import numpy as np

    audio = cut.load_audio()          # (channels, samples)
    if audio.size == 0:
        return -200.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20.0 * np.log10(rms + 1e-10)


def should_skip_quiet(rms_val: float) -> bool:
    """Return True if the sample should be skipped (silent/empty audio)."""
    return math.isnan(rms_val) or rms_val < MIN_RMS_DB


def make_rms_filter_fn():
    """Return a (map_fn, filter_fn) pair for computing rms_db and filtering quiet cuts.

    Usage with lhotse CutSet:
        compute_rms, keep_loud = make_rms_filter_fn()
        cuts = cuts.map(compute_rms).filter(keep_loud)

    Usage in conversion loops:
        compute_rms, keep_loud = make_rms_filter_fn()
        cut = compute_rms(cut)
        if not keep_loud(cut):
            skipped += 1; continue
    """
    def _compute_rms(cut):
        cut.custom = cut.custom or {}
        cut.custom["rms_db"] = rms_db(cut)
        return cut

    def _keep_loud(cut) -> bool:
        val = (cut.custom or {}).get("rms_db", 0.0)
        return not should_skip_quiet(val)

    return _compute_rms, _keep_loud


def apply_audio_pipeline(
    cut,
    *,
    target_sr: int | None = None,
    mono_downmix: bool = True,
    tokenize_fn=None,
    runtime_counts: Counter,
):
    """Resample → mono → tokenize text → RMS filter in one call.

    Returns ``(cut, True)`` if the cut should be skipped (quiet/empty),
    or ``(cut, False)`` if it passed all checks and is ready to write.
    """
    if target_sr and cut.sampling_rate != target_sr:
        cut = cut.resample(target_sr)
        runtime_counts["resampled"] += 1

    cut = to_mono(cut, mono_downmix=mono_downmix, stats=runtime_counts)

    if tokenize_fn is not None:
        cut = tokenize_fn(cut)

    cut.custom = cut.custom or {}
    rms_val = rms_db(cut)
    if should_skip_quiet(rms_val):
        runtime_counts["skipped_quiet_audio"] += 1
        return cut, True
    cut.custom["rms_db"] = rms_val

    return cut, False


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
    """Per-process initialisation for pool workers."""
    from lhotse.audio.resampling_backend import (
        available_resampling_backends,
        set_current_resampling_backend,
    )

    backend = resampling_backend or os.environ.get(
        "LHOTSE_RESAMPLING_BACKEND", "soxr"
    )
    if backend == "torchaudio":  # Lhotse registers torchaudio as "default"
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


# ---------------------------------------------------------------------------
# CLI argument group builders
# ---------------------------------------------------------------------------


def add_shar_output_args(parser, *, shard_size_default=2000, shar_dir_required=True):
    """Add --shar-dir, --shard-size, --shar-format."""
    parser.add_argument("--shar-dir", type=Path, required=shar_dir_required,
                        default=None,
                        help="Output directory for Shar format")
    parser.add_argument("--shard-size", type=int, default=shard_size_default,
                        help=f"Samples per Shar shard (default: {shard_size_default})")
    parser.add_argument("--shar-format", type=str, default="flac",
                        choices=["flac", "wav", "mp3", "opus"],
                        help="Audio format in Shar (default: flac)")


def add_audio_processing_args(parser, *, target_sr_default=24000,
                              include_min_sr=False, include_mono_downmix=False):
    """Add --target-sr, --resampling-backend, and optionally --min-sr, --no-mono-downmix."""
    parser.add_argument("--target-sr", type=int, default=target_sr_default,
                        help=f"Target sample rate (default: {target_sr_default})")
    parser.add_argument("--resampling-backend", type=str, default="soxr",
                        choices=["torchaudio", "soxr", "ffmpeg"],
                        help="Resampling backend (default: soxr)")
    if include_min_sr:
        parser.add_argument("--min-sr", type=int, default=16000,
                            help="Drop audio below this sample rate (default: 16000)")
    if include_mono_downmix:
        parser.add_argument("--no-mono-downmix", action="store_true",
                            help="Select channel 0 instead of averaging stereo channels")


def add_language_arg(parser):
    """Add --language for setting supervision.language on all cuts."""
    parser.add_argument("--language", type=str, default=None,
                        help="Language tag to set on all supervisions (e.g. fi, en, zh). "
                             "Overridden by --language-column if both are set.")


def add_text_tokenizer_args(parser, *, include_custom_columns=False):
    """Add --text-tokenizer and optionally --text-tokenize-custom-columns."""
    parser.add_argument("--text-tokenizer", type=str, default=None,
                        help="Path to tokenizer.json for pre-tokenizing supervision text")
    if include_custom_columns:
        parser.add_argument("--text-tokenize-custom-columns", type=str, nargs="*",
                            default=None,
                            help="Custom columns to also pre-tokenize. "
                                 "Stored as {col}_tokens in cut.custom.")


# ---------------------------------------------------------------------------
# Unified columnar metadata interface (shared by parquet + HF arrow)
# ---------------------------------------------------------------------------


def add_columnar_metadata_args(
    parser,
    *,
    id_column_default=None,
    text_column_default="text",
    duration_column_default=None,
):
    """Add shared CLI args for extracting metadata from columnar sources.

    Used by both ``prepare_parquet_to_shar`` and ``prepare_hf_to_shar``
    to provide a consistent interface for text, language, and custom fields.
    """
    parser.add_argument("--id-column", type=str, nargs="*", default=id_column_default,
                        help="Column name(s) for row ID. Multiple columns are joined with '_'. "
                             "Dotted paths like 'audio.path' access nested struct fields. "
                             "Omit to auto-generate IDs from filename + row index.")
    parser.add_argument("--audio-column", type=str, default="audio",
                        help="Column name for audio struct (default: 'audio')")
    parser.add_argument("--text-column", type=str, default=text_column_default,
                        help=f"Column name for transcription text (default: {text_column_default!r})")
    parser.add_argument("--duration-column", type=str, default=duration_column_default,
                        help="Column name for duration in seconds (for filtering)")
    parser.add_argument("--language-column", type=str, default=None,
                        help="Column name for per-row language code. "
                             "Takes precedence over --language when set.")
    parser.add_argument("--custom-columns", type=str, nargs="*", default=None,
                        help="Additional columns to store in cut.custom dict")
    add_language_arg(parser)
    add_text_tokenizer_args(parser, include_custom_columns=True)


def _require_id_field(row, col):
    value = _get_field(row, col)
    if value is _MISSING or value is None:
        raise ValueError(f"id_column {col!r} is missing or null in row")
    return value


def extract_row_metadata(
    row,
    *,
    id_column=None,
    text_column=None,
    language_column=None,
    language=None,
    custom_columns=None,
    fallback_id=None,
):
    """Extract metadata from a columnar row (parquet or arrow).

    Returns ``(row_id, text, language, custom)`` where *custom* is a dict
    of extra fields to store in ``cut.custom``. A required ``id_column``
    that resolves to missing OR null raises ``ValueError``.
    """
    if not id_column:
        row_id = fallback_id
    elif isinstance(id_column, (list, tuple)):
        row_id = "_".join(str(_require_id_field(row, c)) for c in id_column)
    else:
        row_id = str(_require_id_field(row, id_column))

    text = None
    if text_column:
        value = _get_field(row, text_column)
        if value is not _MISSING and value is not None:
            text = value

    lang = None
    if language_column:
        value = _get_field(row, language_column)
        if value is not _MISSING and value is not None:
            lang = value
    if lang is None:
        lang = language

    custom = {}
    if custom_columns:
        for col in custom_columns:
            val = _get_field(row, col)
            if val is not _MISSING and val is not None:
                custom[col] = val

    return row_id, text, lang, custom


def _get_field(row, path: str):
    """Resolve a flat or dotted path against a row dict-like object.

    Returns ``_MISSING`` when any intermediate key is absent. A key present
    with value ``None`` returns ``None`` — the caller distinguishes the two
    to tell "null in source" from "field not projected".
    """
    if "." not in path:
        return row.get(path, _MISSING) if isinstance(row, dict) else _MISSING
    cur = row
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return _MISSING
        cur = cur[key]
    return cur


def _projected_columns(*cols) -> list[str]:
    """Plan a minimal projection for parquet reading.

    Dotted leaves are preserved unless an ancestor path is already requested.
    """
    requested: list[str] = []
    for spec in cols:
        if spec is None:
            continue
        items = spec if isinstance(spec, (list, tuple)) else [spec]
        for col in items:
            if col and col not in requested:
                requested.append(col)

    out = []
    requested_set = set(requested)
    for col in requested:
        parts = col.split(".")
        ancestors = {".".join(parts[:i]) for i in range(1, len(parts))}
        if not ancestors.intersection(requested_set):
            out.append(col)
    return out


def add_parallelism_args(parser, *, num_workers_default=20,
                         include_mp_start_method=False):
    """Add --num-workers and optionally --mp-start-method."""
    parser.add_argument("--num-workers", type=int, default=num_workers_default,
                        help=f"Number of parallel workers (default: {num_workers_default})")
    if include_mp_start_method:
        parser.add_argument("--mp-start-method", type=str, default="forkserver",
                            choices=["fork", "forkserver", "spawn"],
                            help="Multiprocessing start method (default: forkserver)")
