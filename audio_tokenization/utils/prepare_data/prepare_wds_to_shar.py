#!/usr/bin/env python3
"""Convert tar-based audio archives to Lhotse Shar format.

Supports two metadata modes:

1. **Standard WDS** (default): Each tar shard contains paired audio +
   sidecar files (e.g. ``sample.wav`` + ``sample.txt``).

2. **External metadata** (``--external-metadata``): Audio-only tar/tar.gz
   archives with transcripts in a separate TSV or JSONL file. Useful for
   datasets like GigaSpeech2, Common Voice, Suno, NB-Tale, etc.

Creates Lhotse Cuts from raw audio bytes (Recording.from_bytes), resamples
to target SR, and writes to Shar. Each worker processes a subset of tar
shards and writes to its own ``worker_XX/`` sub-directory. After all workers
finish, a merged ``shar_index.json`` is written.

Usage (WDS mode):
    python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
        --wds-shards '/path/to/shards/*.tar' \
        --shar-dir /output/path/shar \
        --target-sr 24000 \
        --num-workers 288 \
        --shard-size 2000 \
        --shar-format flac \
        --min-sr 16000

Usage (external metadata mode — e.g. GigaSpeech2):
    python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
        --wds-shards '/data/th/train/*.tar.gz' \
        --external-metadata /data/th/train_refined.tsv \
        --shar-dir /output/gigaspeech2_th/shar \
        --target-sr 24000 \
        --num-workers 64 \
        --shard-size 5000 \
        --shar-format flac
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import glob
import logging
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from audio_tokenization.utils.prepare_data.common import (
    PREPARE_STATE_FILE,
    add_audio_processing_args,
    add_external_metadata_args,
    add_input_clip_id_parser_arg,
    add_parallelism_args,
    add_shar_output_args,
    add_text_tokenizer_args,
    apply_audio_pipeline,
    build_recording_from_audio_bytes,
    check_worker_reuse,
    distribute_round_robin,
    ensure_worker_assignment,
    init_worker_process,
    load_external_metadata,
    load_text_tokenizer,
    lookup_external_metadata,
    make_text_tokenize_fn,
    normalize_optional_path,
    resolve_input_source_and_clip_num,
    run_aggregate,
    run_pool_and_finalize,
    set_universal_cut_id,
    validate_or_write_prepare_state,
    write_worker_result,
)
from audio_tokenization.utils.clip_id_parsers import get_clip_id_parser
from audio_tokenization.utils.prepare_data.preprocess.chunking import (
    VADChunkingConfig,
    canonical_sample_key,
    load_vad_from_per_shard_dir,
    shard_name_from_tar_path,
    split_cut_by_vad,
    vad_per_shard_file,
)

logging.basicConfig(
    
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WDS → Lhotse cuts iterator
# ---------------------------------------------------------------------------

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".opus", ".ogg")
SIDECAR_SUFFIXES = (".txt", ".json")

MetadataEntry = tuple[Optional[str], dict[str, Any]]


@dataclass
class TarScanResult:
    """Per-tar scan state used by metadata providers."""

    audio_members: list
    metadata_state: Any = None
    scan_complete: bool = True
    scan_error: Optional[str] = None


class MetadataProvider:
    """Abstract metadata source for tar members."""

    def scan_tar(self, tf, stats: Optional[Counter] = None) -> TarScanResult:
        raise NotImplementedError

    def lookup(
        self,
        stem: str,
        scan: TarScanResult,
        stats: Optional[Counter] = None,
    ) -> MetadataEntry:
        raise NotImplementedError


@dataclass
class SidecarMetadataProvider(MetadataProvider):
    """Read transcripts and custom fields from sidecars stored inside each tar."""

    text_field: str = "text"
    custom_fields: Optional[Tuple[str, ...]] = None

    def scan_tar(self, tf, stats: Optional[Counter] = None) -> TarScanResult:
        import tarfile

        metas: dict[str, MetadataEntry] = {}
        audio_members = []
        try:
            for member in tf:
                if not member.isfile():
                    continue
                dot = member.name.rfind(".")
                ext = member.name[dot:] if dot >= 0 else ""
                if ext in SIDECAR_SUFFIXES:
                    stem = member.name[:dot]
                    try:
                        extracted = tf.extractfile(member)
                        if extracted is None:
                            if stats is not None:
                                stats["text_decode_failed"] += 1
                            continue
                        raw = extracted.read()
                    except (EOFError, tarfile.ReadError, OSError) as e:
                        return TarScanResult(
                            audio_members=[],
                            metadata_state=None,
                            scan_complete=False,
                            scan_error=str(e),
                        )
                    try:
                        text, custom = _parse_sidecar(
                            raw, ext, self.text_field, self.custom_fields,
                        )
                        prev = metas.get(stem)
                        if prev:
                            text = text or prev[0]
                            custom = {**prev[1], **custom}
                        metas[stem] = (text, custom)
                    except Exception:
                        if stats is not None:
                            stats["text_decode_failed"] += 1
                elif ext in AUDIO_SUFFIXES:
                    audio_members.append(member)
        except (EOFError, tarfile.ReadError, OSError) as e:
            return TarScanResult(
                audio_members=[],
                metadata_state=None,
                scan_complete=False,
                scan_error=str(e),
            )
        return TarScanResult(audio_members=audio_members, metadata_state=metas)

    def lookup(
        self,
        stem: str,
        scan: TarScanResult,
        stats: Optional[Counter] = None,
    ) -> MetadataEntry:
        metas = scan.metadata_state or {}
        return metas.get(stem, (None, {}))


@dataclass
class ExternalMetadataProvider(MetadataProvider):
    """Read transcripts and custom fields from an external TSV or JSONL mapping."""

    metadata: dict[str, MetadataEntry]

    def scan_tar(self, tf, stats: Optional[Counter] = None) -> TarScanResult:
        audio_members = [
            m for m in tf
            if m.isfile() and m.name[m.name.rfind("."):] in AUDIO_SUFFIXES
        ]
        return TarScanResult(audio_members=audio_members)

    def lookup(
        self,
        stem: str,
        scan: TarScanResult,
        stats: Optional[Counter] = None,
    ) -> MetadataEntry:
        return lookup_external_metadata(
            self.metadata,
            stem,
            stats=stats,
            allow_extensions=AUDIO_SUFFIXES,
        )


# Module-level global for COW sharing of a large external metadata map across
# forked workers. Sidecar mode constructs its provider inside each worker.
_METADATA_PROVIDER: MetadataProvider | None = None


def _parse_sidecar(
    raw: bytes, ext: str, text_field: str = "text",
    custom_fields: Optional[Tuple[str, ...]] = None,
) -> Tuple[Optional[str], dict]:
    """Parse a sidecar into ``(text, custom_dict)``."""
    if ext == ".json":
        import orjson
        obj = orjson.loads(raw)
        text = obj.get(text_field)
        custom = {k: obj[k] for k in (custom_fields or ()) if k in obj}
        return text, custom
    return raw.decode("utf-8").strip(), {}


def iter_tar_cuts(
    tar_paths,
    provider: MetadataProvider,
    stats: Optional[Counter] = None,
    keep_ids: Optional[set] = None,
    language: Optional[str] = None,
):
    """Iterate over tar shards and yield Lhotse cuts with supervisions.

    Args:
        tar_paths: Paths to WDS tar shards.
        provider: Metadata source (sidecar or external).
        stats: Optional counter for tracking decode/skip events.
        keep_ids: If set, only decode members whose ``canonical_sample_key(stem)``
            is in this set.  Skips expensive audio decoding for irrelevant
            recordings (e.g. per-language VAD filtering).
    """
    import tarfile
    from lhotse import SupervisionSegment

    for tar_path in tar_paths:
        try:
            tf = tarfile.open(tar_path)
        except (EOFError, tarfile.ReadError, OSError) as e:
            logger.warning(f"Skipping corrupt tar: {tar_path}: {e}")
            if stats is not None:
                stats["skipped_corrupt_tar"] += 1
            continue

        with tf:
            try:
                scan = provider.scan_tar(tf, stats=stats)
            except (EOFError, tarfile.ReadError, OSError) as e:
                logger.warning(f"Skipping corrupt tar: {tar_path}: {e}")
                if stats is not None:
                    stats["skipped_corrupt_tar"] += 1
                continue
            if not scan.scan_complete:
                logger.warning(
                    f"Skipping corrupt tar after partial scan: {tar_path}: "
                    f"{scan.scan_error or 'incomplete tar scan'}"
                )
                if stats is not None:
                    stats["skipped_corrupt_tar"] += 1
                continue

            for member in scan.audio_members:
                stem = member.name[:member.name.rfind(".")]

                # Pre-filter: skip decoding for recordings we know we won't use.
                if keep_ids is not None and canonical_sample_key(stem) not in keep_ids:
                    if stats is not None:
                        stats["skipped_no_match"] += 1
                    continue

                try:
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        if stats is not None:
                            stats["missing_payload"] += 1
                        raise ValueError("tar member has no readable payload")
                    recording = build_recording_from_audio_bytes(
                        extracted.read(),
                        stem,
                        runtime_counts=stats,
                    )
                    cut = recording.to_cut()
                except Exception as e:
                    if stats is not None:
                        stats["failed_build_cut"] += 1
                    logger.warning(f"Skipping {stem}: failed to build cut ({e})")
                    continue

                text, custom = provider.lookup(stem, scan, stats=stats)
                if text:
                    cut.supervisions = [SupervisionSegment(
                        id=cut.id,
                        recording_id=cut.recording_id,
                        start=0.0,
                        duration=cut.duration,
                        text=text,
                        language=language,
                    )]
                if custom:
                    cut.custom = custom

                if stats is not None:
                    stats["cuts_yielded"] += 1
                yield cut


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _convert_worker(args_tuple):
    """Convert a subset of WDS tar shards to Shar.

    Each worker writes to its own ``worker_XX/`` directory to avoid contention.
    Resume is considered complete only when ``worker_XX/_SUCCESS`` exists.
    Partial output (cuts manifests without marker) is deleted and recomputed.
    """
    (
        worker_id,
        tar_paths,
        shar_dir,
        target_sr,
        shard_size,
        shar_format,
        min_sr,
        text_field,
        custom_fields,
        mono_downmix,
        vad_per_shard_dir,
        vad_max_chunk_sec,
        vad_min_chunk_sec,
        vad_sample_rate,
        vad_max_merge_gap_sec,
        vad_max_duration_sec,
        text_tokenizer,
        resampling_backend,
        input_clip_id_parser_name,
        language,
    ) = args_tuple

    reused = check_worker_reuse(worker_id, shar_dir)
    if reused is not None:
        return reused
    init_worker_process(resampling_backend)

    from lhotse.shar import SharWriter

    use_vad_segmenting = bool(vad_per_shard_dir)
    reason_counts = Counter()

    # Load VAD entries from per-shard files for this worker's tar shards.
    if use_vad_segmenting:
        vad_cfg = VADChunkingConfig(
            max_chunk_sec=float(vad_max_chunk_sec),
            min_chunk_sec=float(vad_min_chunk_sec),
            sample_rate=int(vad_sample_rate),
            max_merge_gap_sec=float(vad_max_merge_gap_sec),
            max_duration_sec=float(vad_max_duration_sec) if vad_max_duration_sec is not None else None,
        )
        vad_lookup, sr_lookup, lang_lookup = load_vad_from_per_shard_dir(
            Path(vad_per_shard_dir), tar_paths, with_lang=True, logger=logger,
        )
    else:
        vad_cfg = None
        vad_lookup = {}
        sr_lookup = {}
        lang_lookup = {}

    worker_dir = Path(shar_dir) / f"worker_{worker_id:02d}"
    t0 = time.time()
    written = skipped = errors = 0
    total_duration_sec = 0.0
    runtime_counts = Counter()
    _tokenize_text = make_text_tokenize_fn(text_tokenizer) if text_tokenizer is not None else None
    input_clip_id_parser = (
        get_clip_id_parser(input_clip_id_parser_name)
        if input_clip_id_parser_name
        else None
    )
    provider = _METADATA_PROVIDER or SidecarMetadataProvider(
        text_field=text_field,
        custom_fields=custom_fields,
    )

    with SharWriter(
        output_dir=str(worker_dir),
        fields={"recording": shar_format},
        shard_size=shard_size,
    ) as writer:
        keep_ids = set(vad_lookup) if use_vad_segmenting else None
        for cut in iter_tar_cuts(tar_paths, provider=provider, stats=runtime_counts, keep_ids=keep_ids, language=language):
            try:
                if min_sr and cut.sampling_rate < min_sr:
                    skipped += 1
                    runtime_counts["skipped_min_sr"] += 1
                    continue

                if target_sr and cut.sampling_rate != target_sr:
                    cut = cut.resample(target_sr)
                    runtime_counts["resampled"] += 1

                # No intermediate WAV dump: decode -> optional resample -> split -> write.
                if use_vad_segmenting:
                    out_cuts, reason = split_cut_by_vad(
                        cut=cut,
                        sample_key=cut.recording_id,
                        vad_lookup=vad_lookup,
                        cfg=vad_cfg,
                        sr_lookup=sr_lookup,
                    )
                    reason_counts[reason] += 1
                else:
                    out_cuts = [cut]

                if not out_cuts:
                    skipped += 1
                    runtime_counts["skipped_empty_output"] += 1
                    continue

                sample_lang = lang_lookup.get(
                    canonical_sample_key(cut.recording_id)
                ) if lang_lookup else None

                for chunk_idx, out_cut in enumerate(out_cuts):
                    if sample_lang is not None:
                        out_cut.custom = out_cut.custom or {}
                        out_cut.custom["lang"] = sample_lang
                    out_cut, skip = apply_audio_pipeline(
                        out_cut,
                        target_sr=None,  # already resampled before VAD
                        mono_downmix=mono_downmix,
                        tokenize_fn=_tokenize_text,
                        runtime_counts=runtime_counts,
                    )
                    if skip:
                        skipped += 1
                        continue
                    source_id, clip_num = resolve_input_source_and_clip_num(
                        cut.recording_id,
                        chunk_idx=chunk_idx,
                        input_clip_id_parser=input_clip_id_parser,
                    )
                    set_universal_cut_id(
                        out_cut,
                        source_id,
                        clip_num,
                        clip_start=(out_cut.custom or {}).get("global_offset_sec", 0.0),
                    )
                    writer.write(out_cut)
                    written += 1
                    total_duration_sec += out_cut.duration
                    runtime_counts["cuts_written"] += 1

                if written % 1000 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"Worker {worker_id}: {written} written, {skipped} skipped, "
                        f"{errors} errors ({written / elapsed:.1f} samples/s)"
                    )

            except Exception as e:
                errors += 1
                runtime_counts["processing_errors"] += 1
                if errors <= 5:
                    logger.warning(f"Worker {worker_id} error on {cut.id}: {e}")

    if use_vad_segmenting and reason_counts:
        logger.info(f"Worker {worker_id} VAD reasons: {dict(reason_counts)}")

    return write_worker_result(
        worker_id=worker_id, worker_dir=worker_dir,
        written=written, skipped=skipped, errors=errors,
        total_duration_sec=total_duration_sec,
        runtime_counts=runtime_counts, t0=t0,
        extra_stats={
            "vad_enabled": use_vad_segmenting,
            "reason_counts": dict(reason_counts),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ITEMS_KEY = "resolved_shards"


def _validate_or_write_prepare_state(args) -> None:
    state_path = args.shar_dir / PREPARE_STATE_FILE
    expected = {
        "text_tokenizer": normalize_optional_path(args.text_tokenizer),
        "text_field": args.text_field,
        "custom_fields": sorted(args.custom_fields) if args.custom_fields else None,
        "external_metadata": normalize_optional_path(args.external_metadata),
        "id_field": args.id_field,
        "input_clip_id_parser": args.input_clip_id_parser,
        "language": args.language,
    }
    wrote = validate_or_write_prepare_state(
        state_path,
        expected=expected,
        invariant_keys=(
            "text_tokenizer",
            "text_field",
            "custom_fields",
            "external_metadata",
            "id_field",
            "input_clip_id_parser",
            "language",
        ),
        guidance=(
            "Use the same --text-tokenizer, --text-field, --custom-fields, "
            "--external-metadata, --id-field, --language, and "
            "--input-clip-id-parser to resume, or "
            f"remove {args.shar_dir} and restart from scratch."
        ),
    )
    if wrote:
        logger.info(f"Wrote prepare state: {state_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert standard WDS → Lhotse Shar (parallel)",
    )

    # Input
    parser.add_argument("--wds-shards", type=str, nargs="+", default=None,
                        help="Glob patterns or file paths for WDS tar shards")

    add_shar_output_args(parser, shard_size_default=5000, shar_dir_required=False)
    add_audio_processing_args(parser, include_min_sr=True, include_mono_downmix=True)

    # Text / metadata
    add_external_metadata_args(parser, include_custom_fields=True)
    parser.add_argument("--language", type=str, default=None,
                        help="Language tag to set on all supervisions (e.g. fi, en, zh)")
    add_input_clip_id_parser_arg(parser)
    add_text_tokenizer_args(parser)
    add_parallelism_args(parser, num_workers_default=None)
    parser.add_argument("--vad-segmentation", action="store_true",
                        help="Split long recordings into speech-aware segments during prepare")
    parser.add_argument("--vad-per-shard-dir", type=Path, default=None,
                        help="Directory of per-shard VAD JSONL files (required with --vad-segmentation)")
    parser.add_argument("--vad-max-chunk-sec", type=float, default=200.0,
                        help="Target max duration while packing VAD segments")
    parser.add_argument("--vad-min-chunk-sec", type=float, default=10.0,
                        help="Drop chunks shorter than this duration")
    parser.add_argument("--vad-sample-rate", type=int, default=16000,
                        help="Sample rate used to decode VAD timestamp units")
    parser.add_argument("--vad-max-merge-gap-sec", type=float, default=0.5,
                        help="Merge adjacent VAD spans when silence gap <= this threshold")
    parser.add_argument("--vad-max-duration-sec", type=float, default=None,
                        help="Drop atomic speech segments longer than this "
                             "(default: same as --vad-max-chunk-sec)")

    parser.add_argument("--aggregate", type=Path, default=None, metavar="SHAR_ROOT",
                        help="Aggregate stats from completed multi-node runs and exit. "
                             "Reads prepare_summary.json from all node_*/ dirs under SHAR_ROOT.")

    args = parser.parse_args(argv)

    # ---- Aggregate mode: read summaries and exit ----
    if args.aggregate is not None:
        run_aggregate(args.aggregate)
        return

    if not args.wds_shards:
        parser.error("--wds-shards is required (unless using --aggregate)")
    if args.shar_dir is None:
        parser.error("--shar-dir is required (unless using --aggregate)")

    resolved = sorted(set(p for pattern in args.wds_shards for p in glob.glob(pattern)))
    if not resolved:
        raise FileNotFoundError(f"No files match patterns: {args.wds_shards}")

    # Pre-filter shards that have no VAD file (avoids empty workers).
    if args.vad_segmentation and args.vad_per_shard_dir:
        before = len(resolved)
        resolved = [
            p for p in resolved
            if vad_per_shard_file(args.vad_per_shard_dir, shard_name_from_tar_path(p)).is_file()
        ]
        skipped_shards = before - len(resolved)
        if skipped_shards:
            logger.info(f"Skipped {skipped_shards} shards with no VAD file ({len(resolved)} remaining)")
        if not resolved:
            logger.info(
                "All shards were skipped (no matching VAD files) — nothing to do."
            )
            return

    args.shar_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_prepare_state(args)

    num_workers = ensure_worker_assignment(
        args.shar_dir, resolved, args.num_workers, _ITEMS_KEY, "WDS shards",
    )

    logger.info(f"Found {len(resolved)} WDS shards, using {num_workers} workers")
    logger.info(f"Output: {args.shar_dir}")

    # Distribute tar shards across workers (round-robin)
    worker_shards = distribute_round_robin(resolved, num_workers)

    if args.vad_segmentation:
        if args.vad_per_shard_dir is None:
            raise ValueError("--vad-per-shard-dir is required with --vad-segmentation")
        if not args.vad_per_shard_dir.is_dir():
            raise NotADirectoryError(f"VAD per-shard directory not found: {args.vad_per_shard_dir}")
        if args.vad_max_chunk_sec <= 0:
            raise ValueError("--vad-max-chunk-sec must be > 0")
        if args.vad_min_chunk_sec < 0:
            raise ValueError("--vad-min-chunk-sec must be >= 0")
        if args.vad_min_chunk_sec > args.vad_max_chunk_sec:
            raise ValueError("--vad-min-chunk-sec must be <= --vad-max-chunk-sec")
        if args.vad_sample_rate <= 0:
            raise ValueError("--vad-sample-rate must be > 0")
        if args.vad_max_merge_gap_sec < 0:
            raise ValueError("--vad-max-merge-gap-sec must be >= 0")
        if args.input_clip_id_parser is not None:
            raise ValueError(
                "--input-clip-id-parser cannot be combined with "
                "--vad-segmentation; input IDs already encode clip numbering."
            )
        logger.info(f"VAD segmenting enabled: per_shard_dir={args.vad_per_shard_dir}")
    else:
        logger.info("VAD segmenting disabled; writing full recordings")

    # Load external metadata before forking (COW-shared via fork).
    global _METADATA_PROVIDER
    if args.external_metadata:
        _METADATA_PROVIDER = ExternalMetadataProvider(
            metadata=load_external_metadata(
                args.external_metadata,
                tuple(args.custom_fields) if args.custom_fields else None,
                id_field=args.id_field,
                text_field=args.text_field,
            )
        )
        mp_start_method = "fork"
        logger.info("Using fork start method for COW sharing of external metadata")
    else:
        _METADATA_PROVIDER = None
        mp_start_method = "forkserver"

    # Load text tokenizer before forking (shared via COW across workers)
    text_tokenizer = load_text_tokenizer(args.text_tokenizer)

    worker_args = [
        (
            wid,
            shards,
            str(args.shar_dir),
            args.target_sr,
            args.shard_size,
            args.shar_format,
            args.min_sr,
            args.text_field,
            tuple(args.custom_fields) if args.custom_fields else None,
            not args.no_mono_downmix,
            str(args.vad_per_shard_dir) if args.vad_segmentation else None,
            args.vad_max_chunk_sec,
            args.vad_min_chunk_sec,
            args.vad_sample_rate,
            args.vad_max_merge_gap_sec,
            args.vad_max_duration_sec,
            text_tokenizer,
            args.resampling_backend,
            args.input_clip_id_parser,
            args.language,
        )
        for wid, shards in enumerate(worker_shards)
        if shards
    ]

    run_pool_and_finalize(_convert_worker, worker_args, args.shar_dir, num_workers,
                          mp_start_method=mp_start_method)


if __name__ == "__main__":
    main()
