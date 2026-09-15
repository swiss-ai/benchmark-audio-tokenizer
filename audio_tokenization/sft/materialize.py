"""Materialize SFT conversations from conversation parquet rows and audio-token cache."""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from audio_tokenization.pipelines.lhotse.checkpoint import open_chunk_writer
from audio_tokenization.pipelines.shard_io import finalize_shard_writer
from audio_tokenization.prepare.runtime import resolve_num_workers

from audio_tokenization.token_cache import (
    AudioTokenCache,
    load_audio_token_cache,
    validate_audio_token_cache_manifest,
)
from audio_tokenization.utils.io import atomic_write_json


_SHARED_AUDIO_CACHE: AudioTokenCache | None = None
_SHARED_CHAT_TOKENIZER: Any = None

_MAX_MISSING_PREVIEW = 10

_SFT_SUM_FIELDS = (
    "samples_seen",
    "samples_processed",
    "missing_audio_skipped",
    "tokens_generated",
    "chunks_written",
    "stage2_samples",
    "lct_samples",
    "stage2_tokens",
    "lct_tokens",
)


def _extend_capped(
    dst: list[str], src: Iterable[str], cap: int, *, dedup: bool
) -> None:
    for item in src:
        if len(dst) >= cap:
            return
        if dedup and item in dst:
            continue
        dst.append(item)


@dataclass(frozen=True)
class SftMaterializeConfig:
    conversations_dir: str | Path
    cache_dir: str | Path
    output_dir: str | Path
    tokenizer_path: str | Path
    max_seq_len: int = 262144
    seq_threshold: int | None = None
    audio_placeholder: str = "<audio>"
    conversations_glob: str = "*.parquet"
    messages_column: str = "messages"
    audio_ids_column: str = "audio_ids"
    num_workers: int | None = None


@dataclass(frozen=True)
class _SftRowGroup:
    """One row group from an SFT conversations parquet file."""

    path: Path
    row_group: int
    num_rows: int


@dataclass(frozen=True)
class _SftWorkerArgs:
    worker_id: int
    row_groups: list[_SftRowGroup]
    config: SftMaterializeConfig
    vocab_size: int


class _SftShardWriter:
    """Lazy Megatron chunk writer for one SFT output bucket."""

    def __init__(self, output_dir: Path, *, worker_id: int, vocab_size: int):
        self.output_dir = output_dir
        self.worker_id = worker_id
        self.vocab_size = vocab_size
        self._opened = False
        self.samples = 0
        self.tokens = 0

    def _open(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (
            self.builder,
            self.cut_id_writer,
            self.tmp_bin,
            self.tmp_idx,
            _tmp_cut_ids,
            self.bin_path,
            self.idx_path,
            _cut_ids_path,
        ) = open_chunk_writer(
            str(self.output_dir),
            rank=self.worker_id,
            chunk_id=0,
            vocab_size=self.vocab_size,
        )
        self._opened = True

    def add(self, sample_id: str, seq: np.ndarray) -> None:
        if not self._opened:
            self._open()
        self.builder.add_item(seq)
        self.builder.end_document()
        self.cut_id_writer.write(sample_id)
        self.samples += 1
        self.tokens += int(seq.size)

    def finalize(self) -> bool:
        if not self._opened:
            return False
        finalize_shard_writer(
            self.builder,
            self.tmp_bin,
            self.tmp_idx,
            self.bin_path,
            self.idx_path,
            self.cut_id_writer,
        )
        return True

    def abort(self) -> None:
        if not self._opened:
            return
        self.cut_id_writer.abort()
        for path in (self.tmp_bin, self.tmp_idx):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


def load_sft_chat_tokenizer(tokenizer_path: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)


def materialize_sft(config: SftMaterializeConfig) -> dict[str, Any]:
    conversations_dir = Path(config.conversations_dir)
    output_dir = Path(config.output_dir)
    if not conversations_dir.is_dir():
        raise FileNotFoundError(f"SFT conversations_dir not found: {conversations_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.seq_threshold is not None:
        (output_dir / "stage2").mkdir(parents=True, exist_ok=True)
        (output_dir / "lct").mkdir(parents=True, exist_ok=True)

    columns = ["sample_id", config.messages_column, config.audio_ids_column]
    row_groups = _discover_sft_row_groups(
        conversations_dir,
        conversations_glob=config.conversations_glob,
        columns=columns,
    )
    num_workers = resolve_num_workers(config.num_workers, num_inputs=len(row_groups))
    assignments = _assign_sft_row_groups(row_groups, num_workers=num_workers)
    # Existing caches without _MANIFEST.json are rejected deliberately: SFT
    # materialization must not guess whether cached audio IDs use this tokenizer.
    cache_manifest = validate_audio_token_cache_manifest(config.cache_dir, tokenizer_path=config.tokenizer_path)
    cache = load_audio_token_cache(config.cache_dir)
    tokenizer = load_sft_chat_tokenizer(config.tokenizer_path)
    vocab_size = _sft_output_vocab_size(tokenizer, cache_manifest)
    worker_args = [
        _SftWorkerArgs(
            worker_id=worker_id,
            row_groups=items,
            config=config,
            vocab_size=vocab_size,
        )
        for worker_id, items in enumerate(assignments)
        if items
    ]

    if worker_args:
        # Build the membership set once in the parent so forked workers share it
        # through copy-on-write instead of rebuilding it per process.
        _ = cache.audio_ids
        _set_shared_audio_cache(cache)
        _set_shared_chat_tokenizer(tokenizer)
        try:
            if len(worker_args) == 1:
                worker_results = [_materialize_sft_worker(worker_args[0])]
            else:
                ctx = mp.get_context("fork")
                with ctx.Pool(processes=len(worker_args)) as pool:
                    worker_results = pool.map(_materialize_sft_worker, worker_args)
        finally:
            _set_shared_audio_cache(None)
            _set_shared_chat_tokenizer(None)
    else:
        worker_results = []

    merged = _merge_sft_worker_results(worker_results)
    if merged["samples_seen"] > 0 and merged["samples_processed"] == 0:
        raise ValueError(
            f"materialize: 0/{merged['samples_seen']} samples processed; "
            f"every conversation referenced audio missing from the cache. "
            f"first missing audio_ids: {merged['first_missing_audio_ids']}; "
            f"first affected sample_ids: {merged['first_missing_sample_ids']}. "
            "Likely wrong cache or wrong dataset config."
        )

    summary = {**merged, "num_workers": len(worker_args)}
    _write_sft_materialize_summary(output_dir, summary)
    return {
        **summary,
        "worker_results": worker_results,
        "output_dir": str(output_dir),
        "success": True,
    }


def _merge_sft_worker_results(worker_results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {k: sum(int(r.get(k, 0)) for r in worker_results) for k in _SFT_SUM_FIELDS}
    first_missing_audio_ids: list[str] = []
    first_missing_sample_ids: list[str] = []
    for r in worker_results:
        _extend_capped(
            first_missing_audio_ids,
            r.get("first_missing_audio_ids", []),
            _MAX_MISSING_PREVIEW,
            dedup=True,
        )
        _extend_capped(
            first_missing_sample_ids,
            r.get("first_missing_sample_ids", []),
            _MAX_MISSING_PREVIEW,
            dedup=False,
        )
    return {
        **totals,
        "first_missing_audio_ids": first_missing_audio_ids,
        "first_missing_sample_ids": first_missing_sample_ids,
    }


def _write_sft_materialize_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    """Persist run-level audit fields next to the materialized outputs.

    Separate from _STAGE_MANIFEST.json so the generic stage runner stays
    agnostic of stage-specific business data (skip counts, previews).
    """
    atomic_write_json(output_dir / "sft_materialize_summary.json", summary)


def _record_missing(
    *,
    missing: list[str],
    sample_id: str,
    first_missing_audio_ids: list[str],
    first_missing_sample_ids: list[str],
) -> None:
    _extend_capped(first_missing_audio_ids, missing, _MAX_MISSING_PREVIEW, dedup=True)
    _extend_capped(
        first_missing_sample_ids, [sample_id], _MAX_MISSING_PREVIEW, dedup=False
    )


def _materialize_sft_worker(args: _SftWorkerArgs) -> dict[str, Any]:
    config = args.config
    output_dir = Path(config.output_dir)
    cache = _require_shared_audio_cache()
    tokenizer = _require_shared_chat_tokenizer()
    vocab_size = args.vocab_size
    cache_audio_ids = cache.audio_ids

    samples_seen = 0
    samples_processed = 0
    missing_audio_skipped = 0
    first_missing_audio_ids: list[str] = []
    first_missing_sample_ids: list[str] = []
    tokens_written = 0
    bucket_samples = {"stage2": 0, "lct": 0}
    bucket_tokens = {"stage2": 0, "lct": 0}
    writers: dict[str, _SftShardWriter] = {}
    try:
        for row in _iter_sft_rows(
            args.row_groups,
            columns=["sample_id", config.messages_column, config.audio_ids_column],
        ):
            samples_seen += 1
            sample_id = str(row["sample_id"])
            messages = coerce_messages(
                row[config.messages_column],
                sample_id=sample_id,
                column=config.messages_column,
            )
            audio_ids = ordered_audio_ids(
                row.get(config.audio_ids_column),
                messages=messages,
                sample_id=sample_id,
            )
            missing = [a for a in audio_ids if a not in cache_audio_ids]
            if missing:
                missing_audio_skipped += 1
                _record_missing(
                    missing=missing,
                    sample_id=sample_id,
                    first_missing_audio_ids=first_missing_audio_ids,
                    first_missing_sample_ids=first_missing_sample_ids,
                )
                continue

            seq = assemble_sft_conversation(
                sample_id=sample_id,
                messages=messages,
                audio_ids=audio_ids,
                cache=cache,
                tokenizer=tokenizer,
                audio_placeholder=config.audio_placeholder,
            )
            seq_len = int(seq.size)
            if seq_len > int(config.max_seq_len):
                raise ValueError(
                    f"SFT sample {sample_id!r} has {seq_len} tokens, "
                    f"exceeding max_seq_len={config.max_seq_len}"
                )
            bucket_name, bucket_dir = _sft_output_bucket(output_dir, config, seq_len)
            writer = writers.get(bucket_name)
            if writer is None:
                writer = _SftShardWriter(bucket_dir, worker_id=args.worker_id, vocab_size=vocab_size)
                writers[bucket_name] = writer
            writer.add(sample_id, seq)
            samples_processed += 1
            tokens_written += seq_len
            if bucket_name in bucket_samples:
                bucket_samples[bucket_name] += 1
                bucket_tokens[bucket_name] += seq_len
        chunks_written = sum(int(writer.finalize()) for writer in writers.values())
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise

    return {
        "worker_id": args.worker_id,
        "row_groups": len(args.row_groups),
        "samples_seen": samples_seen,
        "samples_processed": samples_processed,
        "missing_audio_skipped": missing_audio_skipped,
        "first_missing_audio_ids": first_missing_audio_ids,
        "first_missing_sample_ids": first_missing_sample_ids,
        "tokens_generated": tokens_written,
        "chunks_written": chunks_written,
        "stage2_samples": bucket_samples["stage2"],
        "lct_samples": bucket_samples["lct"],
        "stage2_tokens": bucket_tokens["stage2"],
        "lct_tokens": bucket_tokens["lct"],
    }


def _sft_output_vocab_size(tokenizer: Any, cache_manifest: dict[str, Any]) -> int:
    cache_vocab_size = cache_manifest.get("vocab_size")
    if cache_vocab_size is None:
        return len(tokenizer)
    return max(len(tokenizer), int(cache_vocab_size))


def _sft_output_bucket(
    output_dir: Path,
    config: SftMaterializeConfig,
    seq_len: int,
) -> tuple[str, Path]:
    if config.seq_threshold is None:
        return "all", output_dir
    if seq_len <= int(config.seq_threshold):
        return "stage2", output_dir / "stage2"
    return "lct", output_dir / "lct"


def _set_shared_audio_cache(cache: AudioTokenCache | None) -> None:
    global _SHARED_AUDIO_CACHE
    _SHARED_AUDIO_CACHE = cache


def _require_shared_audio_cache() -> AudioTokenCache:
    if _SHARED_AUDIO_CACHE is None:
        raise RuntimeError("SFT audio token cache was not initialised before worker start")
    return _SHARED_AUDIO_CACHE


def _set_shared_chat_tokenizer(tokenizer: Any) -> None:
    global _SHARED_CHAT_TOKENIZER
    _SHARED_CHAT_TOKENIZER = tokenizer


def _require_shared_chat_tokenizer() -> Any:
    if _SHARED_CHAT_TOKENIZER is None:
        raise RuntimeError("SFT chat tokenizer was not initialised before worker start")
    return _SHARED_CHAT_TOKENIZER


def assemble_sft_conversation(
    *,
    sample_id: str,
    messages: list[dict[str, Any]],
    audio_ids: list[str],
    cache: AudioTokenCache,
    tokenizer: Any,
    audio_placeholder: str = "<audio>",
) -> np.ndarray:
    render_messages = render_structured_audio_attachments(
        messages,
        audio_placeholder=audio_placeholder,
        sample_id=sample_id,
    )
    rendered = _render_messages(tokenizer, render_messages)
    text_spans = rendered.split(audio_placeholder)
    placeholder_count = len(text_spans) - 1
    if placeholder_count != len(audio_ids):
        raise ValueError(
            f"SFT sample {sample_id!r} has {placeholder_count} audio placeholders "
            f"but {len(audio_ids)} audio ids"
        )

    text_tokens = _tokenize_text_spans(tokenizer, text_spans)
    chunks: list[np.ndarray] = []
    for idx, encoded in enumerate(text_tokens):
        if encoded.size:
            chunks.append(encoded)
        if idx < len(audio_ids):
            chunks.append(cache.read(audio_ids[idx]))

    body = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
    return _ensure_bos_eos(
        body,
        bos_id=_require_token_id(tokenizer, "bos_token_id"),
        eos_id=_require_token_id(tokenizer, "eos_token_id"),
    )


def _discover_sft_row_groups(
    conversations_dir: Path,
    *,
    conversations_glob: str,
    columns: list[str],
) -> list[_SftRowGroup]:
    paths = sorted(conversations_dir.glob(conversations_glob))
    if not paths:
        raise FileNotFoundError(
            f"No SFT conversation parquet files matching {conversations_glob!r} under {conversations_dir}"
        )
    row_groups: list[_SftRowGroup] = []
    required = ("sample_id", columns[1])
    for path in paths:
        pf = pq.ParquetFile(path)
        select_conversation_columns(pf, path=path, columns=columns, required=required)
        for row_group in range(pf.num_row_groups):
            metadata = pf.metadata.row_group(row_group)
            if metadata.num_rows <= 0:
                continue
            row_groups.append(
                _SftRowGroup(
                    path=path,
                    row_group=row_group,
                    num_rows=int(metadata.num_rows),
                )
            )
    return row_groups


def _assign_sft_row_groups(
    row_groups: list[_SftRowGroup],
    *,
    num_workers: int,
) -> list[list[_SftRowGroup]]:
    assignments: list[list[_SftRowGroup]] = [[] for _ in range(max(1, num_workers))]
    totals = [0 for _ in assignments]
    for item in sorted(row_groups, key=lambda x: x.num_rows, reverse=True):
        worker_id = min(range(len(assignments)), key=totals.__getitem__)
        assignments[worker_id].append(item)
        totals[worker_id] += item.num_rows
    return assignments


def _iter_sft_rows(
    row_groups: list[_SftRowGroup],
    *,
    columns: list[str],
) -> Iterable[dict[str, Any]]:
    by_path: dict[Path, list[int]] = {}
    for row_group in row_groups:
        by_path.setdefault(row_group.path, []).append(row_group.row_group)
    required = ("sample_id", columns[1])
    for path, group_ids in by_path.items():
        pf = pq.ParquetFile(path)
        selected = select_conversation_columns(pf, path=path, columns=columns, required=required)
        for batch in pf.iter_batches(columns=selected, row_groups=group_ids):
            for row in batch.to_pylist():
                yield row


def select_conversation_columns(
    pf: pq.ParquetFile,
    *,
    path: Path,
    columns: list[str],
    required: tuple[str, ...] = ("sample_id",),
) -> list[str]:
    available = set(pf.schema_arrow.names)
    selected = [column for column in columns if column in available]
    missing_required = set(required) - set(selected)
    if missing_required:
        raise ValueError(f"{path} is missing required SFT columns {sorted(missing_required)}")
    return selected


def coerce_messages(value: Any, *, sample_id: str, column: str) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SFT sample {sample_id!r} has invalid JSON in {column!r}"
            ) from exc
    if not isinstance(value, list):
        raise TypeError(
            f"SFT sample {sample_id!r} column {column!r} must be a message list "
            f"or JSON-encoded message list; got {type(value).__name__}"
        )
    if not all(isinstance(message, dict) for message in value):
        raise TypeError(f"SFT sample {sample_id!r} column {column!r} contains non-object messages")
    return value


def ordered_audio_ids(
    value: Any,
    *,
    messages: list[dict[str, Any]],
    sample_id: str,
) -> list[str]:
    # Distinguish "column absent" (fall back to messages) from "column present
    # but empty" (text-only sample, no audio).
    message_audio_ids = _message_audio_ids(messages)
    if value is not None:
        column_audio_ids = [str(audio_id) for audio_id in value]
        if message_audio_ids and column_audio_ids != message_audio_ids:
            raise ValueError(
                f"SFT sample {sample_id!r} audio_ids column does not match "
                "messages[].audio order"
            )
        return column_audio_ids

    return message_audio_ids


def _message_audio_ids(messages: list[dict[str, Any]]) -> list[str]:
    audio_ids: list[str] = []
    for message in messages or []:
        for audio in _message_audio_entries(message):
            audio_id = audio.get("audio_id")
            if audio_id:
                audio_ids.append(str(audio_id))
    return audio_ids


def _message_audio_entries(message: dict[str, Any]) -> list[dict[str, Any]]:
    value = message.get("audio")
    if not value:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError("SFT message field 'audio' must be a list of objects when present")
    return value


def render_structured_audio_attachments(
    messages: list[dict[str, Any]],
    *,
    audio_placeholder: str,
    sample_id: str,
) -> list[dict[str, Any]]:
    """Render messages[].audio attachments as placeholders at their own turn.

    Some processed SFT packages store placement structurally instead of writing
    literal ``<audio>`` into the message text. We only synthesize placeholders
    from those explicit attachments; a bare top-level ``audio_ids`` list still
    needs literal placeholders and will fail during the final count check.
    """
    rendered: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        attachments = _message_audio_entries(message)
        if not attachments:
            rendered.append(message)
            continue

        updated = dict(message)
        content = "" if message.get("content") is None else str(message.get("content"))
        placeholder_count = content.count(audio_placeholder)
        if placeholder_count == 0:
            placeholders = "\n".join(audio_placeholder for _ in attachments)
            updated["content"] = placeholders if not content else f"{placeholders}\n{content}"
        elif placeholder_count != len(attachments):
            raise ValueError(
                f"SFT sample {sample_id!r} message {message_index} has "
                f"{placeholder_count} audio placeholders but {len(attachments)} "
                "messages[].audio attachments"
            )
        rendered.append(updated)
    return rendered


def _render_messages(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            rendered = apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(rendered, str):
                return rendered
        except Exception:
            if getattr(tokenizer, "chat_template", None):
                raise
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def _tokenize_text(tokenizer: Any, text: str) -> np.ndarray:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = getattr(encoded, "input_ids", encoded.get("input_ids") if isinstance(encoded, dict) else encoded)
    return np.asarray(ids, dtype=np.int32)


def _tokenize_text_spans(tokenizer: Any, spans: list[str]) -> list[np.ndarray]:
    """Tokenize text spans in one batch where the tokenizer supports list input."""
    if not spans:
        return []
    if len(spans) == 1:
        return [_tokenize_text(tokenizer, spans[0])]
    try:
        encoded = tokenizer(spans, add_special_tokens=False)
    except (TypeError, AssertionError):
        return [_tokenize_text(tokenizer, span) for span in spans]
    ids = getattr(encoded, "input_ids", None)
    if ids is None and isinstance(encoded, dict):
        ids = encoded.get("input_ids")
    if ids is None or len(ids) != len(spans):
        return [_tokenize_text(tokenizer, span) for span in spans]
    return [np.asarray(row, dtype=np.int32) for row in ids]


def _ensure_bos_eos(tokens: np.ndarray, *, bos_id: int, eos_id: int) -> np.ndarray:
    add_bos = tokens.size == 0 or int(tokens[0]) != int(bos_id)
    add_eos = tokens.size == 0 or int(tokens[-1]) != int(eos_id)
    if not add_bos and not add_eos and tokens.dtype == np.int32:
        return tokens
    out = np.empty(int(add_bos) + tokens.size + int(add_eos), dtype=np.int32)
    cursor = 0
    if add_bos:
        out[cursor] = bos_id
        cursor += 1
    if tokens.size:
        out[cursor:cursor + tokens.size] = tokens
        cursor += tokens.size
    if add_eos:
        out[cursor] = eos_id
    return out


def _require_token_id(tokenizer: Any, name: str) -> int:
    value = getattr(tokenizer, name, None)
    if value is None:
        raise ValueError(f"text tokenizer must define {name}")
    return int(value)
