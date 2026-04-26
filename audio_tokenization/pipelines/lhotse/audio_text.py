"""AudioText mode handler for the Lhotse tokenization pipeline.

Supports two output formats:
- ``interleaved`` (default): Parquet cache files for downstream interleaving.
- ``direct``: Megatron indexed dataset (bin/idx) with full training sequences.
"""

import os

import torch

from audio_tokenization.config.schema import TokenizeSpec

from .checkpoint import finalize_shard_writer, open_chunk_writer

TASK_TOKEN_MAP = {
    "transcribe": "speech_transcribe_id",
    "translate": "stt_translate_id",
    "annotate": "audio_annotate_id",
}


def resolve_interleaving_metadata(cut):
    """Resolve canonical interleaving metadata for a cut.

    Reads ``source_id``, ``clip_num``, and optional ``clip_start`` /
    ``clip_duration`` from ``cut.custom["interleave"]``. These must be set
    during SHAR conversion. ``clip_end`` is derived for materialize-time gap
    detection and is not stored in SHAR metadata.
    """
    custom = cut.custom or {}
    interleave = custom.get("interleave")
    if not isinstance(interleave, dict):
        interleave = {}

    source_id = interleave.get("source_id")
    clip_num = interleave.get("clip_num")
    clip_start = interleave.get("clip_start")
    clip_duration = interleave.get("clip_duration")
    if clip_start is not None and clip_duration is not None:
        clip_start = float(clip_start)
        clip_duration = float(clip_duration)
    else:
        clip_start = None
        clip_duration = None

    if source_id is not None and clip_num is not None:
        return str(source_id), int(clip_num), clip_start, clip_duration

    raise ValueError(
        f"Cut {cut.id!r} is missing interleaving metadata "
        "(cut.custom.interleave.source_id / clip_num). Reconvert the SHAR "
        "with interleave metadata before tokenization."
    )


class AudioTextHandler:
    """Handler for audio-text tokenization mode.

    Uses WavTokenizer to encode audio into token sequences, pairs them
    with text metadata from Lhotse cuts, and writes output in one of two
    formats controlled by ``audio_text_format``:

    - ``interleaved``: Structured cache with metadata Parquet plus flat token bins.
    - ``direct``: Full ``[BOS, audio_start, audio_tokens, audio_end,
      task_token, text_tokens, EOS]`` sequences to Megatron bin/idx.
    """

    def __init__(self, spec: TokenizeSpec, *, dataset_name: str):
        self.dataset_name = dataset_name
        self.chunk_samples = 0

        self.audio_text_format = spec.audio_text_format
        self.audio_text_task = spec.audio_text_task
        if self.audio_text_format not in ("direct", "interleaved"):
            raise ValueError(
                f"Unsupported audio_text_format: {self.audio_text_format!r}. "
                f"Must be 'direct' or 'interleaved'."
            )
        if self.audio_text_task not in TASK_TOKEN_MAP:
            raise ValueError(
                f"Unsupported audio_text_task: {self.audio_text_task!r}. "
                f"Must be one of {list(TASK_TOKEN_MAP)}."
            )
        self.partitioning = spec.partitioning
        self.chunks_written = 0

    def create_dataset(self):
        from lhotse.dataset import K2SpeechRecognitionDataset
        from lhotse.dataset.input_strategies import AudioSamples
        return K2SpeechRecognitionDataset(
            return_cuts=True,
            input_strategy=AudioSamples(),
        )

    # ------------------------------------------------------------------
    # Writer lifecycle
    # ------------------------------------------------------------------

    def setup_writer(self, output_dir, rank, writer_state, tokenizer):
        if self.audio_text_format == "direct":
            self._setup_writer_direct(output_dir, rank, writer_state, tokenizer)
        else:
            self._setup_writer_interleaved(output_dir, rank, writer_state)
        self.chunk_samples = 0

    def _setup_writer_direct(self, output_dir, rank, writer_state, tokenizer):
        self._output_dir = output_dir
        self._rank = rank
        self._chunk_id = int(writer_state)
        self._vocab_size = len(tokenizer.omni_tokenizer)
        (
            self._builder,
            self._cut_ids,
            self._tmp_bin,
            self._tmp_idx,
            self._tmp_cut_ids,
            self._bin,
            self._idx,
            self._cut_ids_path,
        ) = \
            open_chunk_writer(output_dir, rank, self._chunk_id, self._vocab_size)

    def _setup_writer_interleaved(self, output_dir, rank, writer_state):
        from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter

        self._writer = StructuredCacheChunkWriter(
            output_dir,
            rank,
            writer_state=writer_state,
            partitioning=self.partitioning,
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_batch(self, batch, tokenizer, stats, target_sr, device):
        if self.audio_text_format == "direct":
            return self._process_batch_direct(batch, tokenizer, stats, target_sr, device)
        else:
            return self._process_batch_interleaved(batch, tokenizer, stats, target_sr, device)

    def _process_batch_direct(self, batch, tokenizer, stats, target_sr, device):
        audios = batch["inputs"]
        cuts = batch["supervisions"]["cut"]

        audio_lens = torch.tensor(
            [c.num_samples for c in cuts], dtype=torch.int64,
        )

        batch_audio_secs = audio_lens.sum().item() / target_sr
        audios_gpu = audios.to(device, non_blocking=True)

        with torch.inference_mode():
            raw_tokens = tokenizer.tokenize_batch_raw(
                audios_gpu,
                target_sr,
                orig_audio_samples=audio_lens.tolist(),
                pad_audio_samples=audios.shape[1],
            )

        task_token_id = getattr(tokenizer, TASK_TOKEN_MAP[self.audio_text_task])
        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id

        batch_audio_tok = 0
        batch_text_tok = 0
        for audio_tok, cut in zip(raw_tokens, cuts):
            text_tokens = cut.custom.get("text_tokens", []) if cut.custom else []
            # audio_tok = [audio_start, offset_audio..., audio_end]
            # Full: [BOS] + audio_tok + [task_token] + text_tokens + [EOS]
            seq = [bos_id] + audio_tok + [task_token_id] + text_tokens + [eos_id]
            t = torch.tensor(seq, dtype=torch.int64)
            self._builder.add_item(t)
            self._builder.end_document()
            self._cut_ids.write(cut.id)

            batch_audio_tok += len(audio_tok)
            batch_text_tok += len(text_tokens)
            self.chunk_samples += 1

        stats.samples_processed += len(raw_tokens)
        stats.tokens_generated += batch_audio_tok
        stats.text_tokens_generated += batch_text_tok

        return batch_audio_secs

    def _process_batch_interleaved(self, batch, tokenizer, stats, target_sr, device):
        audios = batch["inputs"]
        cuts = batch["supervisions"]["cut"]

        audio_lens = torch.tensor(
            [c.num_samples for c in cuts], dtype=torch.int64,
        )

        batch_audio_secs = audio_lens.sum().item() / target_sr
        audios_gpu = audios.to(device, non_blocking=True)

        with torch.inference_mode():
            raw_tokens = tokenizer.tokenize_batch_raw(
                audios_gpu,
                target_sr,
                orig_audio_samples=audio_lens.tolist(),
                pad_audio_samples=audios.shape[1],
            )

        rows = []
        batch_audio_tok = 0
        batch_text_tok = 0
        for tokens, cut in zip(raw_tokens, cuts):
            source_id, clip_num, clip_start, clip_duration = resolve_interleaving_metadata(cut)
            text = cut.supervisions[0].text if cut.supervisions else ""
            speaker = cut.supervisions[0].speaker if cut.supervisions else ""
            text_tokens = cut.custom.get("text_tokens", []) if cut.custom else []

            rows.append({
                "clip_id": cut.id,
                "source_id": source_id,
                "clip_num": clip_num,
                "clip_start": clip_start,
                "clip_duration": clip_duration,
                "speaker": speaker or "",
                "duration": cut.duration,
                "text": text or "",
                "text_tokens": text_tokens,
                "audio_tokens": tokens,
                "dataset": self.dataset_name,
            })
            if self.partitioning and self.partitioning["type"] == "field":
                field = self.partitioning["field"]
                if field not in rows[-1]:
                    raise ValueError(
                        f"Field partitioning requires generated cache column {field!r}; "
                        f"available columns: {sorted(rows[-1])}"
                    )
                rows[-1]["_partition_value"] = rows[-1][field]
            batch_audio_tok += len(tokens)
            batch_text_tok += len(text_tokens)

        self._writer.add_rows(rows)

        stats.samples_processed += len(rows)
        stats.tokens_generated += batch_audio_tok
        stats.text_tokens_generated += batch_text_tok
        self.chunk_samples += len(rows)

        return batch_audio_secs

    # ------------------------------------------------------------------
    # Checkpoint / finalize
    # ------------------------------------------------------------------

    def checkpoint_writer(self):
        if self.audio_text_format == "direct":
            return self._checkpoint_writer_direct()
        else:
            return self._checkpoint_writer_interleaved()

    def _checkpoint_writer_direct(self) -> int:
        finalize_shard_writer(
            self._builder,
            self._tmp_bin,
            self._tmp_idx,
            self._bin,
            self._idx,
            self._cut_ids,
        )
        done = self._chunk_id
        self._chunk_id += 1
        self.chunk_samples = 0
        self.chunks_written += 1
        (
            self._builder,
            self._cut_ids,
            self._tmp_bin,
            self._tmp_idx,
            self._tmp_cut_ids,
            self._bin,
            self._idx,
            self._cut_ids_path,
        ) = \
            open_chunk_writer(self._output_dir, self._rank, self._chunk_id, self._vocab_size)
        return done

    def _checkpoint_writer_interleaved(self):
        done_id = self._writer.finalize()
        self.chunk_samples = 0
        self.chunks_written += len(done_id)
        return done_id

    def get_writer_state(self):
        if self.audio_text_format == "direct":
            return self._chunk_id
        return self._writer.get_state()

    def finalize_writer(self):
        if self.audio_text_format == "direct":
            self._finalize_writer_direct()
        else:
            self._finalize_writer_interleaved()

    def _finalize_writer_direct(self):
        if self.chunk_samples > 0:
            finalize_shard_writer(
                self._builder,
                self._tmp_bin,
                self._tmp_idx,
                self._bin,
                self._idx,
                self._cut_ids,
            )
            self.chunks_written += 1
        else:
            self._cut_ids.abort()
            for p in (self._tmp_bin, self._tmp_idx):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    def _finalize_writer_interleaved(self):
        if self.chunk_samples > 0:
            done = self._writer.finalize()
            self.chunks_written += len(done)
