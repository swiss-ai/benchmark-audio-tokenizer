"""AudioOnly mode handler for the Lhotse tokenization pipeline.

Writes Megatron indexed dataset micro-shards (``rank_XXXX_chunk_YYYY.{bin,idx}``).
"""

import os

import torch

from audio_tokenization.config.schema import TokenizeSpec

from ._unsupervised_batch import tokenize_unsupervised_batch
from .checkpoint import finalize_shard_writer, open_chunk_writer


class AudioOnlyHandler:
    """Handler for audio-only tokenization mode.

    Uses WavTokenizer to encode audio into token sequences wrapped in
    ``[BOS, audio_start, tokens..., audio_end, EOS]`` and writes them
    to Megatron indexed dataset micro-shards.
    """

    def __init__(self, spec: TokenizeSpec):
        self.chunk_samples = 0
        self.chunks_written = 0

    def create_dataset(self):
        from lhotse.dataset import UnsupervisedWaveformDataset
        return UnsupervisedWaveformDataset(collate=True)

    def setup_writer(self, output_dir, rank, writer_state, tokenizer):
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
        self.chunk_samples = 0

    def process_batch(self, batch, tokenizer, stats, target_sr, device):
        encoded = tokenize_unsupervised_batch(
            batch,
            tokenizer,
            target_sr=target_sr,
            device=device,
            dtype=torch.int64,
        )
        stats.errors += encoded.errors

        for t, cut in encoded.tokens_and_cuts:
            self._builder.add_item(t)
            self._builder.end_document()
            self._cut_ids.write(cut.id)
            stats.samples_processed += 1
            stats.tokens_generated += len(t)
            self.chunk_samples += 1

        return encoded.audio_seconds

    def checkpoint_writer(self) -> int:
        finalize_shard_writer(
            self._builder,
            self._tmp_bin,
            self._tmp_idx,
            self._bin,
            self._idx,
            self._cut_ids,
        )
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
        return self._chunk_id

    def get_writer_state(self) -> int:
        return self._chunk_id

    def finalize_writer(self):
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
            self._chunk_id += 1
        else:
            self._cut_ids.abort()
            for p in (self._tmp_bin, self._tmp_idx):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
