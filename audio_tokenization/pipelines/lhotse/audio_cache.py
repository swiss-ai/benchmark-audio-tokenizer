"""Audio-cache mode handler for reusable SFT audio token assets."""

from __future__ import annotations

import torch

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.token_cache import AudioTokenCacheWriter, write_audio_token_cache_manifest

from ._unsupervised_batch import tokenize_unsupervised_batch


class AudioCacheHandler:
    """Tokenize audio-only SHAR cuts into ``audio_id -> token span`` cache rows."""

    def __init__(self, spec: TokenizeSpec):
        self.spec = spec
        self.chunk_samples = 0
        self.chunks_written = 0

    def create_dataset(self):
        from lhotse.dataset import UnsupervisedWaveformDataset

        return UnsupervisedWaveformDataset(collate=True)

    def setup_writer(self, output_dir, rank, writer_state, tokenizer):
        # The manifest describes the shared cache root, not a per-rank chunk.
        if int(rank) == 0:
            write_audio_token_cache_manifest(
                output_dir,
                tokenizer_path=self.spec.tokenizer.path,
                vocab_size=len(tokenizer.omni_tokenizer),
            )
        self._writer = AudioTokenCacheWriter(
            output_dir,
            rank=rank,
            chunk_id=int(writer_state),
        )
        self.chunk_samples = 0

    def process_batch(self, batch, tokenizer, stats, target_sr, device):
        encoded = tokenize_unsupervised_batch(
            batch,
            tokenizer,
            target_sr=target_sr,
            device=device,
            dtype=torch.int32,
            trim_prefix_tokens=1,
            trim_suffix_tokens=1,
        )
        stats.errors += encoded.errors

        for tokens, cut in encoded.tokens_and_cuts:
            self._writer.add(
                audio_id=cut.id,
                tokens=tokens.numpy(),
                duration_sec=cut.duration,
            )
            stats.samples_processed += 1
            stats.tokens_generated += int(tokens.numel())
            self.chunk_samples += 1

        return encoded.audio_seconds

    def checkpoint_writer(self) -> int:
        self._writer.finalize()
        self.chunk_samples = 0
        self.chunks_written += 1
        return self._writer.get_state()

    def get_writer_state(self) -> int:
        return self._writer.get_state()

    def finalize_writer(self):
        if self.chunk_samples > 0:
            self._writer.finalize()
            self.chunks_written += 1
