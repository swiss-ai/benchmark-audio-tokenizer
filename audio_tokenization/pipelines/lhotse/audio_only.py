"""AudioOnly mode handler for the Lhotse tokenization pipeline.

Writes Megatron indexed dataset micro-shards (``rank_XXXX_chunk_YYYY.{bin,idx}``).
"""

import torch

from audio_tokenization.config.schema import TokenizeSpec

from ._unsupervised_batch import tokenize_unsupervised_batch
from .checkpoint import MegatronWriterMixin


class AudioOnlyHandler(MegatronWriterMixin):
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
        self._setup_megatron_writer(output_dir, rank, writer_state, tokenizer)

    def process_batch(self, batch, tokenizer, stats, target_sr, device):
        encoded = tokenize_unsupervised_batch(
            batch,
            tokenizer,
            target_sr=target_sr,
            device=device,
            dtype=torch.int64,
        )

        for t, cut in encoded.tokens_and_cuts:
            self._builder.add_item(t)
            self._builder.end_document()
            self._cut_ids.write(cut.id)
            stats.samples_processed += 1
            stats.tokens_generated += len(t)
            self.chunk_samples += 1

        return encoded.audio_seconds

    def checkpoint_writer(self) -> int:
        return self._rotate_megatron_writer()

    def finalize_writer(self):
        self._finalize_megatron_writer()

    def abort_writer(self):
        self._abort_megatron_writer()
