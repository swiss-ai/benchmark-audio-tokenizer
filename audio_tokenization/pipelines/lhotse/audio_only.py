"""AudioOnly mode handler for the Lhotse tokenization pipeline.

Writes Megatron indexed dataset micro-shards (``rank_XXXX_chunk_YYYY.{bin,idx}``).
"""

import os

import torch

from .checkpoint import finalize_shard_writer, open_chunk_writer


class AudioOnlyHandler:
    """Handler for audio-only tokenization mode.

    Uses WavTokenizer to encode audio into token sequences wrapped in
    ``[BOS, audio_start, tokens..., audio_end, EOS]`` and writes them
    to Megatron indexed dataset micro-shards.
    """

    def __init__(self, cfg):
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
        self._builder, self._tmp_bin, self._tmp_idx, self._bin, self._idx = \
            open_chunk_writer(output_dir, rank, self._chunk_id, self._vocab_size)
        self.chunk_samples = 0

    def process_batch(self, batch, tokenizer, stats, target_sr, device):
        # UnsupervisedWaveformDataset(collate=True) returns:
        #   {"audio": tensor (B, T), "audio_lens": tensor (B,)}
        audios = batch["audio"]           # (B, T) float
        audio_lens = batch["audio_lens"]  # (B,) int -- original lengths

        batch_audio_secs = audio_lens.sum().item() / target_sr
        audios_gpu = audios.to(device, non_blocking=True)

        with torch.inference_mode():
            token_list = tokenizer.tokenize_batch(
                audios_gpu,
                target_sr,
                orig_audio_samples=audio_lens.tolist(),
                pad_audio_samples=audios.shape[1],
            )

        # Single batched GPU→CPU transfer: cat, one sync, split
        valid = [t for t in token_list if t is not None]
        stats.errors += len(token_list) - len(valid)
        if not valid:
            return batch_audio_secs

        lengths = [t.shape[0] for t in valid]
        all_cpu = torch.cat(valid).to(dtype=torch.int64).cpu()
        cpu_tensors = all_cpu.split(lengths)

        for t in cpu_tensors:
            self._builder.add_item(t)
            self._builder.end_document()
            stats.samples_processed += 1
            stats.tokens_generated += len(t)
            self.chunk_samples += 1

        return batch_audio_secs

    def checkpoint_writer(self) -> int:
        finalize_shard_writer(self._builder, self._tmp_bin, self._tmp_idx, self._bin, self._idx)
        done = self._chunk_id
        self._chunk_id += 1
        self.chunk_samples = 0
        self.chunks_written += 1
        self._builder, self._tmp_bin, self._tmp_idx, self._bin, self._idx = \
            open_chunk_writer(self._output_dir, self._rank, self._chunk_id, self._vocab_size)
        return done

    def get_writer_state(self) -> int:
        return self._chunk_id

    def finalize_writer(self):
        if self.chunk_samples > 0:
            finalize_shard_writer(self._builder, self._tmp_bin, self._tmp_idx, self._bin, self._idx)
            self.chunks_written += 1
        else:
            for p in (self._tmp_bin, self._tmp_idx):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
