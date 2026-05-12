"""Tokenization helpers for Lhotse ``UnsupervisedWaveformDataset`` batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TokenizedUnsupervisedBatch:
    """CPU token tensors matched to the original unsupervised Lhotse cuts."""

    audio_seconds: float
    errors: int
    tokens_and_cuts: list[tuple[torch.Tensor, Any]]


def tokenize_unsupervised_batch(
    batch: dict[str, Any],
    tokenizer: Any,
    *,
    target_sr: int,
    device: str,
    dtype: torch.dtype,
    trim_prefix_tokens: int = 0,
    trim_suffix_tokens: int = 0,
) -> TokenizedUnsupervisedBatch:
    """Tokenize ``audio/audio_lens/cuts`` batches from ``UnsupervisedWaveformDataset``."""
    if trim_prefix_tokens < 0 or trim_suffix_tokens < 0:
        raise ValueError("trim token counts must be >= 0")

    audios = batch["audio"]
    audio_lens = batch["audio_lens"]
    cuts = list(batch["cuts"])

    audio_seconds = float(audio_lens.sum().item() / target_sr)
    audios_gpu = audios.to(device, non_blocking=True)

    with torch.inference_mode():
        token_list = tokenizer.tokenize_batch(
            audios_gpu,
            target_sr,
            orig_audio_samples=audio_lens.tolist(),
            pad_audio_samples=audios.shape[1],
        )

    valid: list[tuple[torch.Tensor, Any]] = [
        (tokens, cut) for tokens, cut in zip(token_list, cuts) if tokens is not None
    ]
    errors = len(token_list) - len(valid)
    if not valid:
        return TokenizedUnsupervisedBatch(audio_seconds, errors, [])

    trimmed = [
        _slice_tokens(
            tokens,
            trim_prefix_tokens=trim_prefix_tokens,
            trim_suffix_tokens=trim_suffix_tokens,
        )
        for tokens, _cut in valid
    ]
    lengths = [int(tokens.shape[0]) for tokens in trimmed]

    # One concatenated GPU-to-CPU copy avoids synchronizing per sample.
    all_cpu = torch.cat(trimmed).to(dtype=dtype).cpu()
    cpu_tokens = all_cpu.split(lengths)
    return TokenizedUnsupervisedBatch(
        audio_seconds,
        errors,
        [(tokens, cut) for tokens, (_raw_tokens, cut) in zip(cpu_tokens, valid)],
    )


def _slice_tokens(
    tokens: torch.Tensor,
    *,
    trim_prefix_tokens: int,
    trim_suffix_tokens: int,
) -> torch.Tensor:
    start = trim_prefix_tokens
    stop = -trim_suffix_tokens if trim_suffix_tokens else None
    return tokens[start:stop]
