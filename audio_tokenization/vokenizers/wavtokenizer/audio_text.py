"""Audio-text tokenizer using WavTokenizer.

Extends WavTokenizerAudioOnly with:
- ``tokenize_batch_raw()``: reuses ``tokenize_batch()`` then strips BOS/EOS,
  yielding ``[audio_start, offset_tokens..., audio_end]`` per clip.
- Caches new special token IDs: speech_transcribe, speech_switch, audio_annotate.

Follows the same pattern as vision's ``EMUImageTextPairTokenizer`` which
inherits from ``EMUImageOnlyTokenizer`` and slices BOS/EOS off the
encapsulated token sequence.
"""

from typing import Optional, Union

import numpy as np
import torch

from .audio_only import WavTokenizerAudioOnly


class WavTokenizerAudioText(WavTokenizerAudioOnly):
    """Audio-text tokenizer for interleaved training data.

    Inherits all audio_only functionality and adds ``tokenize_batch_raw()``
    which returns per-clip tokens suitable for Parquet caching.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._speech_transcribe_id = None
        self._speech_switch_id = None
        self._stt_translate_id = None
        self._audio_annotate_id = None

    def _cache_special_tokens(self):
        super()._cache_special_tokens()

        # Reuse mapping already loaded by parent (no second file read)
        st = self._mapping["structure_tokens"]

        required = ["stt_transcribe", "stt_continue", "stt_translate", "audio_annotate"]
        for key in required:
            if key not in st:
                raise ValueError(
                    f"Task token '{key}' missing from structure_tokens. "
                    "Rebuild the tokenizer with the latest omnitok modalities.py."
                )

        self._speech_transcribe_id = st["stt_transcribe"]
        self._speech_switch_id = st["stt_continue"]
        self._stt_translate_id = st["stt_translate"]
        self._audio_annotate_id = st["audio_annotate"]

    @property
    def speech_transcribe_id(self) -> int:
        if self._speech_transcribe_id is None:
            self._cache_special_tokens()
        return self._speech_transcribe_id

    @property
    def speech_switch_id(self) -> int:
        if self._speech_switch_id is None:
            self._cache_special_tokens()
        return self._speech_switch_id

    @property
    def stt_translate_id(self) -> int:
        if self._stt_translate_id is None:
            self._cache_special_tokens()
        return self._stt_translate_id

    @property
    def audio_annotate_id(self) -> int:
        if self._audio_annotate_id is None:
            self._cache_special_tokens()
        return self._audio_annotate_id

    def tokenize_batch_raw(
        self,
        audios: Union[list, np.ndarray, torch.Tensor],
        sample_rate: int,
        orig_audio_samples: Optional[list] = None,
        pad_audio_samples: Optional[int] = None,
    ) -> list:
        """Tokenize a batch and return per-clip tokens without BOS/EOS.

        Reuses ``tokenize_batch()`` which produces::

            [BOS, audio_start, offset_tokens..., audio_end, EOS]

        Then strips the first (BOS) and last (EOS) tokens to yield::

            [audio_start, offset_tokens..., audio_end]

        Returns:
            List of ``list[int]``, one per clip.
        """
        token_tensors = self.tokenize_batch(
            audios,
            sample_rate,
            orig_audio_samples=orig_audio_samples,
            pad_audio_samples=pad_audio_samples,
        )
        # Strip BOS (first) and EOS (last), then single batched GPU→CPU transfer
        stripped = [t[1:-1] for t in token_tensors]
        if not stripped:
            return []
        lengths = [t.shape[0] for t in stripped]
        all_cpu = torch.cat(stripped).cpu()
        return [chunk.tolist() for chunk in all_cpu.split(lengths)]
