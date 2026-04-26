"""Audio-only tokenizer using WavTokenizer.

Follows the same pattern as vision_tokenization/vokenizers/emu/image_only.py:
- Loads omni_tokenizer to get special token IDs
- Gets audio token offset by querying: convert_tokens_to_ids("<|audio token 0|>")
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import numpy as np

# Add the src directory to path
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from audio_tokenizers.implementations.wavtokenizer import WavTokenizer40


class WavTokenizerAudioOnly:
    """Audio-only tokenizer using WavTokenizer-40.

    Wraps WavTokenizer40 and adds encapsulation with
    <|audio_start|> and <|audio_end|> tokens.

    Output: [audio_start] [audio_tokens...] [audio_end]

    The audio token offset is determined by querying the omni_tokenizer
    for the first audio token, just like vision does:
        first_audio_token = tokenizer.convert_tokens_to_ids("<|audio token 0|>")
    """

    def __init__(
        self,
        omni_tokenizer_path: str,
        device: str = "cuda",
        torch_compile: bool = True,
        trim_last_tokens: int = 5,
        **kwargs,
    ):
        self.device = torch.device(device)
        self.omni_tokenizer_path = omni_tokenizer_path
        self.torch_compile = torch_compile
        self.trim_last_tokens = max(0, int(trim_last_tokens))

        # Lazy load
        self._wavtokenizer = None
        self._omni_tokenizer = None
        self._bos_id = None
        self._eos_id = None
        self._audio_start_id = None
        self._audio_end_id = None
        self._audio_token_offset = None

    @property
    def wavtokenizer(self) -> WavTokenizer40:
        if self._wavtokenizer is None:
            self._wavtokenizer = WavTokenizer40(device=self.device, torch_compile=self.torch_compile)
        return self._wavtokenizer

    @property
    def omni_tokenizer(self):
        """Load and cache omni_tokenizer."""
        if self._omni_tokenizer is None:
            from transformers import AutoTokenizer
            self._omni_tokenizer = AutoTokenizer.from_pretrained(
                self.omni_tokenizer_path, trust_remote_code=True, use_fast=True
            )
        return self._omni_tokenizer

    def _cache_special_tokens(self):
        """Cache special token IDs from omni_tokenizer (same pattern as vision)."""
        from audio_tokenization.utils.token_mapping import load_audio_token_mapping

        # BOS and EOS tokens
        assert self.omni_tokenizer.bos_token is not None, "BOS token must be defined"
        assert self.omni_tokenizer.eos_token is not None, "EOS token must be defined"
        self._bos_id = self.omni_tokenizer.bos_token_id
        self._eos_id = self.omni_tokenizer.eos_token_id

        mapping = load_audio_token_mapping(self.omni_tokenizer_path)
        st = mapping["structure_tokens"]

        self._audio_start_id = st["audio_start"]
        self._audio_end_id = st["audio_end"]
        self._audio_token_offset = mapping["audio_token_offset"]

        # Store for subclass reuse (avoids re-reading the file)
        self._mapping = mapping

    @property
    def bos_id(self) -> int:
        if self._bos_id is None:
            self._cache_special_tokens()
        return self._bos_id

    @property
    def eos_id(self) -> int:
        if self._eos_id is None:
            self._cache_special_tokens()
        return self._eos_id

    @property
    def audio_start_id(self) -> int:
        if self._audio_start_id is None:
            self._cache_special_tokens()
        return self._audio_start_id

    @property
    def audio_end_id(self) -> int:
        if self._audio_end_id is None:
            self._cache_special_tokens()
        return self._audio_end_id

    @property
    def audio_token_offset(self) -> int:
        """Offset for audio tokens, determined from omni_tokenizer."""
        if self._audio_token_offset is None:
            self._cache_special_tokens()
        return self._audio_token_offset

    def tokenize(
        self,
        audio: Union[np.ndarray, torch.Tensor, bytes],
        sample_rate: int,
        text: Optional[str] = None,
    ) -> torch.Tensor:
        """Tokenize audio with encapsulation.

        Returns:
            Token tensor: [BOS, audio_start, audio_tokens..., audio_end, EOS]
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure batch dimension for single samples
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (T,) -> (1, T)

        # Encode (handles resampling internally)
        tokens, _ = self.wavtokenizer.encode(audio, sr=sample_rate)
        tokens = tokens.flatten()
        num_audio_tokens = tokens.numel()

        # Pre-allocate output tensor for efficiency (same pattern as vision)
        # Structure: BOS + audio_start + audio_tokens + audio_end + EOS
        total_size = 1 + 1 + num_audio_tokens + 1 + 1  # BOS + audio_start + tokens + audio_end + EOS
        device = tokens.device
        output = torch.empty(total_size, dtype=torch.long, device=device)

        # Fill tokens using slicing (no Python list operations)
        output[0] = self.bos_id
        output[1] = self.audio_start_id
        # Apply offset and copy audio tokens
        if num_audio_tokens:
            output[2:2 + num_audio_tokens] = tokens + self.audio_token_offset
        output[-2] = self.audio_end_id
        output[-1] = self.eos_id

        return output

    def tokenize_batch(
        self,
        audios: Union[list, np.ndarray, torch.Tensor],
        sample_rate: int,
        orig_audio_samples: Optional[list] = None,
        pad_audio_samples: Optional[int] = None,
    ) -> list:
        """Tokenize a batch of audio samples efficiently on GPU.

        All audio samples should have the same length for efficient batching.
        Use bucket filtering to ensure same-length samples.

        Args:
            audios: List/array/tensor of audio samples, all same length
            sample_rate: Sample rate of all audio samples
            orig_audio_samples: Original (unpadded) sample lengths per audio
            pad_audio_samples: Padding length in samples. Kept for API
                compatibility; trimming is based on each original length, not
                on whether that sample was the longest item in this batch.

        Returns:
            List of token tensors, each: [BOS, audio_start, audio_tokens..., audio_end, EOS]
        """
        if audios is None:
            return []

        if isinstance(audios, np.ndarray):
            if audios.size == 0:
                return []
            if audios.ndim == 1:
                audios = audios[None, :]
            batch_audio = torch.from_numpy(audios).float()
            if batch_audio.device != self.device:
                batch_audio = batch_audio.to(self.device, non_blocking=True)
            batch_size = batch_audio.shape[0]
        elif isinstance(audios, torch.Tensor):
            if audios.numel() == 0:
                return []
            batch_audio = audios
            if batch_audio.dim() == 1:
                batch_audio = batch_audio.unsqueeze(0)
            elif batch_audio.dim() == 3:
                batch_audio = batch_audio.squeeze(1)
            if batch_audio.device != self.device:
                batch_audio = batch_audio.to(self.device, non_blocking=True)
            batch_size = batch_audio.shape[0]
        else:
            if not audios:
                return []
            batch_size = len(audios)
            first = audios[0]
            # Stack numpy arrays first (fast CPU operation), then single transfer to GPU
            if isinstance(first, np.ndarray):
                stacked = np.stack(audios, axis=0)  # (B, T) numpy
                batch_audio = torch.from_numpy(stacked).float().to(self.device, non_blocking=True)
            else:
                # Already tensors - stack and move
                batch_audio = torch.stack([a if a.dim() == 1 else a.squeeze(0) for a in audios], dim=0)
                if batch_audio.device != self.device:
                    batch_audio = batch_audio.to(self.device, non_blocking=True)

        if orig_audio_samples is None:
            # Encode entire batch at once (handles resampling internally).
            batch_tokens, _ = self.wavtokenizer.encode(batch_audio, sr=sample_rate)
            # batch_tokens shape: (B, N) where N is number of audio tokens per sample
            device = batch_tokens.device
            num_audio_tokens = batch_tokens.shape[1]

            # Pre-allocate output for entire batch
            # Structure per sample: BOS + audio_start + audio_tokens + audio_end + EOS
            total_size = 1 + 1 + num_audio_tokens + 1 + 1
            batch_output = torch.empty(batch_size, total_size, dtype=torch.long, device=device)

            # Fill structure tokens (same for all samples)
            batch_output[:, 0] = self.bos_id
            batch_output[:, 1] = self.audio_start_id
            batch_output[:, -2] = self.audio_end_id
            batch_output[:, -1] = self.eos_id

            if num_audio_tokens:
                # Apply offset and copy audio tokens for all samples at once
                batch_output[:, 2:2 + num_audio_tokens] = (
                    batch_tokens[:, :num_audio_tokens] + self.audio_token_offset
                )

            # Return as list of tensors (one per sample)
            return [batch_output[i] for i in range(batch_size)]

        if len(orig_audio_samples) != batch_size:
            raise ValueError("orig_audio_samples length must match batch size")

        if pad_audio_samples is None and orig_audio_samples:
            pad_audio_samples = max(int(n) for n in orig_audio_samples)

        orig_array = np.asarray(orig_audio_samples, dtype=np.int64)
        ds = int(self.wavtokenizer.downsample_rate)

        # Encode entire batch at once (handles resampling internally).
        batch_tokens, _ = self.wavtokenizer.encode(batch_audio, sr=sample_rate)
        # batch_tokens shape: (B, N) where N is number of audio tokens per sample
        device = batch_tokens.device
        token_counts = (orig_array + ds - 1) // ds

        if self.trim_last_tokens:
            # This trim must be per-cut deterministic, not batch-dependent.
            # A cut can be the longest item in one local batch and padded in
            # another distributed run; conditioning on batch padding would make
            # cut_id -> tokens vary with world size and sampler locality.
            token_counts = np.maximum(token_counts - self.trim_last_tokens, 0)

        max_count = int(token_counts.max()) if token_counts.size else 0
        total_size = 1 + 1 + max_count + 1 + 1
        batch_output = torch.empty((batch_size, total_size), dtype=torch.long, device=device)
        batch_output[:, 0] = self.bos_id
        batch_output[:, 1] = self.audio_start_id

        if max_count:
            batch_output[:, 2:2 + max_count] = batch_tokens[:, :max_count] + self.audio_token_offset

        end_pos = torch.as_tensor(token_counts, device=device) + 2
        rows = torch.arange(batch_size, device=device)
        batch_output[rows, end_pos] = self.audio_end_id
        batch_output[rows, end_pos + 1] = self.eos_id

        outputs = []
        for i, count in enumerate(token_counts.tolist()):
            outputs.append(batch_output[i, : 1 + 1 + count + 1 + 1])

        return outputs

    def detokenize(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Decode token sequence(s) back to raw codec indices and audio.

        Works with audio_only and audio_text direct sequences. Finds
        audio_start/audio_end markers, extracts audio tokens between them,
        reverses the offset, and decodes to audio.

        Args:
            tokens: 1-D tensor (single sequence) or 2-D tensor (B, L) batch.

        Returns:
            (raw_codes, audio, info):
                raw_codes: raw codec indices (0-4095).
                           Single sequence -> 1-D tensor (N,).
                           Batch -> list of 1-D tensors (variable length).
                audio: reconstructed waveform from WavTokenizer40.decode().
                       Single sequence -> 1-D tensor (T,).
                       Batch -> 2-D tensor (B, T).
                info: dict with output_sample_rate, num_tokens.
        """
        single = tokens.dim() == 1
        if single:
            tokens = tokens.unsqueeze(0)  # (L,) -> (1, L)

        batch_size = tokens.shape[0]
        raw_codes_list = []

        for i in range(batch_size):
            seq = tokens[i]

            # Find audio_start and audio_end positions
            start_mask = (seq == self.audio_start_id).nonzero(as_tuple=False)
            end_mask = (seq == self.audio_end_id).nonzero(as_tuple=False)

            if start_mask.numel() == 0 or end_mask.numel() == 0:
                raise ValueError(
                    f"Sequence {i}: could not find audio_start ({self.audio_start_id}) "
                    f"and/or audio_end ({self.audio_end_id}) markers."
                )

            start_pos = start_mask[0].item()
            end_pos = end_mask[0].item()

            # Extract audio tokens between markers and reverse offset
            audio_tokens = seq[start_pos + 1 : end_pos] - self.audio_token_offset
            raw_codes_list.append(audio_tokens)

        # Pad to uniform length for batch decode
        max_len = max(codes.numel() for codes in raw_codes_list)
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=tokens.device)
        for i, codes in enumerate(raw_codes_list):
            padded[i, : codes.numel()] = codes

        # Decode with WavTokenizer
        audio, dec_info = self.wavtokenizer.decode(padded)

        info = {
            "output_sample_rate": dec_info["output_sample_rate"],
            "num_tokens": [codes.numel() for codes in raw_codes_list],
        }

        if single:
            return raw_codes_list[0], audio.squeeze(0), info

        return raw_codes_list, audio, info
