"""WavTokenizer implementations."""

from .audio_only import WavTokenizerAudioOnly
from .audio_text import WavTokenizerAudioText

_SUPPORTED_MODES = ("audio_only", "audio_text", "audio_cache")


def create_tokenizer(
    omni_tokenizer_path: str,
    mode: str = "audio_only",
    device: str = "cuda",
    **kwargs,
):
    """Factory function to create WavTokenizer.

    Args:
        omni_tokenizer_path: Path to omni_tokenizer (must have audio tokens added)
        mode: ``"audio_only"``, ``"audio_text"``, or ``"audio_cache"``
        device: Device for tokenization

    Note:
        The audio token offset is automatically determined from the omni_tokenizer
        by querying: convert_tokens_to_ids("<|audio token 0|>")
    """
    if mode in ("audio_only", "audio_cache"):
        return WavTokenizerAudioOnly(
            omni_tokenizer_path=omni_tokenizer_path,
            device=device,
            **kwargs,
        )
    elif mode == "audio_text":
        return WavTokenizerAudioText(
            omni_tokenizer_path=omni_tokenizer_path,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Expected one of: {_SUPPORTED_MODES}")
