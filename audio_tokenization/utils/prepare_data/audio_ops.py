"""Audio processing helpers for prepare_data."""

from __future__ import annotations

import math
from collections import Counter

from audio_tokenization.utils.prepare_data.constants import MIN_RMS_DB


def _ffmpeg_cli_to_wav(audio_bytes: bytes, recording_id: str) -> bytes:
    """Transcode audio bytes to WAV via ffmpeg CLI subprocess."""
    import subprocess

    result = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-f", "wav", "-acodec", "pcm_s16le", "pipe:1"],
        input=audio_bytes,
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg CLI decode failed for {recording_id}: "
            f"{result.stderr.decode(errors='replace')[:200]}"
        )
    if not result.stdout:
        raise RuntimeError(f"ffmpeg CLI produced empty output for {recording_id}")
    return result.stdout


def build_recording_from_audio_bytes(
    audio_bytes: bytes,
    recording_id: str,
    *,
    runtime_counts: Counter | None = None,
):
    """Create a Lhotse Recording from encoded audio bytes."""
    from lhotse import Recording

    recording_id = str(recording_id)
    if runtime_counts is not None:
        runtime_counts["recording_from_bytes"] += 1

    try:
        return Recording.from_bytes(data=audio_bytes, recording_id=recording_id)
    except (ValueError, RuntimeError, OSError):
        pass

    if runtime_counts is not None:
        runtime_counts["ffmpeg_cli_fallback"] += 1
    wav_bytes = _ffmpeg_cli_to_wav(audio_bytes, recording_id)
    return Recording.from_bytes(data=wav_bytes, recording_id=recording_id)


def normalize_batch_peak(audios, target_db: float = -3.0):
    """Peak-normalize each sample in a padded batch (vectorized, no Python loop)."""
    target_peak = 10 ** (target_db / 20.0)
    peaks = audios.abs().max(dim=1).values.clamp(min=1e-10)
    scale = target_peak / peaks
    return audios * scale.unsqueeze(1)


def rms_db(cut) -> float:
    """Compute RMS level in dB for a cut's audio."""
    import numpy as np

    audio = cut.load_audio()
    if audio.size == 0:
        return -200.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20.0 * np.log10(rms + 1e-10)


def should_skip_quiet(rms_val: float) -> bool:
    """Return True if the sample should be skipped (silent/empty audio)."""
    return math.isnan(rms_val) or rms_val < MIN_RMS_DB


def make_rms_filter_fn():
    """Return a (map_fn, filter_fn) pair for computing rms_db and filtering quiet cuts."""

    def _compute_rms(cut):
        cut.custom = cut.custom or {}
        cut.custom["rms_db"] = rms_db(cut)
        return cut

    def _keep_loud(cut) -> bool:
        val = (cut.custom or {}).get("rms_db", 0.0)
        return not should_skip_quiet(val)

    return _compute_rms, _keep_loud


def apply_audio_pipeline(
    cut,
    *,
    target_sr: int | None = None,
    mono_downmix: bool = True,
    tokenize_fn=None,
    runtime_counts: Counter,
):
    """Resample -> mono -> tokenize text -> RMS filter in one call."""
    if target_sr and cut.sampling_rate != target_sr:
        cut = cut.resample(target_sr)
        runtime_counts["resampled"] += 1

    cut = to_mono(cut, mono_downmix=mono_downmix, stats=runtime_counts)

    if tokenize_fn is not None:
        cut = tokenize_fn(cut)

    cut.custom = cut.custom or {}
    rms_val = rms_db(cut)
    if should_skip_quiet(rms_val):
        runtime_counts["skipped_quiet_audio"] += 1
        return cut, True
    cut.custom["rms_db"] = rms_val

    return cut, False


def to_mono(cut, mono_downmix=True, stats=None):
    """Convert a multi-channel cut to mono."""
    if cut.num_channels <= 1:
        return cut
    if mono_downmix:
        try:
            result = cut.to_mono(mono_downmix=True)
            result.load_audio()
            return result
        except Exception:
            if stats is not None:
                stats["downmix_fallback_ch0"] += 1
    result = cut.to_mono(mono_downmix=False)
    if isinstance(result, list):
        return result[0]
    return result

