"""Audio processing helpers for prepare scripts."""

from __future__ import annotations

import math
from collections import Counter
from contextlib import contextmanager
import os

from audio_tokenization.prepare.constants import MIN_RMS_DB


@contextmanager
def _suppress_stderr_fd():
    """Temporarily silence C libraries that write directly to stderr fd 2."""
    saved_fd = os.dup(2)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def load_audio_quietly(cut, *args, **kwargs):
    """Load audio while suppressing noisy C decoder stderr output."""
    with _suppress_stderr_fd():
        return cut.load_audio(*args, **kwargs)


def extract_audio_bytes(value):
    """Accept HF audio structs and plain binary audio columns."""
    if isinstance(value, dict):
        return value.get("bytes")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return None


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
        with _suppress_stderr_fd():
            return Recording.from_bytes(data=audio_bytes, recording_id=recording_id)
    except (ValueError, RuntimeError, OSError):
        pass

    if runtime_counts is not None:
        runtime_counts["ffmpeg_cli_fallback"] += 1
    wav_bytes = _ffmpeg_cli_to_wav(audio_bytes, recording_id)
    with _suppress_stderr_fd():
        return Recording.from_bytes(data=wav_bytes, recording_id=recording_id)


def normalize_batch_peak(audios, target_db: float = -3.0):
    """Peak-normalize each sample in a padded batch (vectorized, no Python loop)."""
    target_peak = 10 ** (target_db / 20.0)
    peaks = audios.abs().max(dim=1).values.clamp(min=1e-10)
    scale = target_peak / peaks
    return audios * scale.unsqueeze(1)


def rms_db(cut) -> float:
    """Compute RMS level in dB for a cut's audio."""
    audio = load_audio_quietly(cut)
    return rms_db_from_audio(audio)


def rms_db_from_audio(audio) -> float:
    """Compute RMS level in dB from an already-decoded waveform."""
    import numpy as np

    if audio.size == 0:
        return -200.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20.0 * np.log10(rms + 1e-10)


def write_cut_to_shar(writer, cut, *, audio=None, runtime_counts: Counter | None = None) -> None:
    """Write a cut to Lhotse SHAR, reusing decoded audio when available.

    Lhotse's ``SharWriter.write(cut)`` calls ``cut.load_audio()`` internally.
    Conversion has already loaded the standardized waveform to compute RMS, so
    this mirrors Lhotse's recording write path and then writes the cut manifest.
    When the writer is a test double or an unsupported shape, it falls back to
    the public Lhotse API.
    """
    writer_fields = set(getattr(writer, "fields", {}) or {})
    if (
        audio is None
        or not getattr(cut, "has_recording", False)
        or writer_fields != {"recording"}
        or not hasattr(writer, "writers")
        or "recording" not in writer.writers
        or "cuts" not in writer.writers
    ):
        writer.write(cut)
        return

    try:
        from lhotse import fastcopy
        from lhotse.shar.utils import to_shar_placeholder
        from lhotse.shar.writers.shar import _aslist

        recording = to_shar_placeholder(cut.recording, cut)
        cut_channels = _aslist(cut.channel)
        if recording.channel_ids != cut_channels:
            recording.sources[0].channels = cut_channels
            recording.channel_ids = cut_channels
        writer.writers["recording"].write(
            cut.id,
            audio,
            cut.sampling_rate,
            manifest=recording,
            original_format=cut.recording.source_format,
        )
        cut = fastcopy(cut, recording=recording, start=0)
        writer.writers["cuts"].write(cut)
        if runtime_counts is not None:
            runtime_counts["reused_decoded_audio_for_shar_write"] += 1
    except Exception:
        if runtime_counts is not None:
            runtime_counts["decoded_audio_write_fallback"] += 1
        writer.write(cut)


def below_rms_threshold(rms_val: float, threshold: float) -> bool:
    """Return True when an RMS value is invalid or below a threshold."""
    return math.isnan(rms_val) or rms_val < threshold


def should_skip_quiet(rms_val: float) -> bool:
    """Return True if the sample should be skipped (silent/empty audio)."""
    return below_rms_threshold(rms_val, MIN_RMS_DB)


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
    """Resample -> mono -> tokenize text -> RMS filter in one call.

    Returns ``(cut, skip, decoded_audio)``. ``decoded_audio`` is the
    materialized waveform when the cut passes the quiet-audio filter, ``None``
    when ``skip`` is ``True``. Callers reuse the waveform to avoid a second
    decode at SHAR write time.
    """
    if target_sr and cut.sampling_rate != target_sr:
        cut = cut.resample(target_sr)
        runtime_counts["resampled"] += 1

    cut = to_mono(cut, mono_downmix=mono_downmix, stats=runtime_counts)

    if tokenize_fn is not None:
        cut = tokenize_fn(cut)

    cut.custom = cut.custom or {}
    audio = load_audio_quietly(cut)
    rms_val = rms_db_from_audio(audio)
    if should_skip_quiet(rms_val):
        runtime_counts["skipped_quiet_audio"] += 1
        return cut, True, None
    cut.custom["rms_db"] = rms_val

    return cut, False, audio


def to_mono(cut, mono_downmix=True, stats=None):
    """Convert a multi-channel cut to mono."""
    if cut.num_channels <= 1:
        return cut
    if mono_downmix:
        try:
            result = cut.to_mono(mono_downmix=True)
            load_audio_quietly(result)
            return result
        except Exception:
            if stats is not None:
                stats["downmix_fallback_ch0"] += 1
    result = cut.to_mono(mono_downmix=False)
    if isinstance(result, list):
        return result[0]
    return result
