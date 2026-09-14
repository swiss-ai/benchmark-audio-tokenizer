"""Decode reuse must preserve Lhotse's channel transforms and cut semantics."""

import copy
import weakref
from collections import Counter

import numpy as np
import pytest
from lhotse import SupervisionSegment, fastcopy
from lhotse.audio.source import AudioSource
from lhotse.augmentation import Resample

from audio_tokenization.prepare import audio_ops
from audio_tokenization.tests.test_prepare_integrity import _cut


@pytest.mark.parametrize("channels", [[0, 1], [3, 1, 0]])
def test_downmix_decodes_once_keeps_channel_resampling_and_releases_source(monkeypatch, channels):
    cut = _cut(max(channels) + 1)
    cut.channel = channels
    cut.supervisions = [SupervisionSegment(
        id="words", recording_id=cut.recording.id, start=0, duration=cut.duration,
        channel=channels, text="some words",
    )]
    cut.custom = {"source": "regression"}
    cut = cut.truncate(offset=0.003, duration=0.032).resample(24000)
    expected = copy.deepcopy(cut).to_mono(mono_downmix=True)
    expected_audio = expected.load_audio()
    before = copy.deepcopy(cut.to_dict())
    original_load = AudioSource.load_audio
    original_resample = Resample.__call__
    decoded = []
    shapes = []

    def load(source, *args, **kwargs):
        audio = original_load(source, *args, **kwargs)
        if source.source == cut.recording.sources[0].source:
            decoded.append(weakref.ref(audio))
        return audio

    def resample(transform, audio, *args, **kwargs):
        shapes.append(audio.shape[0])
        return original_resample(transform, audio, *args, **kwargs)

    monkeypatch.setattr(AudioSource, "load_audio", load)
    monkeypatch.setattr(Resample, "__call__", resample)
    for repeat in range(2):
        result, audio = audio_ops._to_mono_with_audio(cut)
        np.testing.assert_array_equal(audio, expected_audio)
        assert result.to_dict() == expected.to_dict()
        assert cut.to_dict() == before
        assert len(decoded) == repeat + 1
        assert all(ref() is None for ref in decoded)
        assert shapes == [1] * (len(channels) * (repeat + 1))
        assert AudioSource.load_audio is load
        assert "load_audio" not in vars(cut.recording.sources[0])


def test_decode_failure_retries_original_first_channel_without_retaining_cache(monkeypatch):
    cut = _cut(2)
    expected = cut.to_mono(mono_downmix=False)[0].load_audio()
    original = AudioSource.load_audio
    attempts = 0

    def load(source, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("injected decoder failure")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(AudioSource, "load_audio", load)
    counts = Counter()
    result, skip, audio = audio_ops.apply_audio_pipeline(cut, runtime_counts=counts)
    assert not skip
    np.testing.assert_array_equal(audio, expected)
    assert result.recording is cut.recording
    assert counts["downmix_fallback_ch0"] == 1
    assert attempts == 2


@pytest.mark.parametrize("kind", ["mono", "first_channel", "multiple_sources"])
def test_other_cut_paths_keep_their_original_behavior(kind):
    cut = _cut(1 if kind == "mono" else 2)
    if kind == "multiple_sources":
        source = _cut().recording.sources[0]
        cut.recording = fastcopy(cut.recording, sources=[source, fastcopy(source, channels=[1])])
    if kind in {"mono", "first_channel"}:
        expected = cut if kind == "mono" else cut.to_mono(mono_downmix=False)[0]
    else:
        expected = cut.to_mono(mono_downmix=True)
    result, audio = audio_ops._to_mono_with_audio(cut, mono_downmix=kind != "first_channel")
    assert result.to_dict() == expected.to_dict()
    np.testing.assert_array_equal(result.load_audio() if audio is None else audio, expected.load_audio())
