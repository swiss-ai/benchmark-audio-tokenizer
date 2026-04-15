from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_audio_inference_module(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    torchaudio = types.ModuleType("torchaudio")
    torchaudio.transforms = types.SimpleNamespace(Resample=object)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)

    wavtokenizer = types.ModuleType("src.audio_tokenizers.implementations.wavtokenizer")
    wavtokenizer.WavTokenizer40 = object
    monkeypatch.setitem(
        sys.modules,
        "src.audio_tokenizers.implementations.wavtokenizer",
        wavtokenizer,
    )

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "audio_inference.py"
    spec = importlib.util.spec_from_file_location("audio_inference_test_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_downmix_audio_array_channels_first_preserves_time_axis(monkeypatch):
    module = _load_audio_inference_module(monkeypatch)
    stereo = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32)

    mono = module._downmix_audio_array(stereo, channel_layout="channels_first")

    np.testing.assert_allclose(mono, np.array([1.5, 3.5, 5.5], dtype=np.float32))


def test_downmix_audio_array_channels_last_preserves_time_axis(monkeypatch):
    module = _load_audio_inference_module(monkeypatch)
    stereo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    mono = module._downmix_audio_array(stereo, channel_layout="channels_last")

    np.testing.assert_allclose(mono, np.array([1.5, 3.5, 5.5], dtype=np.float32))
