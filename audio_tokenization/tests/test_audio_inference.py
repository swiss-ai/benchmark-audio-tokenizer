from __future__ import annotations

import importlib.util
import io
import sys
import types
import wave
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


def _make_wav_bytes(sr: int = 16000, num_samples: int = 160) -> bytes:
    buf = io.BytesIO()
    samples = np.zeros(num_samples, dtype=np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


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


def test_load_parquet_dataset_empty_dir_returns_no_samples(monkeypatch, tmp_path):
    module = _load_audio_inference_module(monkeypatch)

    samples = module.load_parquet_dataset(str(tmp_path), "audio", num_samples=3)

    assert samples == []


def test_load_parquet_dataset_recomputes_optional_columns_per_file(monkeypatch, tmp_path):
    module = _load_audio_inference_module(monkeypatch)

    _write_parquet(
        tmp_path / "shard_00000.parquet",
        [{"audio": {"bytes": b"", "sampling_rate": 16000}}],
    )
    _write_parquet(
        tmp_path / "shard_00001.parquet",
        [
            {
                "audio": {
                    "bytes": _make_wav_bytes(),
                    "sampling_rate": 16000,
                },
                "text": "hello later shard",
                "id": "sample-late",
            }
        ],
    )

    samples = module.load_parquet_dataset(str(tmp_path), "audio", num_samples=1)

    assert len(samples) == 1
    assert samples[0]["text"] == "hello later shard"
    assert samples[0]["id"] == "sample-late"
