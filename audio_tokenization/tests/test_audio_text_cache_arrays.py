"""Check token buffers and persisted cache output without loading a GPU model."""

import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.pipelines.lhotse.audio_text import AudioTextHandler
from audio_tokenization.pipelines.lhotse.checkpoint import WorkerStats
from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter
from audio_tokenization.vokenizers.wavtokenizer.audio_text import WavTokenizerAudioText


def _tokenizer(monkeypatch, tensors):
    tokenizer = WavTokenizerAudioText(omni_tokenizer_path="unused", device="cpu")
    # Only replace model tokenization; raw packing, handlers and writers remain real.
    monkeypatch.setattr(tokenizer, "tokenize_batch", lambda *a, **kw: tensors)
    return tokenizer


def test_raw_token_buffers_preserve_ragged_boundaries_and_lifetime(monkeypatch):
    tensors = [torch.tensor([1, 10, 70000, 2147483647, 11, 2]),
               torch.tensor([1, 10, 11, 2]), torch.tensor([1, 2])]
    tokenizer = _tokenizer(monkeypatch, tensors)
    raw = tokenizer.tokenize_batch_raw(None, 24000)
    # A buffer consumer must see only the interior tokens, including an empty span.
    assert [np.frombuffer(memoryview(span), dtype=np.int64).tolist() for span in raw] == [
        [10, 70000, 2147483647, 11], [10, 11], [],
    ]
    saved = raw[1]
    del raw, tokenizer, tensors
    gc.collect()
    assert saved.tolist() == [10, 11]


def test_raw_token_empty_batch(monkeypatch):
    assert _tokenizer(monkeypatch, []).tokenize_batch_raw(None, 24000) == []


def test_raw_tokens_flow_through_interleaved_handler(monkeypatch, tmp_path):
    tokenizer = _tokenizer(monkeypatch, [torch.tensor([1, 10, 70000, 11, 2]),
                                       torch.tensor([1, 10, 11, 2])])
    spec = TokenizeSpec.model_validate({
        "tokenizer": {"path": "unused"}, "output": {"output_dir": str(tmp_path)},
        "mode": "audio_text", "audio_text_format": "interleaved",
        "partitioning": {"type": "field", "field": "source_id"},
    })
    handler = AudioTextHandler(spec, dataset_name="ds")
    handler.setup_writer(str(tmp_path), 0, 0, tokenizer)
    cuts = [SimpleNamespace(id=f"clip-{i}", num_samples=16, duration=1.0,
                            custom={"text_tokens": text, "interleave": {
                                "source_id": "a", "clip_num": i}},
                            supervisions=[SimpleNamespace(text="hello", speaker="")])
            for i, text in enumerate([[101, 102], []])]
    stats = WorkerStats()
    handler.process_batch({"inputs": torch.zeros(2, 16), "supervisions": {"cut": cuts}},
                          tokenizer, stats, 24000, "cpu")
    handler.finalize_writer()
    rank = tmp_path / "source_id=a/rank_0000"
    assert np.fromfile(rank / "audio_tokens.000000.bin", dtype=np.int32).tolist() == [
        10, 70000, 11, 10, 11,
    ]
    assert np.fromfile(rank / "text_tokens.000000.bin", dtype=np.int32).tolist() == [101, 102]
    rows = pq.read_table(rank / "clips.000000.parquet").to_pylist()
    assert [(r["clip_id"], r["audio_token_offset"], r["audio_token_length"],
             r["text_token_offset"], r["text_token_length"]) for r in rows] == [
        ("clip-0", 0, 3, 0, 2), ("clip-1", 12, 2, 8, 0),
    ]
    assert (stats.samples_processed, stats.tokens_generated, stats.text_tokens_generated) == (2, 5, 2)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_cache_buffers_preserve_bytes_offsets_and_release_inputs(tmp_path, dtype, strided):
    writer = StructuredCacheChunkWriter(str(tmp_path), rank=0,
                                        partitioning={"type": "field", "field": "source_id"})
    audio = np.array([70000, 9, 70001, 8, 70002, 7], dtype=dtype)
    text = np.array([101, 102, 103], dtype=dtype)
    if strided:
        audio, text = audio[::2], text[::-1]
    want_audio = [70000, 70001, 70002] if strided else [70000, 9, 70001, 8, 70002, 7]
    want_text = [103, 102, 101] if strided else [101, 102, 103]
    refs = [weakref.ref(audio), weakref.ref(text)]
    metadata = {"clip_id": "a0", "source_id": "a", "clip_num": 0, "clip_start": None,
                "speaker": "", "duration": 1.0, "text": "", "dataset": "ds"}
    writer.add_rows([{**metadata, "audio_tokens": audio, "text_tokens": text}])
    # The writer must consume values synchronously and retain metadata only.
    audio[:] = 0
    text[:] = 0
    del audio, text
    gc.collect()
    assert all(ref() is None for ref in refs)
    writer.add_rows([{**metadata, "clip_id": "a1", "clip_num": 1,
                      "audio_tokens": np.array([], dtype=dtype), "text_tokens": []}])
    writer.finalize()
    rank = tmp_path / "source_id=a/rank_0000"
    assert (rank / "audio_tokens.000000.bin").read_bytes() == np.array(want_audio, dtype=np.int32).tobytes()
    assert (rank / "text_tokens.000000.bin").read_bytes() == np.array(want_text, dtype=np.int32).tobytes()
    rows = pq.read_table(rank / "clips.000000.parquet").to_pylist()
    assert [(r["audio_token_offset"], r["audio_token_length"],
             r["text_token_offset"], r["text_token_length"]) for r in rows] == [
        (0, len(want_audio), 0, 3), (4 * len(want_audio), 0, 12, 0),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Checks GPU input lifetime")
@pytest.mark.parametrize("output_format", ["direct", "interleaved"])
@pytest.mark.parametrize("normalized", [False, True])
def test_handler_device_audio_lifetime_during_writes(monkeypatch, tmp_path, output_format, normalized):
    """Release owned device copies while preserving caller-owned normalized input."""
    audio_refs = []
    calls = []

    def encode(audios, sample_rate, *, orig_audio_samples, pad_audio_samples):
        assert audios.is_cuda
        assert sample_rate == 24000
        assert orig_audio_samples == [12, 16]
        assert pad_audio_samples == 16
        audio_refs.append(weakref.ref(audios))
        return [np.array([10, 70000, 11]), np.array([10, 11])]

    tokenizer = SimpleNamespace(tokenize_batch_raw=encode, omni_tokenizer=range(100000),
                                bos_id=1, eos_id=2, speech_transcribe_id=3)
    spec = TokenizeSpec.model_validate({
        "tokenizer": {"path": "unused"}, "output": {"output_dir": str(tmp_path)},
        "mode": "audio_text", "audio_text_format": output_format,
        "partitioning": {"type": "field", "field": "source_id"},
    })
    handler = AudioTextHandler(spec, dataset_name="ds")
    handler.setup_writer(str(tmp_path), 0, 0, tokenizer)
    writer = handler._builder if output_format == "direct" else handler._writer
    method = "add_item" if output_format == "direct" else "add_rows"
    write = getattr(writer, method)

    def check_lifetime(*args, **kwargs):
        assert audio_refs
        if normalized:
            assert audio_refs[-1]() is batch["inputs"]
        else:
            assert audio_refs[-1]() is None
        calls.append(True)
        return write(*args, **kwargs)

    monkeypatch.setattr(writer, method, check_lifetime)
    cuts = [SimpleNamespace(id=f"clip-{i}", num_samples=length, duration=length / 24000,
                            custom={"interleave": {"source_id": "a", "clip_num": i}},
                            supervisions=[]) for i, length in enumerate([12, 16])]
    batch = {"inputs": torch.ones(2, 16), "supervisions": {"cut": cuts}}
    if normalized:
        from audio_tokenization.pipelines.lhotse.core import _normalize_batch
        batch = _normalize_batch(batch, -3.0, "cuda")
    stats = WorkerStats()
    try:
        seconds = handler.process_batch(
            batch,
            tokenizer, stats, 24000, "cuda",
        )
        handler.finalize_writer()
    finally:
        handler.abort_writer()
    assert calls
    assert seconds == pytest.approx(28 / 24000)
    assert (stats.samples_processed, stats.tokens_generated, stats.text_tokens_generated) == (2, 5, 0)
