from types import SimpleNamespace

import pytest
import torch

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.pipelines.lhotse.checkpoint import WorkerStats
from audio_tokenization.pipelines.lhotse.audio_text import (
    AudioTextHandler,
    resolve_interleaving_metadata,
)


def _make_cut(*, cut_id="clip", start=0.0, duration=1.0, custom=None):
    return SimpleNamespace(id=cut_id, start=start, duration=duration, custom=custom)


class _FakeBuilder:
    def __init__(self):
        self.items = []
        self.documents = 0

    def add_item(self, tensor):
        self.items.append(tensor.clone())

    def end_document(self):
        self.documents += 1


class _FakeCutIds:
    def __init__(self):
        self.ids = []

    def write(self, cut_id):
        self.ids.append(cut_id)


class _FakeDirectTokenizer:
    bos_id = 1
    eos_id = 2
    speech_transcribe_id = 3

    def tokenize_batch_raw(self, audios, target_sr, orig_audio_samples, pad_audio_samples):
        assert audios.shape == (2, 16)
        assert target_sr == 24_000
        assert orig_audio_samples == [12, 16]
        assert pad_audio_samples == 16
        return [[10, 11], [20, 21, 22]]


class _FakeInterleavedTokenizer:
    def tokenize_batch_raw(self, audios, target_sr, orig_audio_samples, pad_audio_samples):
        assert audios.shape == (1, 16)
        assert target_sr == 24_000
        assert orig_audio_samples == [16]
        assert pad_audio_samples == 16
        return [[100, 101, 102]]


def _tokenize_spec(**overrides):
    payload = {
        "tokenizer": {"path": "/tmp/tokenizer"},
        "output": {"output_dir": "/tmp/out"},
        "mode": "audio_text",
        "audio_text_format": "interleaved",
        "audio_text_task": "transcribe",
    }
    payload.update(overrides)
    return TokenizeSpec.model_validate(payload)


def test_resolve_interleaving_metadata_reads_canonical_fields():
    cut = _make_cut(
        cut_id="clip_00005",
        start=9.0,
        custom={
            "interleave": {
                "source_id": "session_a",
                "clip_num": 7,
                "clip_start": 1.25,
                "clip_duration": 1.5,
            },
        },
    )

    source_id, clip_num, clip_start, clip_duration = resolve_interleaving_metadata(cut)

    assert source_id == "session_a"
    assert clip_num == 7
    assert clip_start == 1.25
    assert clip_duration == 1.5


def test_resolve_interleaving_metadata_leaves_timestamps_absent_when_not_provided():
    cut = _make_cut(
        cut_id="clip_0001",
        start=3.5,
        duration=2.25,
        custom={"interleave": {"source_id": "session_b", "clip_num": 1}},
    )

    source_id, clip_num, clip_start, clip_duration = resolve_interleaving_metadata(cut)

    assert source_id == "session_b"
    assert clip_num == 1
    assert clip_start is None
    assert clip_duration is None


def test_resolve_interleaving_metadata_requires_canonical_fields():
    cut = _make_cut(cut_id="clip_5", start=0.0, custom=None)

    with pytest.raises(ValueError, match="missing interleaving metadata"):
        resolve_interleaving_metadata(cut)


def test_audio_text_schema_rejects_removed_cache_layout_version():
    with pytest.raises(ValueError, match="cache_layout_version"):
        _tokenize_spec(cache_layout_version="v1")


def test_audio_text_handler_selects_v2_writer(monkeypatch, tmp_path):
    created = {}

    class FakeWriter:
        @staticmethod
        def _normalize_partitioning(partitioning):
            return partitioning or {"type": "hash", "field": "source_id", "num_buckets": 16}

        def __init__(self, output_dir, rank, writer_state, partitioning):
            created["args"] = (output_dir, rank, writer_state, partitioning)

    monkeypatch.setattr(
        "audio_tokenization.pipelines.shard_io.StructuredCacheChunkWriter",
        FakeWriter,
    )

    handler = AudioTextHandler(_tokenize_spec(), dataset_name="test_dataset")
    handler._setup_writer_interleaved(str(tmp_path), 3, 11)

    assert created["args"] == (
        str(tmp_path),
        3,
        11,
        {"type": "hash", "field": "source_id", "num_buckets": 16},
    )


def test_audio_text_direct_writes_cut_ids_in_document_order():
    handler = AudioTextHandler(
        _tokenize_spec(audio_text_format="direct"),
        dataset_name="test_dataset",
    )
    handler._builder = _FakeBuilder()
    handler._cut_ids = _FakeCutIds()
    stats = WorkerStats()

    cuts = [
        SimpleNamespace(id="cut-a", num_samples=12, custom={"text_tokens": [101]}),
        SimpleNamespace(id="cut-b", num_samples=16, custom={"text_tokens": [201, 202]}),
    ]
    handler._process_batch_direct(
        {
            "inputs": torch.zeros(2, 16),
            "supervisions": {"cut": cuts},
        },
        _FakeDirectTokenizer(),
        stats,
        target_sr=24_000,
        device="cpu",
    )

    assert handler._cut_ids.ids == ["cut-a", "cut-b"]
    assert handler._builder.documents == 2
    assert [item.tolist() for item in handler._builder.items] == [
        [1, 10, 11, 3, 101, 2],
        [1, 20, 21, 22, 3, 201, 202, 2],
    ]
    assert stats.samples_processed == 2
    assert stats.tokens_generated == 5
    assert stats.text_tokens_generated == 3


def test_audio_text_interleaved_field_partitioning_reads_generated_row_fields():
    handler = AudioTextHandler(
        _tokenize_spec(partitioning={"type": "field", "field": "source_id"}),
        dataset_name="test_dataset",
    )
    captured = {}

    class FakeWriter:
        def add_rows(self, rows):
            captured["rows"] = rows

    handler._writer = FakeWriter()
    stats = WorkerStats()
    supervision = SimpleNamespace(text="hello", speaker="spk")
    cut = SimpleNamespace(
        id="clip-a",
        num_samples=16,
        duration=1.0,
        custom={
            "text_tokens": [301],
            "interleave": {"source_id": "session-a", "clip_num": 3},
        },
        supervisions=[supervision],
    )

    handler._process_batch_interleaved(
        {
            "inputs": torch.zeros(1, 16),
            "supervisions": {"cut": [cut]},
        },
        _FakeInterleavedTokenizer(),
        stats,
        target_sr=24_000,
        device="cpu",
    )

    assert captured["rows"][0]["source_id"] == "session-a"
    assert captured["rows"][0]["_partition_value"] == "session-a"
    assert stats.samples_processed == 1


def test_audio_text_interleaved_field_partitioning_requires_cache_column():
    handler = AudioTextHandler(
        _tokenize_spec(partitioning={"type": "field", "field": "language"}),
        dataset_name="test_dataset",
    )
    handler._writer = SimpleNamespace(add_rows=lambda _rows: None)
    stats = WorkerStats()
    supervision = SimpleNamespace(text="hello", speaker="spk", language="en")
    cut = SimpleNamespace(
        id="clip-a",
        num_samples=16,
        duration=1.0,
        custom={
            "text_tokens": [301],
            "interleave": {"source_id": "session-a", "clip_num": 3},
        },
        supervisions=[supervision],
    )

    with pytest.raises(ValueError, match="generated cache column 'language'"):
        handler._process_batch_interleaved(
            {
                "inputs": torch.zeros(1, 16),
                "supervisions": {"cut": [cut]},
            },
            _FakeInterleavedTokenizer(),
            stats,
            target_sr=24_000,
            device="cpu",
        )
