from types import SimpleNamespace

import pytest

from audio_tokenization.pipelines.lhotse.audio_text import (
    AudioTextHandler,
    resolve_partition_value,
    resolve_interleaving_metadata,
)


def _make_cut(*, cut_id="clip", start=0.0, custom=None):
    return SimpleNamespace(id=cut_id, start=start, custom=custom)


def test_resolve_interleaving_metadata_reads_canonical_fields():
    cut = _make_cut(
        cut_id="legacy_00005",
        start=9.0,
        custom={
            "source_id": "session_a",
            "clip_num": 7,
            "clip_start": 1.25,
        },
    )

    source_id, clip_num, clip_start = resolve_interleaving_metadata(cut)

    assert source_id == "session_a"
    assert clip_num == 7
    assert clip_start == 1.25


def test_resolve_interleaving_metadata_falls_back_to_cut_start():
    cut = _make_cut(
        cut_id="clip_0001",
        start=3.5,
        custom={"source_id": "session_b", "clip_num": 1},
    )

    source_id, clip_num, clip_start = resolve_interleaving_metadata(cut)

    assert source_id == "session_b"
    assert clip_num == 1
    assert clip_start == 3.5


def test_resolve_interleaving_metadata_requires_canonical_fields():
    cut = _make_cut(cut_id="legacy_5", start=0.0, custom=None)

    with pytest.raises(ValueError, match="missing interleaving metadata"):
        resolve_interleaving_metadata(cut)


def test_audio_text_handler_rejects_unknown_cache_layout_version():
    with pytest.raises(ValueError, match="cache_layout_version"):
        AudioTextHandler({
            "audio_text_format": "interleaved",
            "audio_text_task": "transcribe",
            "cache_layout_version": "v3",
        })


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

    handler = AudioTextHandler({
        "audio_text_format": "interleaved",
        "audio_text_task": "transcribe",
        "cache_layout_version": "v2",
    })
    handler._setup_writer_interleaved(str(tmp_path), 3, 11)

    assert created["args"] == (
        str(tmp_path),
        3,
        11,
        {"type": "hash", "field": "source_id", "num_buckets": 16},
    )


def test_audio_text_handler_selects_v1_writer_by_default(monkeypatch, tmp_path):
    created = {}

    class FakeWriter:
        def __init__(self, output_dir, rank, chunk_id):
            created["args"] = (output_dir, rank, chunk_id)

    monkeypatch.setattr(
        "audio_tokenization.pipelines.shard_io.ParquetChunkWriter",
        FakeWriter,
    )

    handler = AudioTextHandler({
        "audio_text_format": "interleaved",
        "audio_text_task": "transcribe",
    })
    handler._setup_writer_interleaved(str(tmp_path), 1, 2)

    assert created["args"] == (str(tmp_path), 1, 2)


def test_resolve_partition_value_prefers_custom_then_supervision():
    supervision = SimpleNamespace(language="de", speaker="sup-spk")
    cut = SimpleNamespace(
        id="cut",
        custom={"language": "fr"},
        supervisions=[supervision],
        speaker="cut-spk",
    )

    assert resolve_partition_value(cut, "language") == "fr"
    assert resolve_partition_value(cut, "speaker") == "sup-spk"


def test_resolve_partition_value_reads_supervision_field():
    supervision = SimpleNamespace(language="it")
    cut = SimpleNamespace(id="cut", custom={}, supervisions=[supervision])

    assert resolve_partition_value(cut, "language") == "it"
