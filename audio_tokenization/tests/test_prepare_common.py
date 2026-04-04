import sys
import types
from collections import Counter
import gzip

from audio_tokenization.utils.prepare_data import common


def test_build_recording_from_audio_bytes_uses_raw_bytes(monkeypatch):
    calls = {}
    fake_lhotse = types.ModuleType("lhotse")

    class _Recording:
        @staticmethod
        def from_bytes(*, data, recording_id):
            calls["data"] = data
            calls["recording_id"] = recording_id
            return "recording"

    fake_lhotse.Recording = _Recording
    monkeypatch.setitem(sys.modules, "lhotse", fake_lhotse)

    stats = Counter()
    recording = common.build_recording_from_audio_bytes(
        b"RIFFpayload",
        "clip-1",
        runtime_counts=stats,
    )

    assert recording == "recording"
    assert calls == {"data": b"RIFFpayload", "recording_id": "clip-1"}
    assert stats["recording_from_bytes"] == 1


def test_build_recording_from_audio_bytes_stringifies_recording_id(monkeypatch):
    calls = {}
    fake_lhotse = types.ModuleType("lhotse")

    class _Recording:
        @staticmethod
        def from_bytes(*, data, recording_id):
            calls["data"] = data
            calls["recording_id"] = recording_id
            return "recording"

    fake_lhotse.Recording = _Recording
    monkeypatch.setitem(sys.modules, "lhotse", fake_lhotse)

    stats = Counter()
    recording = common.build_recording_from_audio_bytes(
        b"audio-bytes",
        123,
        runtime_counts=stats,
    )

    assert recording == "recording"
    assert calls == {"data": b"audio-bytes", "recording_id": "123"}
    assert stats["recording_from_bytes"] == 1
def test_resolve_input_source_and_clip_num_defaults_to_raw_id_and_chunk_idx():
    assert common.resolve_input_source_and_clip_num("clip-1", chunk_idx=3) == (
        "clip-1",
        3,
    )


def test_resolve_input_source_and_clip_num_uses_legacy_parser():
    parsed = common.resolve_input_source_and_clip_num(
        "conv_07f9708fc0b8316a9dea85d473db112b_00005",
        input_clip_id_parser=lambda clip_id: (clip_id.rsplit("_", 1)[0], 5),
    )

    assert parsed == ("conv_07f9708fc0b8316a9dea85d473db112b", 5)


def test_resolve_input_source_and_clip_num_rejects_chunked_legacy_parser():
    try:
        common.resolve_input_source_and_clip_num(
            "row00000_seg003",
            chunk_idx=1,
            input_clip_id_parser=lambda clip_id: ("row00000", 3),
        )
    except ValueError as exc:
        assert "cannot be combined with chunked outputs" in str(exc)
    else:
        raise AssertionError("expected ValueError for chunked parser mode")


def test_set_universal_cut_id_updates_cut_and_supervision_ids():
    supervision = types.SimpleNamespace(id="old-sup")
    cut = types.SimpleNamespace(id="old-cut", supervisions=[supervision], custom=None)

    common.set_universal_cut_id(cut, "source", 7, clip_start=1.25)

    assert cut.id == "source@000007"
    assert supervision.id == "source@000007"
    assert cut.custom == {
        "source_id": "source",
        "clip_num": 7,
        "clip_start": 1.25,
        "legacy_cut_id": "old-cut",
    }


def test_load_external_metadata_reads_jsonl_gz(tmp_path):
    path = tmp_path / "meta.jsonl.gz"
    with gzip.open(path, "wb") as f:
        f.write(
            b'{"id":"clip-1","text":"hello","speaker":"ann"}\n'
            b'{"id":"clip-2","text":"bye","speaker":"bob"}\n'
        )

    metadata = common.load_external_metadata(
        str(path),
        ("speaker",),
        id_field="id",
        text_field="text",
    )

    assert metadata == {
        "clip-1": ("hello", {"speaker": "ann"}),
        "clip-2": ("bye", {"speaker": "bob"}),
    }


def test_resolve_sample_text_and_custom_prefers_external_metadata():
    text, custom = common.resolve_sample_text_and_custom(
        "clip-1",
        default_text="row text",
        default_custom={"speaker": "row", "lang": "yue"},
        external_metadata={
            "clip-1": ("external text", {"speaker": "ext", "topic": "budget"})
        },
    )

    assert text == "external text"
    assert custom == {"speaker": "ext", "lang": "yue", "topic": "budget"}
