import sys
import types
from collections import Counter
import gzip

import pytest

from audio_tokenization.prepare import common


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


# ---------------------------------------------------------------------------
# extract_row_metadata
# ---------------------------------------------------------------------------


def test_extract_row_metadata_basic():
    row = {"id": "abc", "text": "hello world", "lang": "en", "speaker": "alice"}
    row_id, text, lang, custom = common.extract_row_metadata(
        row,
        id_column="id",
        text_column="text",
        language_column="lang",
        custom_columns=["speaker"],
    )
    assert row_id == "abc"
    assert text == "hello world"
    assert lang == "en"
    assert custom == {"speaker": "alice"}


def test_extract_row_metadata_fallback_id_when_no_column():
    row = {"text": "hello"}
    row_id, _, _, _ = common.extract_row_metadata(
        row,
        id_column=None,
        text_column="text",
        fallback_id="filename_42",
    )
    assert row_id == "filename_42"


def test_extract_row_metadata_multi_column_id_joined_with_underscore():
    row = {"session": 19, "seg": 108}
    row_id, _, _, _ = common.extract_row_metadata(
        row,
        id_column=["session", "seg"],
    )
    assert row_id == "19_108"


def test_extract_row_metadata_id_is_always_string():
    row = {"id": 42}
    row_id, _, _, _ = common.extract_row_metadata(row, id_column="id")
    assert row_id == "42"
    assert isinstance(row_id, str)


def test_extract_row_metadata_language_column_takes_precedence_over_global():
    row = {"lang": "de"}
    _, _, lang, _ = common.extract_row_metadata(
        row,
        language_column="lang",
        language="fr",  # global fallback, should be overridden
    )
    assert lang == "de"


def test_extract_row_metadata_language_falls_back_to_global_when_column_missing():
    row = {"text": "bonjour"}
    _, _, lang, _ = common.extract_row_metadata(
        row,
        language_column="lang",  # column not in row
        language="fr",
    )
    assert lang == "fr"


def test_extract_row_metadata_language_global_when_no_column_specified():
    row = {"text": "hello"}
    _, _, lang, _ = common.extract_row_metadata(
        row,
        language="en",
    )
    assert lang == "en"


def test_extract_row_metadata_text_none_when_no_column():
    row = {"id": "abc"}
    _, text, _, _ = common.extract_row_metadata(row, id_column="id")
    assert text is None


def test_extract_row_metadata_custom_skips_none_values():
    row = {"id": "abc", "speaker": "alice", "age": None, "dialect": "us"}
    _, _, _, custom = common.extract_row_metadata(
        row,
        id_column="id",
        custom_columns=["speaker", "age", "dialect"],
    )
    assert custom == {"speaker": "alice", "dialect": "us"}


def test_extract_row_metadata_custom_empty_when_no_columns():
    row = {"id": "abc"}
    _, _, _, custom = common.extract_row_metadata(row, id_column="id")
    assert custom == {}


def test_extract_row_metadata_custom_empty_when_all_values_none():
    row = {"id": "abc", "speaker": None}
    _, _, _, custom = common.extract_row_metadata(
        row,
        id_column="id",
        custom_columns=["speaker"],
    )
    assert custom == {}


def test_extract_row_metadata_dotted_path():
    row = {
        "audio": {"path": "x.wav"},
        "meta": {"text": "hello", "lang": "fa"},
        "speaker": {"info": {"name": "alice"}},
    }
    row_id, text, lang, custom = common.extract_row_metadata(
        row,
        id_column="audio.path",
        text_column="meta.text",
        language_column="meta.lang",
        custom_columns=["speaker.info.name"],
    )
    assert row_id == "x.wav"
    assert text == "hello"
    assert lang == "fa"
    assert custom == {"speaker.info.name": "alice"}


def test_extract_row_metadata_missing_required_id_raises():
    row = {"audio": {}}
    with pytest.raises(ValueError, match="audio.path"):
        common.extract_row_metadata(
            row,
            id_column="audio.path",
        )


def test_extract_row_metadata_null_required_id_raises():
    row = {"audio": {"path": None}}
    with pytest.raises(ValueError, match="audio.path"):
        common.extract_row_metadata(
            row,
            id_column="audio.path",
        )


def test_extract_row_metadata_multi_column_id_any_null_raises():
    row = {"audio": {"path": "x.wav"}, "meta": {"seg": None}}
    with pytest.raises(ValueError, match="meta.seg"):
        common.extract_row_metadata(
            row,
            id_column=["audio.path", "meta.seg"],
        )


def test_extract_row_metadata_dotted_path_missing_intermediate_is_graceful():
    row = {"audio": {}}
    row_id, text, lang, custom = common.extract_row_metadata(
        row,
        id_column=None,
        text_column="audio.path",
        language_column="meta.lang",
        language="fa",
        custom_columns=["speaker.info.name"],
        fallback_id="fallback",
    )
    assert row_id == "fallback"
    assert text is None
    assert lang == "fa"
    assert custom == {}


def test_projected_columns_keeps_dotted_leaf_when_no_ancestor():
    assert common._projected_columns("audio.path") == ["audio.path"]


def test_projected_columns_drops_leaf_when_ancestor_present():
    assert common._projected_columns("audio", "audio.path") == ["audio"]


def test_projected_columns_deduplicates():
    assert common._projected_columns("audio.path", "audio.path") == ["audio.path"]


def test_projected_columns_handles_list_id_column():
    assert common._projected_columns(["audio.path"], "text") == ["audio.path", "text"]
