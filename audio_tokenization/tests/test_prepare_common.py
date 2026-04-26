import sys
import types
from collections import Counter
import gzip

import numpy as np
import pytest

from audio_tokenization.prepare.audio_ops import (
    apply_audio_pipeline,
    build_recording_from_audio_bytes,
)
from audio_tokenization.prepare.columnar import _projected_columns, extract_row_metadata
from audio_tokenization.prepare.identity import (
    resolve_input_source_and_clip_num,
    set_interleave_metadata,
)
from audio_tokenization.prepare.metadata import (
    load_external_metadata,
    resolve_sample_text_and_custom,
)


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
    recording = build_recording_from_audio_bytes(
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
    recording = build_recording_from_audio_bytes(
        b"audio-bytes",
        123,
        runtime_counts=stats,
    )

    assert recording == "recording"
    assert calls == {"data": b"audio-bytes", "recording_id": "123"}
    assert stats["recording_from_bytes"] == 1
def test_resolve_input_source_and_clip_num_defaults_to_raw_id_and_chunk_idx():
    assert resolve_input_source_and_clip_num("clip-1", chunk_idx=3) == (
        "clip-1",
        3,
    )


def test_resolve_input_source_and_clip_num_uses_explicit_parser():
    parsed = resolve_input_source_and_clip_num(
        "conv_07f9708fc0b8316a9dea85d473db112b_00005",
        input_clip_id_parser=lambda clip_id: (clip_id.rsplit("_", 1)[0], 5),
    )

    assert parsed == ("conv_07f9708fc0b8316a9dea85d473db112b", 5)


def test_resolve_input_source_and_clip_num_rejects_chunked_parser_mode():
    try:
        resolve_input_source_and_clip_num(
            "row00000_seg003",
            chunk_idx=1,
            input_clip_id_parser=lambda clip_id: ("row00000", 3),
        )
    except ValueError as exc:
        assert "cannot be combined with chunked outputs" in str(exc)
    else:
        raise AssertionError("expected ValueError for chunked parser mode")


def test_set_interleave_metadata_preserves_cut_and_supervision_ids():
    supervision = types.SimpleNamespace(id="old-sup")
    cut = types.SimpleNamespace(id="old-cut", duration=2.0, supervisions=[supervision], custom=None)

    set_interleave_metadata(cut, "source", 7, clip_start=1.25)

    assert cut.id == "old-cut"
    assert supervision.id == "old-sup"
    assert cut.custom == {
        "interleave": {
            "source_id": "source",
            "clip_num": 7,
            "clip_start": 1.25,
            "clip_duration": 2.0,
        },
    }


def test_set_interleave_metadata_records_optional_clip_duration():
    cut = types.SimpleNamespace(id="old-cut", duration=3.0, supervisions=[], custom=None)

    set_interleave_metadata(cut, "source", 1, clip_start=1.0, clip_duration=1.5)

    assert cut.custom["interleave"]["clip_start"] == 1.0
    assert cut.custom["interleave"]["clip_duration"] == 1.5


def test_apply_audio_pipeline_can_return_decoded_audio_for_shar_write():
    audio = np.full((1, 16000), 0.1, dtype=np.float32)
    calls = Counter()

    class Cut:
        sampling_rate = 16000
        num_channels = 1
        custom = None

        def load_audio(self):
            calls["load_audio"] += 1
            return audio

    cut, skip, decoded_audio = apply_audio_pipeline(
        Cut(),
        target_sr=None,
        runtime_counts=Counter(),
    )

    assert skip is False
    assert calls["load_audio"] == 1
    assert decoded_audio is audio
    assert cut.custom["rms_db"] == pytest.approx(-20.0)


def test_load_external_metadata_reads_jsonl_gz(tmp_path):
    path = tmp_path / "meta.jsonl.gz"
    with gzip.open(path, "wb") as f:
        f.write(
            b'{"id":"clip-1","text":"hello","speaker":"ann"}\n'
            b'{"id":"clip-2","text":"bye","speaker":"bob"}\n'
        )

    metadata = load_external_metadata(
        str(path),
        ("speaker",),
        id_field="id",
        text_field="text",
    )

    assert metadata == {
        "clip-1": ("hello", {"speaker": "ann"}),
        "clip-2": ("bye", {"speaker": "bob"}),
    }


def test_load_external_metadata_reads_headered_tsv(tmp_path):
    path = tmp_path / "meta.tsv"
    path.write_text(
        "audio_id\tspeaker_id\tduration\tnormalized_text\n"
        "utt-1\tspk-a\t1.2\thalló\n"
        "utt-2\tspk-b\t2.3\tbless\n",
        encoding="utf-8",
    )

    metadata = load_external_metadata(
        str(path),
        ("speaker_id", "duration"),
        id_field="audio_id",
        text_field="normalized_text",
    )

    assert metadata == {
        "utt-1": ("halló", {"speaker_id": "spk-a", "duration": "1.2"}),
        "utt-2": ("bless", {"speaker_id": "spk-b", "duration": "2.3"}),
    }


def test_resolve_sample_text_and_custom_prefers_external_metadata():
    text, custom = resolve_sample_text_and_custom(
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
    row_id, text, lang, custom = extract_row_metadata(
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
    row_id, _, _, _ = extract_row_metadata(
        row,
        id_column=None,
        text_column="text",
        fallback_id="filename_42",
    )
    assert row_id == "filename_42"


def test_extract_row_metadata_multi_column_id_joined_with_underscore():
    row = {"session": 19, "seg": 108}
    row_id, _, _, _ = extract_row_metadata(
        row,
        id_column=["session", "seg"],
    )
    assert row_id == "19_108"


def test_extract_row_metadata_id_is_always_string():
    row = {"id": 42}
    row_id, _, _, _ = extract_row_metadata(row, id_column="id")
    assert row_id == "42"
    assert isinstance(row_id, str)


def test_extract_row_metadata_language_column_takes_precedence_over_global():
    row = {"lang": "de"}
    _, _, lang, _ = extract_row_metadata(
        row,
        language_column="lang",
        language="fr",  # global fallback, should be overridden
    )
    assert lang == "de"


def test_extract_row_metadata_language_falls_back_to_global_when_column_missing():
    row = {"text": "bonjour"}
    _, _, lang, _ = extract_row_metadata(
        row,
        language_column="lang",  # column not in row
        language="fr",
    )
    assert lang == "fr"


def test_extract_row_metadata_language_global_when_no_column_specified():
    row = {"text": "hello"}
    _, _, lang, _ = extract_row_metadata(
        row,
        language="en",
    )
    assert lang == "en"


def test_extract_row_metadata_text_none_when_no_column():
    row = {"id": "abc"}
    _, text, _, _ = extract_row_metadata(row, id_column="id")
    assert text is None


def test_extract_row_metadata_custom_skips_none_values():
    row = {"id": "abc", "speaker": "alice", "age": None, "dialect": "us"}
    _, _, _, custom = extract_row_metadata(
        row,
        id_column="id",
        custom_columns=["speaker", "age", "dialect"],
    )
    assert custom == {"speaker": "alice", "dialect": "us"}


def test_extract_row_metadata_custom_empty_when_no_columns():
    row = {"id": "abc"}
    _, _, _, custom = extract_row_metadata(row, id_column="id")
    assert custom == {}


def test_extract_row_metadata_custom_empty_when_all_values_none():
    row = {"id": "abc", "speaker": None}
    _, _, _, custom = extract_row_metadata(
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
    row_id, text, lang, custom = extract_row_metadata(
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
        extract_row_metadata(
            row,
            id_column="audio.path",
        )


def test_extract_row_metadata_null_required_id_raises():
    row = {"audio": {"path": None}}
    with pytest.raises(ValueError, match="audio.path"):
        extract_row_metadata(
            row,
            id_column="audio.path",
        )


def test_extract_row_metadata_multi_column_id_any_null_raises():
    row = {"audio": {"path": "x.wav"}, "meta": {"seg": None}}
    with pytest.raises(ValueError, match="meta.seg"):
        extract_row_metadata(
            row,
            id_column=["audio.path", "meta.seg"],
        )


def test_extract_row_metadata_dotted_path_missing_intermediate_is_graceful():
    row = {"audio": {}}
    row_id, text, lang, custom = extract_row_metadata(
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
    assert _projected_columns("audio.path") == ["audio.path"]


def test_projected_columns_drops_leaf_when_ancestor_present():
    assert _projected_columns("audio", "audio.path") == ["audio"]


def test_projected_columns_deduplicates():
    assert _projected_columns("audio.path", "audio.path") == ["audio.path"]


def test_projected_columns_handles_list_id_column():
    assert _projected_columns(["audio.path"], "text") == ["audio.path", "text"]
