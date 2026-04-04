from types import SimpleNamespace

import pytest

from audio_tokenization.pipelines.lhotse.audio_text import resolve_interleaving_metadata


def _make_cut(*, cut_id="clip", start=0.0, custom=None):
    return SimpleNamespace(id=cut_id, start=start, custom=custom)


def test_resolve_interleaving_metadata_prefers_cut_custom():
    cut = _make_cut(
        cut_id="legacy_00005",
        start=9.0,
        custom={
            "source_id": "session_a",
            "clip_num": 7,
            "clip_start": 1.25,
        },
    )

    source_id, clip_num, clip_start = resolve_interleaving_metadata(
        cut,
        clip_id_parser=lambda _: ("wrong", 999),
    )

    assert source_id == "session_a"
    assert clip_num == 7
    assert clip_start == 1.25


def test_resolve_interleaving_metadata_falls_back_to_parser_for_legacy_shar():
    cut = _make_cut(cut_id="session_a@000003", start=0.5, custom=None)

    source_id, clip_num, clip_start = resolve_interleaving_metadata(
        cut,
        clip_id_parser=lambda clip_id: ("session_a", 3),
    )

    assert source_id == "session_a"
    assert clip_num == 3
    assert clip_start == 0.5


def test_resolve_interleaving_metadata_requires_custom_or_parser():
    cut = _make_cut(cut_id="legacy_5", start=0.0, custom=None)

    with pytest.raises(ValueError, match="missing canonical interleaving metadata"):
        resolve_interleaving_metadata(cut, clip_id_parser=None)
