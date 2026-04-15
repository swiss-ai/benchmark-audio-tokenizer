from __future__ import annotations

import pytest

from audio_tokenization.utils.prepare_data.postprocess import add_captions_to_shar


def test_add_captions_rejects_id_rewrite_when_recordings_are_symlinked(tmp_path):
    raw_dicts = [
        {"id": "MRSMusic/music093/chunk_000", "duration": 1.0},
    ]

    with pytest.raises(RuntimeError, match="Full shard rewrite is required"):
        add_captions_to_shar._assert_id_stability_for_symlinked_recordings(
            raw_dicts,
            cuts_path=tmp_path / "cuts.000000.jsonl.gz",
        )


def test_add_captions_allows_stable_ids_when_no_rewrite_is_needed(tmp_path):
    raw_dicts = [
        {"id": "source@000007", "duration": 1.0},
    ]

    add_captions_to_shar._assert_id_stability_for_symlinked_recordings(
        raw_dicts,
        cuts_path=tmp_path / "cuts.000000.jsonl.gz",
    )
