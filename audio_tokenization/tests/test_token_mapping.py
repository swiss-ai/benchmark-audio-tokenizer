"""Tests for audio_tokenization.utils.token_mapping."""

import json
import os

import pytest

from audio_tokenization.utils.token_mapping import (
    get_structure_tokens,
    load_audio_token_mapping,
)


@pytest.fixture
def mapping_dir(tmp_path):
    """Create a temp directory with a valid audio_token_mapping.json."""
    mapping = {
        "audio_token_offset": 262344,
        "structure_tokens": {
            "audio_start": 100,
            "audio_end": 101,
            "stt_transcribe": 102,
            "stt_continue": 103,
            "tts_continue": 104,
        },
    }
    (tmp_path / "audio_token_mapping.json").write_text(json.dumps(mapping))
    return str(tmp_path)


class TestLoadAudioTokenMapping:
    def test_loads_valid(self, mapping_dir):
        m = load_audio_token_mapping(mapping_dir)
        assert m["audio_token_offset"] == 262344
        assert "structure_tokens" in m

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_audio_token_mapping(str(tmp_path))

    def test_missing_structure_tokens(self, tmp_path):
        (tmp_path / "audio_token_mapping.json").write_text("{}")
        with pytest.raises(ValueError, match="No 'structure_tokens'"):
            load_audio_token_mapping(str(tmp_path))


class TestGetStructureTokens:
    def test_returns_dict(self, mapping_dir):
        st = get_structure_tokens(mapping_dir)
        assert st["audio_start"] == 100

    def test_required_keys_pass(self, mapping_dir):
        st = get_structure_tokens(mapping_dir, required=["audio_start", "audio_end"])
        assert st["audio_start"] == 100

    def test_required_keys_fail(self, mapping_dir):
        with pytest.raises(ValueError, match="'nonexistent' missing"):
            get_structure_tokens(mapping_dir, required=["nonexistent"])
