"""Tests for audio_tokenization.utils.clip_id_parsers."""

import pytest

from audio_tokenization.utils.clip_id_parsers import (
    get_clip_id_parser,
    parse_aishell_clip_id,
    parse_coral_clip_id,
    parse_emilia_clip_id,
    parse_generic_clip_id,
    parse_legco_clip_id,
    parse_peoples_speech_clip_id,
    parse_spc_clip_id,
    parse_wenetspeech_clip_id,
)


class TestEmilia:
    def test_basic(self):
        assert parse_emilia_clip_id("EN_tKvmUvxYZXI_W000006") == (
            "EN_tKvmUvxYZXI",
            6,
        )

    def test_leading_zeros(self):
        assert parse_emilia_clip_id("ZH_abc123_W000000") == ("ZH_abc123", 0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_emilia_clip_id("no_w_suffix")


class TestPeoplesSpeech:
    def test_with_flac(self):
        assert parse_peoples_speech_clip_id(
            "forum_SLASH_foo_DOT_mp3_00002.flac"
        ) == ("forum_SLASH_foo_DOT_mp3", 2)

    def test_without_extension(self):
        assert parse_peoples_speech_clip_id("src_00010") == ("src", 10)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_peoples_speech_clip_id("no_number")


class TestWenetSpeech:
    def test_basic(self):
        assert parse_wenetspeech_clip_id("L_T0000005699_S00003") == (
            "L_T0000005699",
            3,
        )

    def test_dev_split(self):
        assert parse_wenetspeech_clip_id("DEV_T0000005699_S00000") == (
            "DEV_T0000005699",
            0,
        )

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_wenetspeech_clip_id("missing_S_prefix")


class TestSPC:
    def test_basic(self):
        assert parse_spc_clip_id("row00000_seg003") == ("row00000", 3)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_spc_clip_id("row00000_003")


class TestAishell:
    def test_basic(self):
        assert parse_aishell_clip_id("BAC009S0002W0122") == ("BAC009S0002", 122)

    def test_zero(self):
        assert parse_aishell_clip_id("BAC009S0002W0000") == ("BAC009S0002", 0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_aishell_clip_id("no_w_marker")


class TestLegco:
    def test_basic(self):
        assert parse_legco_clip_id("rIa-Qb8EYsA_123") == ("rIa-Qb8EYsA", 123)

    def test_dedup_suffix(self):
        assert parse_legco_clip_id("rIa-Qb8EYsA_123-0") == ("rIa-Qb8EYsA", 123)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_legco_clip_id("")


class TestCoral:
    def test_basic(self):
        assert parse_coral_clip_id(
            "conv_07f9708fc0b8316a9dea85d473db112b_00005"
        ) == ("conv_07f9708fc0b8316a9dea85d473db112b", 5)

    def test_dedup_suffix(self):
        assert parse_coral_clip_id(
            "conv_07f9708fc0b8316a9dea85d473db112b_00005-1"
        ) == ("conv_07f9708fc0b8316a9dea85d473db112b", 5)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_coral_clip_id("")


class TestGeneric:
    def test_returns_id_and_zero(self):
        assert parse_generic_clip_id("anything_at_all") == ("anything_at_all", 0)

    def test_empty_string(self):
        assert parse_generic_clip_id("") == ("", 0)


class TestRegistry:
    def test_all_known_parsers(self):
        for name in [
            "emilia",
            "peoples_speech",
            "wenetspeech",
            "spc",
            "aishell",
            "legco",
            "coral",
            "generic",
        ]:
            parser = get_clip_id_parser(name)
            assert callable(parser)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown clip_id_parser"):
            get_clip_id_parser("nonexistent")
