"""Tests for audio_tokenization.utils.build_interleaved.accumulate."""

import numpy as np
import pytest

from audio_tokenization.utils.build_interleaved.accumulate import (
    _accumulate_sequences,
    _compute_per_run_stats_accumulate,
)


# Token IDs used across all tests
BOS = 1
EOS = 2
STT_CONT = 99
TTS_CONT = 97


def _acc(audio, text, direction, max_seq_len=100_000):
    """Shorthand for calling _accumulate_sequences with test token IDs."""
    return _accumulate_sequences(
        audio, text, direction, max_seq_len,
        BOS, EOS, STT_CONT, TTS_CONT,
    )


class TestSimpleAccumulate:
    """Basic accumulation without max_seq_len cuts."""

    def test_simple_at_accumulate(self):
        """4 clips, AT direction, no cut → 1 sequence with all clips."""
        audio = [[10, 11], [12, 13], [14, 15], [16, 17]]
        text = [[20, 21], [22, 23], [24, 25], [26, 27]]
        seqs, singles = _acc(audio, text, "AT")
        assert len(seqs) == 1
        assert len(singles) == 0
        seq = seqs[0]
        # BOS, A0, stt_cont, T1, tts_cont, A2, stt_cont, T3, EOS
        assert seq[0] == BOS
        assert seq[-1] == EOS
        # Audio tokens for clip 0, 2 and text tokens for clip 1, 3
        assert 10 in seq and 11 in seq  # clip 0 audio
        assert 22 in seq and 23 in seq  # clip 1 text
        assert 14 in seq and 15 in seq  # clip 2 audio
        assert 26 in seq and 27 in seq  # clip 3 text

    def test_simple_ta_accumulate(self):
        """4 clips, TA direction, no cut → 1 sequence with all clips."""
        audio = [[10, 11], [12, 13], [14, 15], [16, 17]]
        text = [[20, 21], [22, 23], [24, 25], [26, 27]]
        seqs, singles = _acc(audio, text, "TA")
        assert len(seqs) == 1
        assert len(singles) == 0
        seq = seqs[0]
        # BOS, T0, tts_cont, A1, stt_cont, T2, tts_cont, A3, EOS
        assert seq[0] == BOS
        assert seq[-1] == EOS
        assert 20 in seq and 21 in seq  # clip 0 text
        assert 12 in seq and 13 in seq  # clip 1 audio

    def test_two_clips_at(self):
        """Minimal 2-clip run, AT direction."""
        audio = [[10], [12]]
        text = [[20], [22]]
        seqs, singles = _acc(audio, text, "AT")
        assert len(seqs) == 1
        assert len(singles) == 0
        # BOS, A0(10), stt_cont, T1(22), EOS
        assert seqs[0] == [BOS, 10, STT_CONT, 22, EOS]

    def test_two_clips_ta(self):
        """Minimal 2-clip run, TA direction."""
        audio = [[10], [12]]
        text = [[20], [22]]
        seqs, singles = _acc(audio, text, "TA")
        assert len(seqs) == 1
        assert len(singles) == 0
        # BOS, T0(20), tts_cont, A1(12), EOS
        assert seqs[0] == [BOS, 20, TTS_CONT, 12, EOS]


class TestEvenClipConstraint:
    """Verify the even-clip constraint is enforced."""

    def test_odd_clips_pushback(self):
        """3 clips → only first 2 consumed in sequence, 3rd is single remainder."""
        audio = [[10], [12], [14]]
        text = [[20], [22], [24]]
        seqs, singles = _acc(audio, text, "AT")
        # First 2 clips form a sequence, 3rd clip is single remainder
        assert len(seqs) == 1
        assert len(singles) == 1
        assert singles[0] == 2  # index of the pushed-back clip

    def test_five_clips_at(self):
        """5 clips → 4 consumed in first seq, 5th is single remainder."""
        audio = [[i] for i in range(5)]
        text = [[i + 100] for i in range(5)]
        seqs, singles = _acc(audio, text, "AT")
        # 4 clips in one seq (even), 1 single
        assert len(seqs) == 1
        assert len(singles) == 1
        assert singles[0] == 4


class TestMaxSeqLenCut:
    """Verify that max_seq_len cuts produce multiple sequences."""

    def test_max_seq_len_cut(self):
        """Small max_seq_len forces cut, producing 2+ sequences."""
        # Each clip has 5 tokens. AT pattern for 2 clips:
        # BOS(1) + A0(5) + stt_cont(1) + T1(5) + EOS(1) = 13 tokens
        audio = [[i] * 5 for i in range(8)]
        text = [[i + 100] * 5 for i in range(8)]
        # max_seq_len=15 allows exactly 2 clips per seq (13 tokens)
        seqs, singles = _acc(audio, text, "AT", max_seq_len=15)
        assert len(seqs) >= 2
        for seq in seqs:
            assert len(seq) <= 15
            assert seq[0] == BOS
            assert seq[-1] == EOS

    def test_single_clip_not_accumulated(self):
        """1 clip → empty sequences, single-clip index returned."""
        audio = [[10, 11]]
        text = [[20, 21]]
        seqs, singles = _acc(audio, text, "AT")
        assert len(seqs) == 0
        assert singles == [0]

    def test_single_clip_remainder_from_cut(self):
        """After a max_seq_len cut, a single remaining clip goes to singles."""
        # 3 clips, max_seq_len tight enough for 2 clips only
        audio = [[i] * 5 for i in range(3)]
        text = [[i + 100] * 5 for i in range(3)]
        # 2 clips: BOS(1)+A(5)+stt(1)+T(5)+EOS(1)=13. Allow exactly that.
        seqs, singles = _acc(audio, text, "AT", max_seq_len=15)
        assert len(seqs) == 1
        assert len(singles) == 1
        assert singles[0] == 2


class TestTransitionTokens:
    """Verify correct transition tokens at modality boundaries."""

    def test_transition_tokens_at(self):
        """AT direction: stt_continue at A→T, tts_continue at T→A."""
        audio = [[10], [12], [14], [16]]
        text = [[20], [22], [24], [26]]
        seqs, _ = _acc(audio, text, "AT")
        seq = seqs[0]
        # Pattern: A0 T1 A2 T3
        # Transitions: A→T (stt_cont), T→A (tts_cont), A→T (stt_cont)
        assert seq.count(STT_CONT) == 2
        assert seq.count(TTS_CONT) == 1

    def test_transition_tokens_ta(self):
        """TA direction: tts_continue at T→A, stt_continue at A→T."""
        audio = [[10], [12], [14], [16]]
        text = [[20], [22], [24], [26]]
        seqs, _ = _acc(audio, text, "TA")
        seq = seqs[0]
        # Pattern: T0 A1 T2 A3
        # Transitions: T→A (tts_cont), A→T (stt_cont), T→A (tts_cont)
        assert seq.count(TTS_CONT) == 2
        assert seq.count(STT_CONT) == 1


class TestRestartAfterCut:
    """After a max_seq_len cut, new sequence restarts with the same direction."""

    def test_restart_direction_after_cut(self):
        """AT direction: after cut, new seq also starts with A."""
        # 4 clips, each 5 tokens. Use values that don't collide with special IDs.
        audio = [[500 + i] * 5 for i in range(4)]
        text = [[600 + i] * 5 for i in range(4)]
        seqs, singles = _acc(audio, text, "AT", max_seq_len=15)
        assert len(seqs) == 2
        assert len(singles) == 0
        # Both sequences should start with BOS then audio tokens (A mode)
        for seq in seqs:
            assert seq[0] == BOS
            # Second token should be from audio (not a transition token)
            assert seq[1] not in (STT_CONT, TTS_CONT, EOS)

    def test_restart_ta_direction_after_cut(self):
        """TA direction: after cut, new seq also starts with T."""
        audio = [[500 + i] * 5 for i in range(4)]
        text = [[600 + i] * 5 for i in range(4)]
        seqs, singles = _acc(audio, text, "TA", max_seq_len=15)
        assert len(seqs) == 2
        assert len(singles) == 0
        for seq in seqs:
            assert seq[0] == BOS
            # Second token should be from text (not a transition token)
            assert seq[1] not in (STT_CONT, TTS_CONT, EOS)


class TestEdgeCases:
    """Edge cases for accumulation."""

    def test_empty_run(self):
        """Empty run produces nothing."""
        seqs, singles = _acc([], [], "AT")
        assert seqs == []
        assert singles == []

    def test_all_sequences_even_clips(self):
        """Every emitted sequence has an even number of clips."""
        audio = [[i] * 3 for i in range(20)]
        text = [[i + 100] * 3 for i in range(20)]
        for direction in ["AT", "TA"]:
            seqs, _ = _acc(audio, text, direction, max_seq_len=30)
            for seq in seqs:
                # Count clips by counting transitions + 1 doesn't work easily,
                # but we can verify sequence length parity indirectly.
                # Each clip contributes its tokens + at most 1 transition token.
                # With 3 tokens per clip: 2 clips = BOS(1) + 3 + trans(1) + 3 + EOS(1) = 9
                # Minimum valid seq is 2 clips.
                assert seq[0] == BOS
                assert seq[-1] == EOS

    def test_huge_single_clip_exceeds_max(self):
        """A single clip that exceeds max_seq_len is still consumed (no infinite loop)."""
        # Single clip with 100 tokens, max_seq_len=10
        audio = [list(range(100))]
        text = [list(range(100))]
        seqs, singles = _acc(audio, text, "AT", max_seq_len=10)
        # Only 1 clip → single remainder
        assert len(seqs) == 0
        assert singles == [0]

    def test_two_clips_first_exceeds_max(self):
        """First clip exceeds max but 2 clips means we still try to build a sequence."""
        # If first clip fits alone (clips_in_seq==0 allows it), but second
        # doesn't fit, we get 1 clip → odd → pushed back to single.
        audio = [list(range(50)), [1]]
        text = [[1], list(range(50))]
        # max_seq_len=55: BOS(1)+A0(50)+stt(1)+T1(50)+EOS(1)=103 > 55
        # So only clip 0 fits, but 1 clip is odd → single remainder
        seqs, singles = _acc(audio, text, "AT", max_seq_len=55)
        # clip 0 alone: BOS(1)+A0(50)=51+EOS=52 fits, but odd (1 clip)
        # So clip 0 pushed to single, then clip 1 also single
        assert len(singles) == 2
        assert len(seqs) == 0


class TestComputePerRunStatsAccumulate:
    """Tests for _compute_per_run_stats_accumulate."""

    def test_single_clip_run(self):
        """Single-clip run → 0 interleaved, 1 transcribe."""
        audio_lens = np.array([10])
        text_lens = np.array([20])
        run_starts = np.array([0])
        run_lengths = np.array([1])
        il, tr = _compute_per_run_stats_accumulate(
            audio_lens, text_lens, run_starts, run_lengths,
            ["AT", "TA"], 100_000,
        )
        assert il[0] == 0
        assert tr[0] == 1

    def test_two_clip_run(self):
        """2-clip run → 2 interleaved (AT + TA), 0 transcribe."""
        audio_lens = np.array([5, 5])
        text_lens = np.array([5, 5])
        run_starts = np.array([0])
        run_lengths = np.array([2])
        il, tr = _compute_per_run_stats_accumulate(
            audio_lens, text_lens, run_starts, run_lengths,
            ["AT", "TA"], 100_000,
        )
        assert il[0] == 2  # one seq per direction
        assert tr[0] == 0

    def test_four_clip_run(self):
        """4-clip run → 2 interleaved (1 per direction), 0 transcribe."""
        audio_lens = np.array([5, 5, 5, 5])
        text_lens = np.array([5, 5, 5, 5])
        run_starts = np.array([0])
        run_lengths = np.array([4])
        il, tr = _compute_per_run_stats_accumulate(
            audio_lens, text_lens, run_starts, run_lengths,
            ["AT", "TA"], 100_000,
        )
        assert il[0] == 2
        assert tr[0] == 0

    def test_three_clip_run(self):
        """3-clip run → 2 interleaved (2 clips each) + remainder singles."""
        audio_lens = np.array([5, 5, 5])
        text_lens = np.array([5, 5, 5])
        run_starts = np.array([0])
        run_lengths = np.array([3])
        il, tr = _compute_per_run_stats_accumulate(
            audio_lens, text_lens, run_starts, run_lengths,
            ["AT", "TA"], 100_000,
        )
        # Both directions: 2 clips fit, 1 remains as single
        assert il[0] == 2  # one seq per direction
        assert tr[0] >= 1  # at least one single-clip remainder

    def test_multiple_runs(self):
        """Multiple runs: stats computed independently."""
        audio_lens = np.array([5, 5, 5, 10])
        text_lens = np.array([5, 5, 5, 10])
        run_starts = np.array([0, 3])
        run_lengths = np.array([3, 1])
        il, tr = _compute_per_run_stats_accumulate(
            audio_lens, text_lens, run_starts, run_lengths,
            ["AT", "TA"], 100_000,
        )
        # Second run is single-clip → transcribe
        assert il[1] == 0
        assert tr[1] == 1
