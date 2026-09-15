"""Unit tests for the VAD chunking contract.

After the per-span RMS prefilter was removed (it dropped <0.05% of spans
while costing a 76× node-throughput regression from BLAS contention on
per-span decodes), this module covers the surviving chunking surface:

  * split_cut_by_vad: drop reasons (too_short, missing_vad, empty_vad,
    invalid_vad_after_clamp), and the chunked happy path producing subcuts.
  * merge_and_pack_vad: max_duration_sec drop, merge-gap behavior,
    max_chunk_sec packing, min_chunk_sec drop.

Worker-level call shapes (audio_dir/WDS typed args, vad_lookup wiring)
are exercised by test_pipeline_prepare.py and test_prepare_wds_to_shar.py.
"""
from collections import Counter

from audio_tokenization.prepare.preprocess.chunking import (
    VADChunkingConfig,
    merge_and_pack_vad,
    split_cut_by_vad,
)


class _FakeCut:
    """Minimal MonoCut stand-in: knows its duration and supports truncate()."""

    def __init__(self, duration: float, *, sampling_rate: int = 16000):
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.id = "fake"
        self.custom = None

    def truncate(self, *, offset: float, duration: float, preserve_id: bool = False):
        del preserve_id
        return _FakeCut(duration, sampling_rate=self.sampling_rate)


_CFG = VADChunkingConfig(
    max_chunk_sec=10.0,
    min_chunk_sec=1.0,
    sample_rate=16000,
    max_merge_gap_sec=0.5,
    max_duration_sec=10.0,
)


# ---------------------------------------------------------------------------
# split_cut_by_vad — drop reasons
# ---------------------------------------------------------------------------

def test_split_too_short_returns_too_short():
    chunks, reason = split_cut_by_vad(
        cut=_FakeCut(0.5),
        timestamps=[(0, 16000)],
        cfg=_CFG,
    )
    assert chunks == []
    assert reason == "too_short"


def test_split_missing_vad_returns_missing_vad():
    chunks, reason = split_cut_by_vad(
        cut=_FakeCut(5.0),
        timestamps=None,
        cfg=_CFG,
    )
    assert chunks == []
    assert reason == "missing_vad"


def test_split_empty_vad_returns_empty_vad():
    chunks, reason = split_cut_by_vad(
        cut=_FakeCut(5.0),
        timestamps=[],
        cfg=_CFG,
    )
    assert chunks == []
    assert reason == "empty_vad"


def test_split_invalid_after_clamp_returns_invalid_vad_after_clamp():
    # All spans start past the cut's duration -> nothing valid after clamp.
    chunks, reason = split_cut_by_vad(
        cut=_FakeCut(2.0),
        timestamps=[(int(3.0 * 16000), int(4.0 * 16000))],
        cfg=_CFG,
    )
    assert chunks == []
    assert reason == "invalid_vad_after_clamp"


# ---------------------------------------------------------------------------
# split_cut_by_vad — happy path
# ---------------------------------------------------------------------------

def test_split_chunked_emits_subcuts_with_provenance():
    cut = _FakeCut(8.0)
    chunks, reason = split_cut_by_vad(
        cut=cut,
        timestamps=[
            (0, int(2.0 * 16000)),
            (int(3.0 * 16000), int(5.0 * 16000)),
        ],
        cfg=VADChunkingConfig(
            max_chunk_sec=10.0,
            min_chunk_sec=1.0,
            sample_rate=16000,
            max_merge_gap_sec=2.0,  # bridge the 1-sec gap
        ),
    )
    assert reason == "chunked"
    assert len(chunks) == 1
    only = chunks[0]
    assert only.custom["source_recording_id"] == cut.id
    assert only.custom["global_offset_sec"] == 0.0
    assert round(only.duration, 2) == 5.0


def test_split_chunked_with_max_duration_dropped_returns_chunks_below_min_duration():
    # One huge raw span that exceeds max_duration_sec gets dropped, leaving nothing.
    chunks, reason = split_cut_by_vad(
        cut=_FakeCut(30.0),
        timestamps=[(0, int(20.0 * 16000))],
        cfg=VADChunkingConfig(
            max_chunk_sec=10.0,
            min_chunk_sec=1.0,
            sample_rate=16000,
            max_merge_gap_sec=0.5,
            max_duration_sec=10.0,
        ),
    )
    assert chunks == []
    assert reason == "chunks_below_min_duration"


# ---------------------------------------------------------------------------
# merge_and_pack_vad — gap, packing, duration filters
# ---------------------------------------------------------------------------

def test_merge_drops_individual_segment_over_max_duration():
    # 25-sec raw segment > max_duration_sec=10 -> dropped.
    out = merge_and_pack_vad(
        timestamps=[(0, int(25.0 * 16000))],
        audio_duration_sec=30.0,
        sample_rate=16000,
        max_merge_gap_sec=0.5,
        max_chunk_sec=10.0,
        min_chunk_sec=1.0,
        max_duration_sec=10.0,
    )
    assert out == []


def test_merge_bridges_small_gap():
    out = merge_and_pack_vad(
        timestamps=[
            (0, int(2.0 * 16000)),
            (int(2.3 * 16000), int(4.0 * 16000)),
        ],
        audio_duration_sec=10.0,
        sample_rate=16000,
        max_merge_gap_sec=0.5,
        max_chunk_sec=10.0,
        min_chunk_sec=1.0,
    )
    assert len(out) == 1
    offset, dur = out[0]
    assert offset == 0.0
    assert round(dur, 2) == 4.0


def test_merge_preserves_large_gap():
    out = merge_and_pack_vad(
        timestamps=[
            (0, int(2.0 * 16000)),
            (int(5.0 * 16000), int(7.0 * 16000)),
        ],
        audio_duration_sec=10.0,
        sample_rate=16000,
        max_merge_gap_sec=0.5,  # gap is 3 sec, far bigger
        max_chunk_sec=10.0,
        min_chunk_sec=1.0,
    )
    assert len(out) == 2
    assert round(out[0][0], 2) == 0.0 and round(out[0][1], 2) == 2.0
    assert round(out[1][0], 2) == 5.0 and round(out[1][1], 2) == 2.0


def test_merge_packs_into_max_chunk_sec():
    # Two ~3-sec segments separated by a small gap, max_chunk_sec=4 forces split.
    out = merge_and_pack_vad(
        timestamps=[
            (0, int(3.0 * 16000)),
            (int(3.2 * 16000), int(6.2 * 16000)),
        ],
        audio_duration_sec=10.0,
        sample_rate=16000,
        max_merge_gap_sec=0.5,
        max_chunk_sec=4.0,
        min_chunk_sec=1.0,
    )
    # Combined would be 6.2s > 4s, so they don't merge into one chunk.
    assert len(out) == 2


def test_merge_drops_chunks_under_min_chunk_sec():
    out = merge_and_pack_vad(
        timestamps=[(0, int(0.5 * 16000))],
        audio_duration_sec=10.0,
        sample_rate=16000,
        max_merge_gap_sec=0.5,
        max_chunk_sec=10.0,
        min_chunk_sec=1.0,
    )
    assert out == []


# ---------------------------------------------------------------------------
# Counter wiring
# ---------------------------------------------------------------------------

def test_split_increments_no_unexpected_counters_on_chunked_path():
    counts = Counter()
    chunks, reason = split_cut_by_vad(
        cut=_FakeCut(5.0),
        timestamps=[(0, int(3.0 * 16000))],
        cfg=_CFG,
        runtime_counts=counts,
    )
    assert reason == "chunked"
    assert len(chunks) == 1
    # Removed RMS counters must not be referenced by surviving code paths.
    for stale in (
        "vad_spans_rms_invalid",
        "vad_spans_rms_errors",
        "vad_spans_rms_checked",
        "vad_spans_rms_dropped",
        "vad_spans_rms_kept",
        "vad_spans_rms_skipped_max_duration",
        "vad_cuts_all_spans_rms_dropped",
    ):
        assert stale not in counts
