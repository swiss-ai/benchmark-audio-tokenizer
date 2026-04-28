from collections import Counter

import numpy as np

from audio_tokenization.prepare.preprocess.chunking import (
    VADChunkingConfig,
    filter_vad_timestamps_by_rms,
    split_cut_by_vad,
)


class _ArrayCut:
    def __init__(
        self,
        audio: np.ndarray,
        *,
        sampling_rate: int,
        start_sample: int = 0,
        num_samples: int | None = None,
        loads: list[int] | None = None,
    ):
        self._audio = audio
        self.sampling_rate = sampling_rate
        self._start_sample = start_sample
        self._num_samples = (
            len(audio) - start_sample
            if num_samples is None
            else num_samples
        )
        self.duration = self._num_samples / sampling_rate
        self.id = "recording"
        self.custom = None
        self._loads = loads if loads is not None else []

    def truncate(self, *, offset: float, duration: float, preserve_id: bool = False):
        del preserve_id
        start = self._start_sample + int(round(offset * self.sampling_rate))
        samples = int(round(duration * self.sampling_rate))
        return _ArrayCut(
            self._audio,
            sampling_rate=self.sampling_rate,
            start_sample=start,
            num_samples=samples,
            loads=self._loads,
        )

    def load_audio(self):
        end = self._start_sample + self._num_samples
        self._loads.append(self._num_samples)
        return self._audio[self._start_sample:end][None, :]


def _audio_with_spans(sample_rate: int = 16000) -> np.ndarray:
    # Loud bands are above -50 dB; the quiet band is below threshold but
    # nonzero, so the threshold comparison is actually exercised.
    audio = np.zeros(sample_rate * 3, dtype=np.float32)
    audio[0:sample_rate] = 0.1
    audio[int(1.2 * sample_rate):int(1.7 * sample_rate)] = 5e-4
    audio[int(1.9 * sample_rate):int(2.9 * sample_rate)] = 0.1
    return audio


def test_rms_prefilter_prevents_quiet_span_from_bridging_loud_spans():
    sample_rate = 16000
    cut = _ArrayCut(_audio_with_spans(sample_rate), sampling_rate=sample_rate)
    timestamps = [
        (0, sample_rate),
        (int(1.2 * sample_rate), int(1.7 * sample_rate)),
        (int(1.9 * sample_rate), int(2.9 * sample_rate)),
    ]

    chunks, reason = split_cut_by_vad(
        cut=cut,
        sample_key="recording",
        vad_lookup={"recording": timestamps},
        cfg=VADChunkingConfig(
            max_chunk_sec=10.0,
            min_chunk_sec=0.5,
            sample_rate=sample_rate,
            max_merge_gap_sec=0.5,
            min_rms_db=-50.0,
        ),
    )

    assert reason == "chunked"
    assert len(chunks) == 2
    assert [c.custom["global_offset_sec"] for c in chunks] == [0.0, 1.9]
    assert [round(c.duration, 1) for c in chunks] == [1.0, 1.0]


def test_rms_prefilter_all_quiet_spans_drop_before_merge():
    sample_rate = 16000
    audio = np.full(sample_rate * 2, 5e-4, dtype=np.float32)
    cut = _ArrayCut(audio, sampling_rate=sample_rate)
    timestamps = [(0, sample_rate), (sample_rate, sample_rate * 2)]

    chunks, reason = split_cut_by_vad(
        cut=cut,
        sample_key="recording",
        vad_lookup={"recording": timestamps},
        cfg=VADChunkingConfig(
            max_chunk_sec=10.0,
            min_chunk_sec=0.5,
            sample_rate=sample_rate,
            max_merge_gap_sec=0.5,
            min_rms_db=-50.0,
        ),
    )

    assert chunks == []
    assert reason == "spans_below_min_rms"


def test_rms_prefilter_decodes_span_subcuts_not_full_recording():
    sample_rate = 16000
    loads: list[int] = []
    cut = _ArrayCut(
        _audio_with_spans(sample_rate),
        sampling_rate=sample_rate,
        loads=loads,
    )

    kept = filter_vad_timestamps_by_rms(
        cut=cut,
        timestamps=[
            (0, sample_rate),
            (int(1.2 * sample_rate), int(1.7 * sample_rate)),
            (int(1.9 * sample_rate), int(2.9 * sample_rate)),
        ],
        sample_rate=sample_rate,
        min_rms_db=-50.0,
    )

    assert len(kept) == 2
    assert len(loads) == 3
    assert max(loads) <= sample_rate


def test_rms_prefilter_skips_over_limit_spans_without_decoding():
    sample_rate = 16000
    loads: list[int] = []
    counts = Counter()
    cut = _ArrayCut(
        np.full(sample_rate * 4, 0.1, dtype=np.float32),
        sampling_rate=sample_rate,
        loads=loads,
    )

    chunks, reason = split_cut_by_vad(
        cut=cut,
        sample_key="recording",
        vad_lookup={"recording": [(0, sample_rate * 4)]},
        cfg=VADChunkingConfig(
            max_chunk_sec=2.0,
            min_chunk_sec=0.5,
            sample_rate=sample_rate,
            max_merge_gap_sec=0.5,
            min_rms_db=-50.0,
        ),
        runtime_counts=counts,
    )

    assert chunks == []
    assert reason == "chunks_below_min_duration"
    assert loads == []
    assert counts["vad_spans_rms_skipped_max_duration"] == 1
    assert counts["vad_spans_rms_checked"] == 0
