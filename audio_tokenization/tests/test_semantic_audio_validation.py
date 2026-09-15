from __future__ import annotations

import gzip
import json

from audio_tokenization.validation import semantic_audio


def test_load_reference_texts_from_shar_reads_supervision_text(tmp_path):
    shar = tmp_path / "shar"
    shar.mkdir()
    with gzip.open(shar / "cuts.000000.jsonl.gz", "wt") as f:
        f.write(
            json.dumps(
                {
                    "id": "cut-a",
                    "supervisions": [{"text": "hello from source"}],
                }
            )
            + "\n"
        )

    assert semantic_audio.load_reference_texts_from_shar([shar]) == {
        "cut-a": "hello from source"
    }


def test_sample_megatron_uses_reference_text_not_decoded_text(tmp_path, monkeypatch):
    prefix = tmp_path / "rank_0000_chunk_0000"

    class FakeReader:
        document_count = 1

        def __init__(self, _prefix):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def close(self):
            pass

        def read_document(self, _doc_idx):
            return (1, 2, 3)

    class FakeDecoder:
        def split_megatron_sequence(self, _seq):
            return [100, 101], [10, 11]

        def decode_text(self, _tokens):
            return "decoded token text"

    monkeypatch.setattr(
        semantic_audio,
        "discover_cut_id_prefixes",
        lambda _root, recursive=False: [prefix],
    )
    monkeypatch.setattr(
        semantic_audio,
        "read_cut_id_sidecar",
        lambda _path: ["cut-a"],
    )
    monkeypatch.setattr(semantic_audio, "MegatronChunkReader", FakeReader)

    samples = semantic_audio.sample_megatron_outputs(
        tmp_path,
        decoder=FakeDecoder(),
        num_samples=1,
        seed=0,
        recursive=False,
        max_audio_seconds=None,
        reference_texts={"cut-a": "source reference text"},
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.expected_text == "source reference text"
    assert sample.decoded_text_tokens == "decoded token text"
    assert sample.metadata["reference_text_missing"] is False
    assert sample.metadata["decoded_text_matches_source"] is False


def test_sample_megatron_marks_missing_reference_unscored(tmp_path, monkeypatch):
    prefix = tmp_path / "rank_0000_chunk_0000"

    class FakeReader:
        document_count = 1

        def __init__(self, _prefix):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def close(self):
            pass

        def read_document(self, _doc_idx):
            return (1, 2, 3)

    class FakeDecoder:
        def split_megatron_sequence(self, _seq):
            return [100, 101], [10, 11]

        def decode_text(self, _tokens):
            return "decoded token text"

    monkeypatch.setattr(
        semantic_audio,
        "discover_cut_id_prefixes",
        lambda _root, recursive=False: [prefix],
    )
    monkeypatch.setattr(
        semantic_audio,
        "read_cut_id_sidecar",
        lambda _path: ["cut-a"],
    )
    monkeypatch.setattr(semantic_audio, "MegatronChunkReader", FakeReader)

    samples = semantic_audio.sample_megatron_outputs(
        tmp_path,
        decoder=FakeDecoder(),
        num_samples=1,
        seed=0,
        recursive=False,
        max_audio_seconds=None,
        reference_texts=None,
    )

    assert samples[0].expected_text == ""
    assert samples[0].metadata["reference_text_missing"] is True


def test_interleave_samples_own_tokens_without_retaining_mapped_views(monkeypatch, tmp_path):
    import gc
    import weakref
    from types import SimpleNamespace

    import numpy as np

    from audio_tokenization.interleave import common
    from audio_tokenization.pipelines.shard_io import StructuredCacheChunkWriter

    writer = StructuredCacheChunkWriter(
        str(tmp_path), rank=0, partitioning={"type": "hash", "field": "source_id", "num_buckets": 1},
    )
    writer.add_rows([
        {"clip_id": str(i), "source_id": "a", "clip_num": i, "clip_start": None,
         "speaker": "", "duration": 1.0, "text": "hello", "dataset": "ds",
         "audio_tokens": [10, 70000 + i, 11], "text_tokens": [1, 2]}
        for i in range(16)
    ])
    writer.finalize()
    get = common._MemmapTokenAccessor.get
    spans = []

    def observe_span(accessor, index):
        span = get(accessor, index)
        spans.append(weakref.ref(span))
        return span

    monkeypatch.setattr(common._MemmapTokenAccessor, "get", observe_span)
    samples = semantic_audio.sample_interleave_cache(
        tmp_path, decoder=SimpleNamespace(decode_text=lambda tokens: "hello"),
        num_samples=16, seed=42, max_audio_seconds=None, context_window=1,
    )
    assert len(samples) == 16
    for sample in samples:
        np.testing.assert_array_equal(sample.audio_tokens, [10, 70000 + sample.metadata["clip_num"], 11])
        assert sample.expected_text == "hello"
    gc.collect()
    assert spans and all(span() is None for span in spans)
