import sys
import types

import pytest
from audio_tokenization.prepare.columnar import (
    ColumnarWorkerArgs,
    derive_timestamp_clip_num,
    extract_row_metadata,
    extract_clip_timestamps,
    extract_interleave_identity,
)
from audio_tokenization.prepare import (
    prepare_hf_to_shar,
    prepare_parquet_to_shar,
)


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def as_py(self):
        return self._value


class _FakeColumn:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        return _FakeScalar(self._values[idx])


class _FakeArrowTable:
    def __init__(self, columns):
        self._columns = columns
        self.num_rows = len(next(iter(columns.values())))

    def column(self, name):
        return _FakeColumn(self._columns[name])

    def to_rows(self):
        return [
            {name: values[i] for name, values in self._columns.items()}
            for i in range(self.num_rows)
        ]


class _FakeArrowBatch:
    def __init__(self, rows):
        self._rows = rows
        self.num_rows = len(rows)

    def to_pylist(self):
        return list(self._rows)

    def slice(self, start, length):
        return _FakeArrowBatch(self._rows[start:start + length])

    def to_table(self):
        return self

    def to_batches(self, *, max_chunksize):
        for start in range(0, len(self._rows), max_chunksize):
            yield _FakeArrowBatch(self._rows[start:start + max_chunksize])


class _FakeArrowReader:
    def __init__(self, table):
        self._table = table

    def __iter__(self):
        yield _FakeArrowBatch(self._table.to_rows())


class _FakeParquetFile:
    def __init__(self, rows):
        self._rows = rows
        self.schema_arrow = types.SimpleNamespace(names=list(rows[0].keys()) if rows else [])

    def iter_batches(self, *, columns=None, batch_size=None, use_threads=False):
        assert use_threads is False
        rows = self._rows
        if columns is not None:
            rows = [{k: row[k] for k in columns if k in row} for row in rows]
        batch_size = batch_size or len(rows)
        for start in range(0, len(rows), batch_size):
            yield _FakeArrowBatch(rows[start:start + batch_size])


class _FakeSupervisionSegment:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeSharWriter:
    def __init__(self, *, sink, **kwargs):
        self._sink = sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def write(self, cut):
        self._sink.append(cut)


def _install_fake_lhotse(monkeypatch, written_cuts):
    fake_lhotse = types.ModuleType("lhotse")
    fake_lhotse.SupervisionSegment = _FakeSupervisionSegment

    fake_lhotse_shar = types.ModuleType("lhotse.shar")
    fake_lhotse_shar.SharWriter = lambda **kwargs: _FakeSharWriter(
        sink=written_cuts,
        **kwargs,
    )

    monkeypatch.setitem(sys.modules, "lhotse", fake_lhotse)
    monkeypatch.setitem(sys.modules, "lhotse.shar", fake_lhotse_shar)


def _install_fake_pyarrow(monkeypatch, table):
    fake_pyarrow = types.ModuleType("pyarrow")
    fake_ipc = types.ModuleType("pyarrow.ipc")
    fake_ipc.open_stream = lambda _: _FakeArrowReader(table)
    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_parquet.ParquetFile = lambda _: _FakeParquetFile(table.to_rows())
    fake_pyarrow.ipc = fake_ipc
    fake_pyarrow.parquet = fake_parquet
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.ipc", fake_ipc)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)
def _install_common_worker_patches(monkeypatch, module, helper_calls):
    if hasattr(module, "_EXTERNAL_METADATA"):
        monkeypatch.setattr(module, "_EXTERNAL_METADATA", None)
    monkeypatch.setattr(module, "check_worker_reuse", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "init_worker_process", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "apply_audio_pipeline",
        lambda cut, **kwargs: (cut, False, None),
    )
    monkeypatch.setattr(
        module,
        "write_worker_result",
        lambda **kwargs: {
            "written": kwargs["written"],
            "skipped": kwargs["skipped"],
            "errors": kwargs["errors"],
            "total_duration_sec": kwargs["total_duration_sec"],
            "worker_stats": {"runtime_counts": dict(kwargs["runtime_counts"])},
        },
    )

    def fake_build_recording(
        audio_bytes,
        recording_id,
        *,
        runtime_counts=None,
    ):
        helper_calls.append(
            {
                "audio_bytes": audio_bytes,
                "recording_id": recording_id,
            }
        )
        if runtime_counts is not None:
            runtime_counts["recording_from_bytes"] += 1

        cut = types.SimpleNamespace(
            id=str(recording_id),
            recording_id=str(recording_id),
            duration=1.5,
            sampling_rate=16000,
            num_channels=1,
            supervisions=[],
            custom=None,
        )
        return types.SimpleNamespace(to_cut=lambda: cut)

    monkeypatch.setattr(module, "build_recording_from_audio_bytes", fake_build_recording)


def test_extract_row_metadata_prefix_and_derived_custom():
    row_id, text, lang, custom = extract_row_metadata(
        {
            "video_id": "abc123",
            "text": "ignored",
            "labels": ["/m/foo"],
        },
        id_column="video_id",
        id_prefix="audioset_unbal_train",
        text_column=None,
        custom_columns=("video_id", "labels"),
        constant_custom={"dataset": "audioset"},
        derived_custom={"source_url": "https://www.youtube.com/watch?v={video_id}"},
    )

    assert row_id == "audioset_unbal_train_abc123"
    assert text is None
    assert lang is None
    assert custom == {
        "dataset": "audioset",
        "video_id": "abc123",
        "labels": ["/m/foo"],
        "source_url": "https://www.youtube.com/watch?v=abc123",
    }


def _hf_worker_args(tmp_path, **overrides):
    payload = dict(
        worker_id=0,
        input_paths=("dataset.arrow",),
        shar_dir=str(tmp_path / "shar"),
        target_sr=None,
        shard_size=100,
        shar_format="flac",
        id_column="id",
        id_prefix=None,
        audio_column="audio",
        text_column="text",
        duration_column=None,
        language_column=None,
        language=None,
        custom_columns=None,
        constant_custom=None,
        derived_custom=None,
        text_tokenize_custom_columns=None,
        text_tokenizer_path=None,
        resampling_backend=None,
        input_clip_id_parser_name=None,
        source_id_column=None,
        clip_num_column=None,
        clip_start_column=None,
        clip_end_column=None,
        clip_duration_column=None,
        read_batch_size=8,
    )
    payload.update(overrides)
    return ColumnarWorkerArgs(**payload)


def _parquet_worker_args(tmp_path, **overrides):
    payload = dict(
        worker_id=0,
        input_paths=("dataset.parquet",),
        shar_dir=str(tmp_path / "shar"),
        target_sr=None,
        shard_size=100,
        shar_format="flac",
        id_column="id",
        id_prefix=None,
        audio_column="audio",
        text_column="text",
        duration_column="duration",
        language_column=None,
        language=None,
        custom_columns=None,
        constant_custom=None,
        derived_custom=None,
        text_tokenize_custom_columns=None,
        text_tokenizer_path=None,
        resampling_backend=None,
        input_clip_id_parser_name=None,
        source_id_column=None,
        clip_num_column=None,
        clip_start_column=None,
        clip_end_column=None,
        clip_duration_column=None,
        read_batch_size=8,
    )
    payload.update(overrides)
    return ColumnarWorkerArgs(**payload)


def test_prepare_hf_worker_uses_shared_audio_bytes_helper(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "id": ["clip-1"],
                "audio": [{"bytes": b"arrow-bytes"}],
                "text": ["hello from hf"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_hf_to_shar, helper_calls)

    result = prepare_hf_to_shar._convert_worker(_hf_worker_args(tmp_path))

    assert helper_calls == [
        {
            "audio_bytes": b"arrow-bytes",
            "recording_id": "clip-1",
        }
    ]
    assert result["written"] == 1
    assert result["worker_stats"]["runtime_counts"]["recording_from_bytes"] == 1
    assert len(written_cuts) == 1
    assert written_cuts[0].id == "clip-1"
    assert written_cuts[0].supervisions[0].id == "clip-1"
    assert written_cuts[0].supervisions[0].text == "hello from hf"


def test_prepare_hf_worker_parser_uses_row_ids_not_row_index_as_chunks(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "id": ["row00000_seg003", "row00000_seg004"],
                "audio": [{"bytes": b"first"}, {"bytes": b"second"}],
                "text": ["first text", "second text"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_hf_to_shar, helper_calls)

    result = prepare_hf_to_shar._convert_worker(
        _hf_worker_args(
            tmp_path,
            input_clip_id_parser_name="spc",
        )
    )

    assert result["written"] == 2
    assert result["errors"] == 0
    assert [cut.custom["interleave"] for cut in written_cuts] == [
        {
            "source_id": "row00000",
            "clip_num": 3,
            "clip_start": None,
            "clip_duration": None,
        },
        {
            "source_id": "row00000",
            "clip_num": 4,
            "clip_start": None,
            "clip_duration": None,
        },
    ]


def test_prepare_hf_worker_external_metadata_overrides_text(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "id": ["clip-1"],
                "audio": [{"bytes": b"arrow-bytes"}],
                "text": ["row text"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_hf_to_shar, helper_calls)
    monkeypatch.setattr(
        prepare_hf_to_shar,
        "_EXTERNAL_METADATA",
        {"clip-1": ("external text", {"speaker": "ext"})},
    )

    result = prepare_hf_to_shar._convert_worker(
        _hf_worker_args(tmp_path, id_prefix="audioset_bal_train")
    )

    assert result["written"] == 1
    assert written_cuts[0].id == "audioset_bal_train_clip-1"
    assert written_cuts[0].supervisions[0].text == "external text"
    assert written_cuts[0].custom == {
        "speaker": "ext",
        "interleave": {
            "source_id": "audioset_bal_train_clip-1",
            "clip_num": 0,
            "clip_start": None,
            "clip_duration": None,
        },
    }


def test_prepare_hf_worker_loads_text_tokenizer_from_worker_args(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    seen = {}
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "id": ["clip-1"],
                "audio": [{"bytes": b"arrow-bytes"}],
                "text": ["row text"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_hf_to_shar, helper_calls)

    tokenizer = object()
    tokenize_fn = object()
    monkeypatch.setattr(prepare_hf_to_shar, "load_text_tokenizer", lambda path: seen.setdefault("path", path) and tokenizer)
    monkeypatch.setattr(
        prepare_hf_to_shar,
        "make_text_tokenize_fn",
        lambda tok, extra=None: seen.update({"tokenizer": tok, "extra": extra}) or tokenize_fn,
    )
    monkeypatch.setattr(
        prepare_hf_to_shar,
        "apply_audio_pipeline",
        lambda cut, **kwargs: (seen.setdefault("tokenize_fn", kwargs["tokenize_fn"]), (cut, False, None))[1],
    )

    result = prepare_hf_to_shar._convert_worker(
        _hf_worker_args(
            tmp_path,
            text_tokenizer_path="/tmp/tokenizer.json",
            text_tokenize_custom_columns=("speaker",),
        )
    )

    assert result["written"] == 1
    assert seen == {
        "path": "/tmp/tokenizer.json",
        "tokenizer": tokenizer,
        "extra": ("speaker",),
        "tokenize_fn": tokenize_fn,
    }


def test_prepare_parquet_worker_uses_shared_audio_bytes_helper(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "id": ["row-1"],
                "audio": [{"bytes": b"parquet-bytes"}],
                "text": ["hello from parquet"],
                "duration": [1.5],
                "lang": ["tr"],
                "speaker": ["narrator"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            language_column="lang",
            custom_columns=("speaker",),
        )
    )

    assert helper_calls == [
        {
            "audio_bytes": b"parquet-bytes",
            "recording_id": "row-1",
        }
    ]
    assert result["written"] == 1
    assert result["worker_stats"]["runtime_counts"]["recording_from_bytes"] == 1
    assert len(written_cuts) == 1
    assert written_cuts[0].id == "row-1"
    assert written_cuts[0].supervisions[0].id == "row-1"
    assert written_cuts[0].supervisions[0].text == "hello from parquet"
    assert written_cuts[0].supervisions[0].language == "tr"
    assert written_cuts[0].custom == {
        "speaker": "narrator",
        "interleave": {
            "source_id": "row-1",
            "clip_num": 0,
            "clip_start": None,
            "clip_duration": None,
        },
    }


def test_prepare_parquet_worker_accepts_binary_audio_column(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "clip_id": ["common_voice_en_1.mp3"],
                "audio_bytes": [b"mp3-bytes"],
                "sentence": ["hello from common voice"],
                "locale": ["en"],
                "split": ["train"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            audio_column="audio_bytes",
            id_column="clip_id",
            text_column="sentence",
            duration_column=None,
            language_column="locale",
            custom_columns=("split",),
        )
    )

    assert helper_calls == [
        {
            "audio_bytes": b"mp3-bytes",
            "recording_id": "common_voice_en_1.mp3",
        }
    ]
    assert result["written"] == 1
    assert written_cuts[0].supervisions[0].text == "hello from common voice"
    assert written_cuts[0].supervisions[0].language == "en"
    assert written_cuts[0].custom["split"] == "train"


def test_prepare_parquet_worker_external_metadata_overrides_text_and_custom(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "id": ["row-1"],
                "audio": [{"bytes": b"parquet-bytes"}],
                "text": ["row text"],
                "duration": [1.5],
                "lang": ["tr"],
                "speaker": ["row-speaker"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)
    monkeypatch.setattr(
        prepare_parquet_to_shar,
        "_EXTERNAL_METADATA",
        {"row-1": ("external text", {"speaker": "ext-speaker", "topic": "budget"})},
    )

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            language_column="lang",
            custom_columns=("speaker",),
        )
    )

    assert result["written"] == 1
    assert written_cuts[0].supervisions[0].text == "external text"
    assert written_cuts[0].custom == {
        "speaker": "ext-speaker",
        "topic": "budget",
        "interleave": {
            "source_id": "row-1",
            "clip_num": 0,
            "clip_start": None,
            "clip_duration": None,
        },
    }


def test_prepare_parquet_worker_loads_text_tokenizer_from_worker_args(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    seen = {}
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "id": ["row-1"],
                "audio": [{"bytes": b"parquet-bytes"}],
                "text": ["row text"],
                "duration": [1.5],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    tokenizer = object()
    tokenize_fn = object()
    monkeypatch.setattr(prepare_parquet_to_shar, "load_text_tokenizer", lambda path: seen.setdefault("path", path) and tokenizer)
    monkeypatch.setattr(
        prepare_parquet_to_shar,
        "make_text_tokenize_fn",
        lambda tok, extra=None: seen.update({"tokenizer": tok, "extra": extra}) or tokenize_fn,
    )
    monkeypatch.setattr(
        prepare_parquet_to_shar,
        "apply_audio_pipeline",
        lambda cut, **kwargs: (seen.setdefault("tokenize_fn", kwargs["tokenize_fn"]), (cut, False, None))[1],
    )

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            text_tokenizer_path="/tmp/tokenizer.json",
            text_tokenize_custom_columns=("speaker",),
        )
    )

    assert result["written"] == 1
    assert seen == {
        "path": "/tmp/tokenizer.json",
        "tokenizer": tokenizer,
        "extra": ("speaker",),
        "tokenize_fn": tokenize_fn,
    }


def test_prepare_parquet_worker_applies_input_clip_id_parser(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "segment_id": ["row00000_seg003"],
                "audio": [{"bytes": b"parquet-bytes"}],
                "text": ["hello from spc"],
                "duration": [1.5],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            id_column="segment_id",
            input_clip_id_parser_name="spc",
        )
    )

    assert result["written"] == 1
    assert helper_calls[0]["recording_id"] == "row00000_seg003"
    assert written_cuts[0].id == "row00000_seg003"
    assert written_cuts[0].supervisions[0].id == "row00000_seg003"
    assert written_cuts[0].custom == {
        "interleave": {
            "source_id": "row00000",
            "clip_num": 3,
            "clip_start": None,
            "clip_duration": None,
        },
    }


def test_prepare_parquet_worker_parser_uses_row_ids_not_row_index_as_chunks(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "segment_id": ["row00000_seg003", "row00000_seg004"],
                "audio": [{"bytes": b"first"}, {"bytes": b"second"}],
                "text": ["first text", "second text"],
                "duration": [1.5, 1.5],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            id_column="segment_id",
            input_clip_id_parser_name="spc",
        )
    )

    assert result["written"] == 2
    assert result["errors"] == 0
    assert [cut.custom["interleave"] for cut in written_cuts] == [
        {
            "source_id": "row00000",
            "clip_num": 3,
            "clip_start": None,
            "clip_duration": None,
        },
        {
            "source_id": "row00000",
            "clip_num": 4,
            "clip_start": None,
            "clip_duration": None,
        },
    ]


def test_prepare_parquet_worker_nested_audio_path_with_basename_parser(monkeypatch, tmp_path):
    """Farsi/Infore2 flow: id_column=audio.path + trailing_number_basename parser.

    Covers the full nested-field → basename-stripping → clip_num flow through
    the parquet worker, locking in dotted-path resolution + directory prefix
    handling.
    """
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "audio": [{"bytes": b"farsi-bytes", "path": "radio_program/foo_042.wav"}],
                "transcription": ["سلام دنیا"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            id_column="audio.path",
            text_column="transcription",
            duration_column=None,
            language="fa",
            input_clip_id_parser_name="trailing_number_basename",
        )
    )

    assert result["written"] == 1
    # Audio path reaches the recording builder unchanged
    assert helper_calls[0]["audio_bytes"] == b"farsi-bytes"
    # recording_id is the full audio.path string (path/foo_042.wav)
    assert helper_calls[0]["recording_id"] == "radio_program/foo_042.wav"
    # cut.id stays as the original row id; interleave identity is nested in custom.
    assert written_cuts[0].id == "radio_program/foo_042.wav"
    assert written_cuts[0].supervisions[0].text == "سلام دنیا"
    assert written_cuts[0].supervisions[0].language == "fa"
    assert written_cuts[0].custom == {
        "interleave": {
            "source_id": "foo",
            "clip_num": 42,
            "clip_start": None,
            "clip_duration": None,
        },
    }


def test_prepare_parquet_worker_populates_clip_timestamps_from_metadata_columns(
    monkeypatch, tmp_path
):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "audio": [{"bytes": b"podcast-bytes", "path": "podcasts/episode_007.wav"}],
                "transcription": ["hello"],
                "parent_start": [12.5],
                "parent_end": [18.25],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            id_column="audio.path",
            text_column="transcription",
            duration_column=None,
            language="en",
            input_clip_id_parser_name="trailing_number_basename",
            clip_start_column="parent_start",
            clip_end_column="parent_end",
        )
    )

    assert result["written"] == 1
    assert helper_calls[0]["recording_id"] == "podcasts/episode_007.wav"
    assert written_cuts[0].custom == {
        "interleave": {
            "source_id": "episode",
            "clip_num": 7,
            "clip_start": 12.5,
            "clip_duration": 5.75,
        },
    }


def test_prepare_parquet_worker_uses_source_column_with_timestamp_order(
    monkeypatch, tmp_path
):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "id": ["clip-a"],
                "audio": [{"bytes": b"audio-bytes"}],
                "text": ["hello"],
                "original_audio_id": ["episode-1"],
                "parent_start": [12.5],
                "parent_end": [18.25],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            source_id_column="original_audio_id",
            clip_start_column="parent_start",
            clip_end_column="parent_end",
        )
    )

    assert result["written"] == 1
    assert written_cuts[0].id == "clip-a"
    expected_clip_num = derive_timestamp_clip_num(
        row_id="clip-a",
        clip_start=12.5,
        clip_duration=5.75,
    )
    assert written_cuts[0].custom == {
        "interleave": {
            "source_id": "episode-1",
            "clip_num": expected_clip_num,
            "clip_start": 12.5,
            "clip_duration": 5.75,
        },
    }


def test_extract_interleave_identity_derives_stable_timestamp_tie_breaker():
    row = {"original_audio_id": "episode-1"}

    first = extract_interleave_identity(
        row,
        row_id="clip-a",
        source_id_column="original_audio_id",
        clip_start=12.5,
        clip_duration=5.75,
    )
    second = extract_interleave_identity(
        row,
        row_id="clip-a",
        source_id_column="original_audio_id",
        clip_start=12.5,
        clip_duration=5.75,
    )
    different_row = extract_interleave_identity(
        row,
        row_id="clip-b",
        source_id_column="original_audio_id",
        clip_start=12.5,
        clip_duration=5.75,
    )

    assert first == second
    assert first[0] == "episode-1"
    assert isinstance(first[1], int)
    assert 0 <= first[1] < (1 << 63)
    assert different_row[1] != first[1]


def test_extract_interleave_identity_rejects_source_column_without_clip_num_or_timestamp():
    with pytest.raises(ValueError, match="requires clip_start_column"):
        extract_interleave_identity(
            {"original_audio_id": "episode-1"},
            row_id="clip-a",
            source_id_column="original_audio_id",
        )


def test_extract_clip_timestamps_rejects_non_positive_clip_duration():
    with pytest.raises(ValueError, match="must be > 0"):
        extract_clip_timestamps(
            {"parent_start": 12.5, "segment_duration": 0.0},
            clip_start_column="parent_start",
            clip_duration_column="segment_duration",
        )


def test_extract_clip_timestamps_rejects_clip_end_before_clip_start():
    with pytest.raises(ValueError, match="must be >="):
        extract_clip_timestamps(
            {"parent_start": 12.5, "parent_end": 11.0},
            clip_start_column="parent_start",
            clip_end_column="parent_end",
        )


def test_extract_clip_timestamps_rejects_non_finite_values():
    with pytest.raises(ValueError, match="must be finite"):
        extract_clip_timestamps(
            {"parent_start": float("nan")},
            clip_start_column="parent_start",
        )


def test_prepare_parquet_worker_rejects_invalid_clip_interval(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "audio": [{"bytes": b"podcast-bytes", "path": "podcasts/episode_007.wav"}],
                "transcription": ["hello"],
                "parent_start": [12.5],
                "parent_end": [11.0],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            id_column="audio.path",
            text_column="transcription",
            duration_column=None,
            language="en",
            input_clip_id_parser_name="trailing_number_basename",
            clip_start_column="parent_start",
            clip_end_column="parent_end",
        )
    )

    assert result["written"] == 0
    assert result["errors"] == 1
    assert result["worker_stats"]["runtime_counts"]["processing_errors"] == 1
    assert written_cuts == []


def test_prepare_parquet_worker_missing_optional_duration_column(monkeypatch, tmp_path):
    """Missing optional duration columns should behave like absent metadata."""
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "audio": [{"bytes": b"infore2-bytes", "path": "books/chapter_022.wav"}],
                "transcription": ["xin chao"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            id_column="audio.path",
            text_column="transcription",
            language="vi",
            input_clip_id_parser_name="trailing_number_basename",
        )
    )

    assert result["written"] == 1
    assert result["errors"] == 0
    assert helper_calls[0]["recording_id"] == "books/chapter_022.wav"
    assert written_cuts[0].id == "books/chapter_022.wav"


def test_prepare_parquet_worker_non_numeric_duration_counts_as_error(monkeypatch, tmp_path):
    written_cuts = []
    helper_calls = []
    _install_fake_lhotse(monkeypatch, written_cuts)
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "audio": [{"bytes": b"infore2-bytes", "path": "books/chapter_022.wav"}],
                "transcription": ["xin chao"],
                "duration": ["1.5"],
            }
        ),
    )
    _install_common_worker_patches(monkeypatch, prepare_parquet_to_shar, helper_calls)

    result = prepare_parquet_to_shar._convert_worker(
        _parquet_worker_args(
            tmp_path,
            id_column="audio.path",
            text_column="transcription",
            language="vi",
            input_clip_id_parser_name="trailing_number_basename",
        )
    )

    assert result["written"] == 0
    assert result["errors"] == 1
    assert result["worker_stats"]["runtime_counts"]["processing_errors"] == 1
    assert helper_calls == []
    assert written_cuts == []


def test_prepare_parquet_preflight_checks_runtime_and_logs_missing_optional_columns(
    monkeypatch, caplog, tmp_path
):
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "audio": [{"bytes": b"parquet-bytes", "path": "books/chapter_022.wav"}],
                "transcription": ["xin chao"],
            }
        ),
    )
    calls = []

    monkeypatch.setattr(
        prepare_parquet_to_shar,
        "validate_prepare_runtime",
        lambda **kwargs: calls.append(kwargs),
    )

    spec = types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            audio_column="audio",
            id_column="audio.path",
            text_column="transcription",
            duration_column="duration",
            language_column=None,
            custom_columns=None,
        ),
        output=types.SimpleNamespace(
            resampling_backend="soxr",
            text_tokenizer=str(tmp_path / "tokenizer.json"),
        ),
    )

    with caplog.at_level("INFO"):
        prepare_parquet_to_shar._preflight_prepare(spec, ["dataset.parquet"])

    assert calls == [
        {
            "resampling_backend": "soxr",
            "require_ffmpeg": True,
            "text_tokenizer_path": str(tmp_path / "tokenizer.json"),
        }
    ]
    assert "optional column roots missing" in caplog.text
    assert "duration" in caplog.text


def test_prepare_parquet_preflight_raises_for_missing_required_audio_column(monkeypatch):
    _install_fake_pyarrow(
        monkeypatch,
        _FakeArrowTable(
            {
                "id": ["row-1"],
                "text": ["hello"],
            }
        ),
    )
    monkeypatch.setattr(
        prepare_parquet_to_shar,
        "validate_prepare_runtime",
        lambda **kwargs: None,
    )

    spec = types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            audio_column="audio",
            id_column="id",
            text_column="text",
            duration_column=None,
            language_column=None,
            custom_columns=None,
        ),
        output=types.SimpleNamespace(
            resampling_backend="soxr",
            text_tokenizer=None,
        ),
    )

    with pytest.raises(RuntimeError, match="required column roots are missing"):
        prepare_parquet_to_shar._preflight_prepare(spec, ["dataset.parquet"])


def test_validate_prepare_runtime_requires_ffmpeg(monkeypatch):
    from audio_tokenization.prepare import runtime

    init_calls = []
    tok_calls = []
    monkeypatch.setattr(runtime, "init_worker_process", lambda backend: init_calls.append(backend))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None if name == "ffmpeg" else "/bin/true")
    monkeypatch.setattr(
        "audio_tokenization.prepare.text_ops.load_text_tokenizer",
        lambda path: tok_calls.append(path),
    )

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        runtime.validate_prepare_runtime(
            resampling_backend="soxr",
            require_ffmpeg=True,
            text_tokenizer_path="/tmp/tokenizer.json",
        )

    assert init_calls == ["soxr"]
    assert tok_calls == ["/tmp/tokenizer.json"]
