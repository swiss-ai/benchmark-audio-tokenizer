import sys
import types

import pytest
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
        lambda cut, **kwargs: (cut, False),
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

    result = prepare_hf_to_shar._convert_worker(
        (
            0,
            ["dataset.arrow"],
            str(tmp_path / "shar"),
            None,
            100,
            "flac",
            "id",
            "audio",
            "text",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            8,
        )
    )

    assert helper_calls == [
        {
            "audio_bytes": b"arrow-bytes",
            "recording_id": "clip-1",
        }
    ]
    assert result["written"] == 1
    assert result["worker_stats"]["runtime_counts"]["recording_from_bytes"] == 1
    assert len(written_cuts) == 1
    assert written_cuts[0].id == "clip-1@000000"
    assert written_cuts[0].supervisions[0].id == "clip-1@000000"
    assert written_cuts[0].supervisions[0].text == "hello from hf"


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
        (
            0,
            ["dataset.arrow"],
            str(tmp_path / "shar"),
            None,
            100,
            "flac",
            "id",
            "audio",
            "text",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            8,
        )
    )

    assert result["written"] == 1
    assert written_cuts[0].supervisions[0].text == "external text"
    assert written_cuts[0].custom == {
        "speaker": "ext",
        "source_id": "clip-1",
        "clip_num": 0,
        "clip_start": 0.0,
        "legacy_cut_id": "clip-1",
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
        (
            0,
            ["dataset.parquet"],
            str(tmp_path / "shar"),
            None,
            100,
            "flac",
            "id",
            "audio",
            "text",
            "duration",
            "lang",
            None,
            ("speaker",),
            None,
            None,
            None,
            None,
            8,
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
    assert written_cuts[0].id == "row-1@000000"
    assert written_cuts[0].supervisions[0].id == "row-1@000000"
    assert written_cuts[0].supervisions[0].text == "hello from parquet"
    assert written_cuts[0].supervisions[0].language == "tr"
    assert written_cuts[0].custom == {
        "speaker": "narrator",
        "source_id": "row-1",
        "clip_num": 0,
        "clip_start": 0.0,
        "legacy_cut_id": "row-1",
    }


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
        (
            0,
            ["dataset.parquet"],
            str(tmp_path / "shar"),
            None,
            100,
            "flac",
            "id",
            "audio",
            "text",
            "duration",
            "lang",
            None,
            ("speaker",),
            None,
            None,
            None,
            None,
            8,
        )
    )

    assert result["written"] == 1
    assert written_cuts[0].supervisions[0].text == "external text"
    assert written_cuts[0].custom == {
        "speaker": "ext-speaker",
        "topic": "budget",
        "source_id": "row-1",
        "clip_num": 0,
        "clip_start": 0.0,
        "legacy_cut_id": "row-1",
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
        (
            0,
            ["dataset.parquet"],
            str(tmp_path / "shar"),
            None,
            100,
            "flac",
            "segment_id",
            "audio",
            "text",
            "duration",
            None,
            None,
            None,
            None,
            None,
            None,
            "spc",
            8,
        )
    )

    assert result["written"] == 1
    assert helper_calls[0]["recording_id"] == "row00000_seg003"
    assert written_cuts[0].id == "row00000@000003"
    assert written_cuts[0].supervisions[0].id == "row00000@000003"
    assert written_cuts[0].custom == {
        "source_id": "row00000",
        "clip_num": 3,
        "clip_start": 0.0,
        "legacy_cut_id": "row00000_seg003",
    }


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
        (
            0,
            ["dataset.parquet"],
            str(tmp_path / "shar"),
            None,
            100,
            "flac",
            "audio.path",       # dotted id_column
            "audio",
            "transcription",
            None,               # duration_column
            None,               # language_column
            "fa",               # language (global)
            None,               # custom_columns
            None,               # text_tokenize_custom_columns
            None,               # text_tokenizer
            None,               # resampling_backend
            "trailing_number_basename",  # input_clip_id_parser
            8,
        )
    )

    assert result["written"] == 1
    # Audio path reaches the recording builder unchanged
    assert helper_calls[0]["audio_bytes"] == b"farsi-bytes"
    # recording_id is the full audio.path string (path/foo_042.wav)
    assert helper_calls[0]["recording_id"] == "radio_program/foo_042.wav"
    # cut.id has source_id derived from basename (no "/"), clip_num from trailing number
    assert written_cuts[0].id == "foo@000042"
    assert "/" not in written_cuts[0].id
    assert written_cuts[0].supervisions[0].text == "سلام دنیا"
    assert written_cuts[0].supervisions[0].language == "fa"
    assert written_cuts[0].custom == {
        "source_id": "foo",
        "clip_num": 42,
        "clip_start": 0.0,
        "legacy_cut_id": "radio_program/foo_042.wav",
    }


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
        (
            0,
            ["dataset.parquet"],
            str(tmp_path / "shar"),
            None,
            100,
            "flac",
            "audio.path",
            "audio",
            "transcription",
            "duration",         # column is configured but absent in the row
            None,
            "vi",
            None,
            None,
            None,
            None,
            "trailing_number_basename",
            8,
        )
    )

    assert result["written"] == 1
    assert result["errors"] == 0
    assert helper_calls[0]["recording_id"] == "books/chapter_022.wav"
    assert written_cuts[0].id == "chapter@000022"


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
        (
            0,
            ["dataset.parquet"],
            str(tmp_path / "shar"),
            None,
            100,
            "flac",
            "audio.path",
            "audio",
            "transcription",
            "duration",
            None,
            "vi",
            None,
            None,
            None,
            None,
            "trailing_number_basename",
            8,
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

    args = types.SimpleNamespace(
        audio_column="audio",
        id_column="audio.path",
        text_column="transcription",
        duration_column="duration",
        language_column=None,
        custom_columns=None,
        resampling_backend="soxr",
        text_tokenizer=str(tmp_path / "tokenizer.json"),
    )

    with caplog.at_level("INFO"):
        prepare_parquet_to_shar._preflight_prepare(args, ["dataset.parquet"])

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

    args = types.SimpleNamespace(
        audio_column="audio",
        id_column="id",
        text_column="text",
        duration_column=None,
        language_column=None,
        custom_columns=None,
        resampling_backend="soxr",
        text_tokenizer=None,
    )

    with pytest.raises(RuntimeError, match="required column roots are missing"):
        prepare_parquet_to_shar._preflight_prepare(args, ["dataset.parquet"])


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
