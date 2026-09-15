"""Regression coverage for preparation writer and materialization boundaries."""

import gzip
import io
import json
import tarfile
import types
import weakref
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from lhotse import Recording
from lhotse.audio.source import AudioSource
from lhotse.shar import SharWriter

from audio_tokenization.prepare import audio_ops, streaming
from audio_tokenization.tests import test_prepare_audio_bytes_workers as helpers


def _cut(channels=1):
    stream = io.BytesIO()
    # Nonconstant channels exercise summation, clipping and FLAC quantization.
    samples = np.linspace(-0.8, 0.8, 800 * channels, dtype=np.float32).reshape(800, channels)
    sf.write(stream, samples, 16000, format="WAV")
    return Recording.from_bytes(stream.getvalue(), recording_id="prepare-probe").to_cut()


@pytest.mark.parametrize("failing_field", ["recording", "cuts"])
def test_shar_append_failure_is_fatal_without_retry(monkeypatch, tmp_path, failing_field):
    cut = _cut()
    audio = cut.load_audio()
    with SharWriter(str(tmp_path), fields={"recording": "wav"}, create_index=False) as writer:
        field_writer = writer.writers[failing_field]
        original = field_writer.write
        attempts = 0

        def write_then_fail(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            original(*args, **kwargs)
            raise OSError("injected failure after append")

        monkeypatch.setattr(field_writer, "write", write_then_fail)
        with pytest.raises(RuntimeError, match="SHAR") as exc:
            audio_ops.write_cut_to_shar(writer, cut, audio=audio)
        assert isinstance(exc.value.__cause__, OSError)
        assert attempts == 1
    with tarfile.open(next(tmp_path.glob("recording.*.tar"))) as archive:
        assert len(archive.getmembers()) == 2


def test_stereo_pipeline_reuses_probe_without_changing_waveform(monkeypatch):
    cut = _cut(2)
    expected = cut.to_mono(mono_downmix=True).load_audio()
    calls = 0
    original = AudioSource.load_audio

    def count_load(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AudioSource, "load_audio", count_load)
    result, skip, audio = audio_ops.apply_audio_pipeline(cut, runtime_counts=Counter())
    assert not skip
    np.testing.assert_array_equal(audio, expected)
    assert result.custom["rms_db"] == audio_ops.rms_db_from_audio(expected)
    assert calls == 2  # One source decode and one materialized FLAC probe.


@pytest.mark.parametrize("kind", ["parquet", "arrow"])
def test_streaming_releases_materialized_rows_before_next_batch(monkeypatch, tmp_path, kind):
    table = pa.table({"id": [1, 2, 3, 4]})
    refs = []
    closed = []

    class Row(dict):
        pass

    class Batch:
        def __init__(self, batch):
            self.batch = batch
            self.num_rows = batch.num_rows

        def slice(self, start, length):
            return Batch(self.batch.slice(start, length))

        def to_pylist(self):
            assert all(ref() is None for ref in refs), "previous rows still retained"
            rows = [Row(row) for row in self.batch.to_pylist()]
            refs.extend(weakref.ref(row) for row in rows)
            return rows

    class Reader:
        def __iter__(self):
            for batch in table.to_batches(max_chunksize=2):
                yield Batch(batch)

        def iter_batches(self, **kwargs):
            return iter(self)

        def close(self):
            closed.append(True)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    if kind == "parquet":
        monkeypatch.setattr(pq, "ParquetFile", lambda _: Reader())
        rows = streaming.iter_parquet_rows("unused", columns=["id"], batch_size=2)
    else:
        monkeypatch.setattr(ipc, "open_stream", lambda _: Reader())
        rows = streaming.iter_arrow_rows("unused", batch_size=2)
    assert next(rows) == {"id": 1}
    assert next(rows) == {"id": 2}
    assert next(rows) == {"id": 3}
    rows.close()
    assert all(ref() is None for ref in refs)
    assert closed == [True]


def test_arrow_projection_preserves_nested_fields_and_optional_missing_columns(tmp_path):
    path = tmp_path / "rows.arrow"
    table = pa.table({"id": [1, 2, 3], "audio": [{"bytes": b"a", "path": "a.wav"}] * 3,
                      "unused_payload": ["large"] * 3})
    with ipc.new_stream(path, table.schema) as sink:
        sink.write_table(table)
    rows = list(streaming.iter_arrow_rows(str(path), batch_size=2,
                                         columns=["id", "audio.bytes", "missing"]))
    assert rows == [{"id": n, "audio": {"bytes": b"a", "path": "a.wav"}} for n in [1, 2, 3]]


def _setup_columnar(monkeypatch, tmp_path, *, segmented=False, kind="parquet"):
    module = helpers.prepare_parquet_to_shar if kind == "parquet" else helpers.prepare_hf_to_shar
    written = []
    helpers._install_fake_lhotse(monkeypatch, written)
    data = {"id": ["row-1"], "audio": [{"bytes": b"audio"}], "text": ["text"]}
    if segmented:
        data["chunks"] = [[{"clip_id": f"c{i}", "clip_start_sec": float(i), "clip_duration_sec": 1.0}
                           for i in range(3)]]
    helpers._install_fake_pyarrow(monkeypatch, helpers._FakeArrowTable(data))
    helpers._install_common_worker_patches(monkeypatch, module, [])

    def build(*args, **kwargs):
        def truncate(**kwargs):
            return types.SimpleNamespace(id="row-1", recording_id="row-1", duration=1.0,
                                         sampling_rate=16000, num_channels=1, custom=None, supervisions=[])
        return types.SimpleNamespace(to_cut=lambda: types.SimpleNamespace(
            id="row-1", truncate=truncate, recording_id="row-1", duration=1.0,
            sampling_rate=16000, num_channels=1, custom=None, supervisions=[]))

    monkeypatch.setattr(module, "build_recording_from_audio_bytes", build)
    args = (helpers._parquet_worker_args(tmp_path, duration_column=None,
                                         chunks_column="chunks" if segmented else None)
            if kind == "parquet" else helpers._hf_worker_args(tmp_path))
    return module, args, written


def test_parquet_chunk_decode_error_does_not_drop_later_chunks(monkeypatch, tmp_path):
    module, args, written = _setup_columnar(monkeypatch, tmp_path, segmented=True)

    def pipeline(cut, **kwargs):
        if cut.id == "c1":
            raise ValueError("injected chunk decode error")
        return cut, False, None

    monkeypatch.setattr(module, "apply_audio_pipeline", pipeline)
    result = module._convert_worker(args)
    assert [cut.id for cut in written] == ["c0", "c2"]
    assert (result["written"], result["errors"], result["skipped"]) == (2, 1, 0)


@pytest.mark.parametrize("kind,segmented", [("parquet", False), ("parquet", True), ("arrow", False)])
def test_columnar_worker_aborts_on_output_failure(monkeypatch, tmp_path, kind, segmented):
    module, args, written = _setup_columnar(monkeypatch, tmp_path, kind=kind, segmented=segmented)

    def fail(self, cut):
        written.append(cut)
        raise OSError("injected output error")

    monkeypatch.setattr(helpers._FakeSharWriter, "write", fail)
    with pytest.raises(RuntimeError, match="SHAR"):
        module._convert_worker(args)
    assert len(written) == 1
    assert not (tmp_path / "shar" / "worker_00" / "_SUCCESS").exists()


def _setup_wds(monkeypatch, tmp_path):
    from audio_tokenization.prepare import prepare_wds_to_shar as module
    written = []
    helpers._install_fake_lhotse(monkeypatch, written)
    helpers._install_common_worker_patches(monkeypatch, module, [])
    cut = types.SimpleNamespace(id="rec", recording_id="rec", sampling_rate=16000)
    chunks = [types.SimpleNamespace(id=f"c{i}", recording_id="rec", sampling_rate=16000,
                                   duration=1.0, custom={}, supervisions=[]) for i in range(3)]
    monkeypatch.setattr(module, "iter_tar_cuts", lambda *args, **kwargs: iter([cut]))
    monkeypatch.setattr(module, "load_vad_from_per_shard_dir", lambda *args, **kwargs: ({"rec": []}, {}))
    monkeypatch.setattr(module, "split_cut_by_vad", lambda **kwargs: (chunks, "split"))
    args = module.WdsWorkerArgs(worker_id=0, tar_paths=("unused.tar",), shar_dir=str(tmp_path / "shar"),
        target_sr=None, shard_size=100, shar_format="wav", min_sr=None, text_field="text",
        custom_fields=None, mono_downmix=False, vad_per_shard_dir=str(tmp_path),
        vad_max_chunk_sec=200.0, vad_min_chunk_sec=1.0, vad_sample_rate=16000,
        vad_max_merge_gap_sec=0.5, vad_max_duration_sec=None, text_tokenizer=None,
        resampling_backend=None, input_clip_id_parser_name=None, language=None)
    return module, args, written


def test_wds_chunk_decode_error_does_not_drop_later_chunks(monkeypatch, tmp_path):
    module, args, written = _setup_wds(monkeypatch, tmp_path)

    def pipeline(cut, **kwargs):
        if cut.id == "c1":
            raise ValueError("injected chunk decode error")
        return cut, False, None

    monkeypatch.setattr(module, "apply_audio_pipeline", pipeline)
    result = module._convert_worker(args)
    assert [cut.id for cut in written] == ["c0", "c2"]
    assert (result["written"], result["errors"], result["skipped"]) == (2, 1, 0)


def test_wds_worker_aborts_on_output_failure(monkeypatch, tmp_path):
    module, args, written = _setup_wds(monkeypatch, tmp_path)

    def fail(self, cut):
        written.append(cut)
        raise OSError("injected output error")

    monkeypatch.setattr(helpers._FakeSharWriter, "write", fail)
    with pytest.raises(RuntimeError, match="SHAR"):
        module._convert_worker(args)
    assert len(written) == 1


@pytest.mark.parametrize("kind", ["parquet", "arrow"])
def test_columnar_projection_keeps_derived_custom_source_fields(monkeypatch, tmp_path, kind):
    from dataclasses import replace
    module, args, written = _setup_columnar(monkeypatch, tmp_path, kind=kind)
    helpers._install_fake_pyarrow(monkeypatch, helpers._FakeArrowTable({
        "id": ["row-1"], "audio": [{"bytes": b"audio"}], "raw_speaker": ["ada"]}))
    result = module._convert_worker(replace(args, derived_custom={"speaker": "{raw_speaker}"}))
    assert result["written"] == 1
    assert written[0].custom["speaker"] == "ada"


def test_audio_dir_worker_aborts_on_output_failure(monkeypatch, tmp_path):
    import lhotse.shar
    from audio_tokenization.tests import test_prepare_audio_dir_to_shar as directory_helpers
    module = directory_helpers.prepare_audio_dir_to_shar
    wav = tmp_path / "clip.wav"
    vad = tmp_path / "vad.jsonl"
    directory_helpers._write_wav(wav, np.full(12 * 16000, 0.1))
    directory_helpers._write_vad_jsonl(vad)
    written = []

    class FailingWriter(helpers._FakeSharWriter):
        def write(self, cut):
            written.append(cut)
            raise OSError("injected output error")

    monkeypatch.setattr(lhotse.shar, "SharWriter", lambda **kwargs: FailingWriter(sink=written))
    with pytest.raises(RuntimeError, match="SHAR"):
        module._convert_worker(directory_helpers._worker_args(tmp_path, vad, wav))
    assert len(written) == 1
    assert not (tmp_path / "shar" / "worker_00" / "_SUCCESS").exists()


@pytest.mark.parametrize("capability_fallback", [False, True])
def test_decoded_shar_write_matches_public_writer(monkeypatch, tmp_path, capability_fallback):
    import lhotse.shar.utils
    cut = _cut(2).to_mono(mono_downmix=False)[1].truncate(offset=0.01, duration=0.02)
    audio = cut.load_audio()
    expected = tmp_path / "public"
    actual = tmp_path / "reused"
    expected.mkdir()
    actual.mkdir()
    with SharWriter(str(expected), fields={"recording": "wav"}, create_index=False) as writer:
        writer.write(cut)
    if capability_fallback:
        def unsupported(*args, **kwargs):
            raise AttributeError("injected unavailable adapter capability")
        # SharWriter holds its own original import; only adapter preflight fails.
        monkeypatch.setattr(lhotse.shar.utils, "to_shar_placeholder", unsupported)
    counts = Counter()
    with SharWriter(str(actual), fields={"recording": "wav"}, create_index=False) as writer:
        audio_ops.write_cut_to_shar(writer, cut, audio=audio, runtime_counts=counts)
    assert next(expected.glob("recording.*.tar")).read_bytes() == next(actual.glob("recording.*.tar")).read_bytes()
    with gzip.open(next(expected.glob("cuts.*.gz")), "rt") as source:
        expected_cut = json.load(source)
    with gzip.open(next(actual.glob("cuts.*.gz")), "rt") as source:
        assert json.load(source) == expected_cut
    assert counts["decoded_audio_write_fallback"] == int(capability_fallback)
    assert counts["reused_decoded_audio_for_shar_write"] == int(not capability_fallback)


def test_downmix_probe_failure_still_uses_first_channel(monkeypatch):
    cut = _cut(2)
    expected = cut.to_mono(mono_downmix=False)[0].load_audio()
    original = audio_ops.load_audio_quietly
    attempts = 0

    def fail_probe(cut, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("injected failed downmix probe")
        return original(cut, *args, **kwargs)

    monkeypatch.setattr(audio_ops, "load_audio_quietly", fail_probe)
    counts = Counter()
    result, skip, audio = audio_ops.apply_audio_pipeline(cut, runtime_counts=counts)
    assert not skip
    assert result.channel == 0
    assert counts["downmix_fallback_ch0"] == 1
    np.testing.assert_array_equal(audio, expected)
