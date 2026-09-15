"""Latest-Lhotse preparation uses pipeline-owned atomic publication."""

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest
from lhotse import CutSet
from lhotse.shar import SharWriter

from audio_tokenization.prepare import runtime, validate_shar
from audio_tokenization.tests.test_prepare_integrity import _cut
from audio_tokenization.tests.test_validate_shar import _write_test_shar


def test_latest_lhotse_public_api_passes_preflight():
    runtime.validate_prepare_runtime(resampling_backend="default")


def test_metadata_validation_never_extracts_audio(monkeypatch, tmp_path):
    shar = _write_test_shar(tmp_path)
    original = tarfile.TarFile.extractfile
    extracted = []

    def extract(self, member):
        name = member.name if isinstance(member, tarfile.TarInfo) else member
        assert name.endswith(".json"), "audio payload must not be extracted"
        extracted.append(name)
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", extract)
    assert sum(validate_shar.validate_shar_directory(shar, num_workers=1).values()) == 2
    assert extracted == ["cut-0000.json", "cut-0001.json"]


@pytest.mark.parametrize("kind", ["symlink", "nonempty_placeholder", "truncated_metadata"])
def test_metadata_validation_rejects_malformed_tar_members(tmp_path, kind):
    shar = _write_test_shar(tmp_path, num_cuts=1)
    path = shar / "recording.000000.tar"
    if kind == "truncated_metadata":
        with tarfile.open(path) as source:
            metadata = source.getmembers()[1]
        with path.open("r+b") as dest:
            dest.truncate(metadata.offset_data + 5)
    else:
        with tarfile.open(path, "w") as dest:
            if kind == "symlink":
                info = tarfile.TarInfo("cut-0000.wav")
                info.type = tarfile.SYMTYPE
                info.linkname = "other.wav"
                dest.addfile(info)
                info = tarfile.TarInfo("cut-0000.json")
                info.size = 2
                dest.addfile(info, io.BytesIO(b"{}"))
            else:
                for name, value in [("cut-0000.nodata", b"bad"), ("cut-0000.nometa", b"")]:
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    dest.addfile(info, io.BytesIO(value))
    with pytest.raises(validate_shar.SharValidationError):
        validate_shar.validate_shar_directory(shar, num_workers=1)


def test_partition_publishes_after_close_and_validation(tmp_path):
    from audio_tokenization.prepare.atomic_shar import atomic_shar_partition
    final = tmp_path / "worker_00"
    final.mkdir()  # Existing callers precreate an empty partition.
    cut = _cut()
    with atomic_shar_partition(final, fields=("recording",)) as staged:
        assert staged.parent == final.parent
        assert staged != final
        with SharWriter(str(staged), fields={"recording": "wav"}, create_index=False) as writer:
            writer.write(cut)
        assert list(final.iterdir()) == []
    assert not staged.exists()
    assert [c.id for c in CutSet.from_shar(in_dir=str(final))] == [cut.id]
    # Pipeline helper doesn't publish a worker/stage success marker itself.
    assert not (final / "_SUCCESS").exists()


@pytest.mark.parametrize("failure", ["body", "validation", "close", "rename"])
def test_partition_failure_keeps_destination_unpublished(monkeypatch, tmp_path, failure):
    from audio_tokenization.prepare import atomic_shar
    final = tmp_path / "worker_00"
    final.mkdir()
    if failure == "rename":
        def fail_rename(*args):
            raise OSError("injected rename failure")
        monkeypatch.setattr(atomic_shar.os, "replace", fail_rename)
    with pytest.raises(Exception):
        with atomic_shar.atomic_shar_partition(final, fields=("recording",)) as staged:
            with SharWriter(str(staged), fields={"recording": "wav"}, create_index=False) as writer:
                writer.write(_cut())
                if failure == "body":
                    raise ValueError("injected body failure")
                if failure == "close":
                    original = writer.close
                    def fail_close():
                        original()
                        raise OSError("injected close failure")
                    monkeypatch.setattr(writer, "close", fail_close)
            if failure == "validation":
                with gzip.open(next(staged.glob("cuts.*.gz")), "wt") as dest:
                    dest.write('{"id":"bad"}\n')
    assert list(final.iterdir()) == []
    assert not staged.exists()


def test_partition_refuses_nonempty_destination(tmp_path):
    from audio_tokenization.prepare.atomic_shar import atomic_shar_partition
    final = tmp_path / "worker_00"
    final.mkdir()
    (final / "existing").write_text("keep")
    with pytest.raises(FileExistsError):
        with atomic_shar_partition(final, fields=("recording",)):
            pytest.fail("must fail before starting a writer")
    assert (final / "existing").read_text() == "keep"


def test_empty_partition_and_restart_leave_unrelated_staging_untouched(tmp_path):
    from audio_tokenization.prepare.atomic_shar import atomic_shar_partition
    stale = tmp_path / ".worker_00.staging-old"
    stale.mkdir()
    (stale / "partial").write_text("stale")
    final = tmp_path / "worker_00"
    with atomic_shar_partition(final, fields=("recording",)) as staged:
        pass
    assert final.is_dir()
    assert list(final.iterdir()) == []
    assert not staged.exists()
    assert (stale / "partial").read_text() == "stale"


def test_recipe_public_writer_publishes_valid_partition(tmp_path):
    from audio_tokenization.prepare.prepare_lhotse_recipe_to_shar import convert_worker
    cut = _cut()
    convert_worker(0, [cut], shar_dir=tmp_path, shar_format="wav", shar_shard_size=100,
                   min_sample_rate=None, target_sample_rate=None)
    final = tmp_path / "part-00000"
    assert (final / "_SUCCESS").exists()
    restored = list(CutSet.from_shar(fields={
        "cuts": sorted(str(p) for p in final.glob("cuts.*.jsonl.gz")),
        "recording": sorted(str(p) for p in final.glob("recording.*.tar")),
    }))
    assert [c.id for c in restored] == [cut.id]


@pytest.mark.parametrize("kind", ["parquet", "arrow", "wds"])
def test_real_byte_workers_publish_valid_relocatable_waveforms(tmp_path, kind):
    from dataclasses import replace
    import numpy as np
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    from audio_tokenization.tests import test_prepare_audio_bytes_workers as helpers
    from audio_tokenization.prepare import prepare_wds_to_shar

    input_cut = _cut()
    encoded = input_cut.recording.sources[0].source
    expected = input_cut.load_audio()
    if kind == "wds":
        path = tmp_path / "input.tar"
        with tarfile.open(path, "w") as dest:
            for name, payload in [("clip.wav", encoded), ("clip.json", b'{"text":"hello"}')]:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                dest.addfile(info, io.BytesIO(payload))
        args = prepare_wds_to_shar.WdsWorkerArgs(worker_id=0, tar_paths=(str(path),),
            shar_dir=str(tmp_path / "shar"), target_sr=None, shard_size=100,
            shar_format="wav", min_sr=None, text_field="text", custom_fields=None,
            mono_downmix=True, vad_per_shard_dir=None, vad_max_chunk_sec=200.0,
            vad_min_chunk_sec=1.0, vad_sample_rate=16000, vad_max_merge_gap_sec=0.5,
            vad_max_duration_sec=None, text_tokenizer=None, resampling_backend="default",
            input_clip_id_parser_name=None, language=None)
        result = prepare_wds_to_shar._convert_worker(args)
    else:
        table = pa.table({"id": ["clip"], "audio": [{"bytes": encoded}], "text": ["hello"]})
        path = tmp_path / f"input.{kind}"
        if kind == "parquet":
            pq.write_table(table, path)
            module = helpers.prepare_parquet_to_shar
            args = helpers._parquet_worker_args(tmp_path, duration_column=None)
        else:
            with ipc.new_stream(path, table.schema) as dest:
                dest.write_table(table)
            module = helpers.prepare_hf_to_shar
            args = helpers._hf_worker_args(tmp_path)
        result = module._convert_worker(replace(args, input_paths=(str(path),),
            resampling_backend="default", shar_format="wav"))
    assert (result["written"], result["errors"]) == (1, 0)
    root = tmp_path / "shar"
    runtime.build_shar_index(root, num_workers=1)
    assert sum(validate_shar.validate_shar_directory(root, num_workers=1).values()) == 1
    index = json.loads((root / "shar_index.json").read_text())["fields"]
    restored = list(CutSet.from_shar(fields={field: [str(root / p) for p in paths]
                                           for field, paths in index.items()}))
    assert [cut.id for cut in restored] == ["clip"]
    np.testing.assert_array_equal(restored[0].load_audio(), expected)
    assert not list(root.glob(".*.staging-*"))
    assert not (root / "_SUCCESS").exists()


def test_partition_missing_field_is_not_published(tmp_path):
    from audio_tokenization.prepare.atomic_shar import atomic_shar_partition
    final = tmp_path / "worker_00"
    with pytest.raises(ValueError, match="different shards"):
        with atomic_shar_partition(final, fields=("recording",)) as staged:
            with SharWriter(str(staged), fields={"recording": "wav"}, create_index=False) as writer:
                writer.write(_cut())
            next(staged.glob("recording.*.tar")).unlink()
    assert not final.exists()
    assert not staged.exists()


def test_partition_preserves_precreated_directory_permissions(tmp_path):
    import stat
    from audio_tokenization.prepare.atomic_shar import atomic_shar_partition
    final = tmp_path / "worker_00"
    final.mkdir()
    final.chmod(0o750)  # Set the fixture independently of the runner's umask.
    with atomic_shar_partition(final, fields=("recording",)):
        pass
    assert stat.S_IMODE(final.stat().st_mode) == 0o750
