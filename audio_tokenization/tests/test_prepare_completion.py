"""Prepare completion reuses validation without weakening restart checks."""

import json
import multiprocessing
import shutil
import types
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pytest
from lhotse import fastcopy
from lhotse.shar import SharWriter

from audio_tokenization.pipelines.lhotse import planning
from audio_tokenization.prepare import runtime, validate_shar
from audio_tokenization.prepare.atomic_shar import atomic_shar_partition
from audio_tokenization.stages.convert import _ensure_convert_shar_manifest
from audio_tokenization.tests.test_prepare_integrity import _cut


class _InlinePool:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap_unordered(self, worker, args):
        return map(worker, args)


def _count_cut_reads(monkeypatch):
    counts = Counter()
    original_rows = validate_shar._iter_jsonl_rows
    original_open = planning.open_compressed

    def rows(path):
        for item in original_rows(path):
            if Path(path).name.startswith("cuts."):
                counts["structural"] += 1
            yield item

    @contextmanager
    def opened(path, *args, **kwargs):
        with original_open(path, *args, **kwargs) as source:
            def lines():
                for line in source:
                    if line.strip():
                        counts["planning"] += 1
                    yield line
            yield lines()

    monkeypatch.setattr(validate_shar, "_iter_jsonl_rows", rows)
    monkeypatch.setattr(planning, "open_compressed", opened)
    monkeypatch.setattr(validate_shar, "resolve_num_workers", lambda *a, **k: 1)
    monkeypatch.setattr(planning, "_resolve_scan_workers", lambda *a: 1)
    return counts


def _write_partition(root, worker_id, *, atomic=True):
    directory = root / f"worker_{worker_id:02d}"
    cut = _cut()
    cut.custom = {"rms_db": -12.0, "interleave": {"source_id": "rec", "clip_num": 2,
                  "clip_start": 0.0, "clip_duration": cut.duration}}

    def write(destination):
        with SharWriter(str(destination), fields={"recording": "wav"},
                        shard_size=2, create_index=False) as writer:
            for i in range(3):
                writer.write(fastcopy(cut, id=f"cut-{worker_id}-{i}"))

    if atomic:
        with atomic_shar_partition(directory, fields=("recording",)) as staged:
            write(staged)
    else:
        directory.mkdir(parents=True)
        write(directory)
    runtime.mark_partition_success(directory)
    return {"worker_id": worker_id, "written": 3, "skipped": 0, "errors": 0,
            "total_duration_sec": 3 * cut.duration}


def _complete(monkeypatch, root, worker, ids=(0,)):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: types.SimpleNamespace(
        Pool=lambda processes: _InlinePool()))
    runtime.run_pool_and_finalize(worker, list(ids), root, len(ids), mp_start_method="fork")
    _ensure_convert_shar_manifest(types.SimpleNamespace(
        family="audio_dir", output=types.SimpleNamespace(shar_dir=str(root))))
    return planning.read_shar_work_manifest(root)


def test_fresh_completion_parses_cuts_once_and_preserves_relocatable_manifest(monkeypatch, tmp_path):
    root = tmp_path / "shar"
    counts = _count_cut_reads(monkeypatch)
    manifest = _complete(monkeypatch, root, lambda worker: _write_partition(root, worker), ids=(0, 2))
    assert counts == {"structural": 6}, "fresh completion must parse each cut exactly once"
    payload = json.loads((root / planning.SHAR_WORK_MANIFEST_FILE).read_text())
    assert payload["schema_version"] == 3
    assert payload["total_cut_count"] == 6
    assert payload["total_duration_sec"] == pytest.approx(6 * _cut().duration)
    assert payload["rms_db_count"] == 6
    assert payload["sample_rate_count"] == 6
    assert payload["source_id_count"] == 6
    assert manifest.to_json() == planning.build_shar_work_manifest(str(root)).to_json()
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    relocated = planning.read_shar_work_manifest(copied)
    assert relocated.to_json() == planning.build_shar_work_manifest(str(copied)).to_json()
    assert all(str(copied) in p for unit in relocated.work_units for paths in unit.fields.values() for p in paths)


def test_legacy_partition_gets_one_structural_metadata_scan(monkeypatch, tmp_path):
    root = tmp_path / "shar"
    result = _write_partition(root, 0, atomic=False)
    counts = _count_cut_reads(monkeypatch)
    manifest = _complete(monkeypatch, root, lambda worker: result)
    assert counts == {"structural": 3}
    assert manifest.to_json() == planning.build_shar_work_manifest(str(root)).to_json()


@pytest.mark.parametrize("changed_field", ["cuts", "recording"])
def test_changed_partition_cannot_reuse_validation(monkeypatch, tmp_path, changed_field):
    root = tmp_path / "shar"
    result = _write_partition(root, 0)
    target = next((root / "worker_00").glob(f"{changed_field}.*"))
    original = target.stat()
    # Keep size and mtime unchanged: replacement/ctime still invalidates evidence.
    replacement = target.with_name("replacement")
    replacement.write_bytes(b"x" * original.st_size)
    import os
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(target)
    with pytest.raises(validate_shar.SharValidationError):
        _complete(monkeypatch, root, lambda worker: result)
    assert not (root / planning.SHAR_WORK_MANIFEST_FILE).exists()
    assert not (root / "_SUCCESS").exists()


def test_unchanged_completion_reuses_evidence_but_standalone_validator_rescans(monkeypatch, tmp_path):
    root = tmp_path / "shar"
    counts = _count_cut_reads(monkeypatch)
    _complete(monkeypatch, root, lambda worker: _write_partition(root, worker))
    counts.clear()
    assert sum(runtime.finalize_shar_directory(root).values()) == 3
    assert counts == {}
    assert sum(validate_shar.validate_shar_directory(root, num_workers=1).values()) == 3
    assert counts == {"structural": 3}


def test_copied_completion_revalidates_once_then_reuses_root_evidence(monkeypatch, tmp_path):
    original = tmp_path / "original"
    _complete(monkeypatch, original, lambda worker: _write_partition(original, worker))
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    counts = _count_cut_reads(monkeypatch)
    assert sum(runtime.finalize_shar_directory(copied).values()) == 3
    assert counts == {"structural": 3}
    counts.clear()
    assert sum(runtime.finalize_shar_directory(copied).values()) == 3
    assert counts == {}


@pytest.mark.parametrize("change", ["missing_shard", "duplicate_shard", "field_pairing"])
def test_root_index_must_match_validated_partition_coverage(tmp_path, change):
    root = tmp_path / "shar"
    _write_partition(root, 0)
    runtime.build_shar_index(root, num_workers=1)
    index_path = root / "shar_index.json"
    index = json.loads(index_path.read_text())
    if change == "missing_shard":
        for paths in index["fields"].values():
            paths.pop()
    elif change == "duplicate_shard":
        for paths in index["fields"].values():
            paths.append(paths[0])
    else:
        index["fields"]["recording"].reverse()
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="cover|pairing"):
        runtime.finalize_shar_directory(root)
    assert not (root / planning.SHAR_WORK_MANIFEST_FILE).exists()


def test_parallel_legacy_validation_preserves_all_planning_metadata(monkeypatch, tmp_path):
    root = tmp_path / "shar"
    _write_partition(root, 0, atomic=False)
    runtime.build_shar_index(root, num_workers=1)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert sum(runtime.finalize_shar_directory(root).values()) == 3
    manifest = planning.read_shar_work_manifest(root)
    assert manifest.to_json() == planning.build_shar_work_manifest(str(root)).to_json()
    assert manifest.to_json()["source_id_count"] == 3


def test_recipe_completion_preserves_custom_index_name_without_rescanning(monkeypatch, tmp_path):
    from audio_tokenization.prepare.prepare_lhotse_recipe_to_shar import (
        build_shar_index, convert_worker,
    )
    counts = _count_cut_reads(monkeypatch)
    convert_worker(0, [_cut()], shar_dir=tmp_path, shar_format="wav", shar_shard_size=100,
                   min_sample_rate=None, target_sample_rate=None)
    build_shar_index(tmp_path, "recipe_index.json", 1)
    assert sum(runtime.finalize_shar_directory(tmp_path, index_filename="recipe_index.json").values()) == 1
    _ensure_convert_shar_manifest(types.SimpleNamespace(
        family="lhotse_recipe", output=types.SimpleNamespace(shar_dir=str(tmp_path)),
        input=types.SimpleNamespace(shar_index_filename="recipe_index.json")))
    assert counts == {"structural": 1}
    manifest = planning.read_shar_work_manifest(tmp_path)
    assert manifest.shar_index_filename == "recipe_index.json"
    assert manifest.to_json() == planning.build_shar_work_manifest(
        str(tmp_path), index_name="recipe_index.json").to_json()


@pytest.mark.parametrize("atomic", [False, True])
def test_completion_normalizes_valid_relative_index_paths(monkeypatch, tmp_path, atomic):
    root = tmp_path / "shar"
    _write_partition(root, 0, atomic=atomic)
    runtime.build_shar_index(root, num_workers=1)
    index_path = root / "shar_index.json"
    index = json.loads(index_path.read_text())
    index["fields"] = {field: [f"worker_00/../{path}" for path in paths]
                       for field, paths in index["fields"].items()}
    index_path.write_text(json.dumps(index))
    counts = _count_cut_reads(monkeypatch)
    assert sum(runtime.finalize_shar_directory(root).values()) == 3
    assert counts == ({} if atomic else {"structural": 3})
    assert planning.read_shar_work_manifest(root).to_json() == planning.build_shar_work_manifest(str(root)).to_json()
