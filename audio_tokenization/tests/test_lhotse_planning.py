from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from audio_tokenization.pipelines.lhotse.planning import (
    SHAR_WORK_MANIFEST_FILE,
    TokenizeFilter,
    build_shar_work_manifest,
    build_tokenize_assignment,
    load_or_build_shar_work_manifest,
    read_shar_work_manifest,
    write_shar_work_manifest,
)


def _write_shar(
    root,
    shard_durations,
    *,
    include_rms: bool = True,
    include_interleave_ids: bool = True,
    include_timestamps: bool = True,
):
    root.mkdir(parents=True, exist_ok=True)
    cuts = []
    recordings = []
    for shard_idx, durations in enumerate(shard_durations):
        cut_name = f"cuts.{shard_idx:06d}.jsonl.gz"
        rec_name = f"recordings.{shard_idx:06d}.jsonl.gz"
        cuts.append(cut_name)
        recordings.append(rec_name)
        with gzip.open(root / cut_name, "wt") as f:
            for cut_idx, duration in enumerate(durations):
                cut = {
                    "id": f"cut-{shard_idx}-{cut_idx}",
                    "duration": duration,
                    "recording": {"sampling_rate": 24000},
                }
                custom = {}
                if include_rms:
                    custom["rms_db"] = -20.0
                if include_interleave_ids:
                    interleave = {
                        "source_id": f"source-{shard_idx}",
                        "clip_num": cut_idx,
                    }
                    if include_timestamps:
                        interleave["clip_start"] = float(cut_idx)
                        interleave["clip_duration"] = float(duration)
                    custom["interleave"] = interleave
                if custom:
                    cut["custom"] = custom
                f.write(json.dumps(cut) + "\n")
        with gzip.open(root / rec_name, "wt") as f:
            f.write("{}\n")
    (root / "shar_index.json").write_text(
        json.dumps({"fields": {"cuts": cuts, "recordings": recordings}}) + "\n"
    )


def test_shar_work_manifest_keeps_unfiltered_duration_for_assignment(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[1.0, 3.0], [10.0]])

    manifest = build_shar_work_manifest(
        str(shar),
        tokenize_filter=TokenizeFilter(min_duration=2.0, max_duration=20.0),
    )

    assert len(manifest.work_units) == 2
    assert manifest.to_json()["total_cut_count"] == 3
    assert manifest.to_json()["total_duration_sec"] == 14.0
    assert manifest.to_json()["rms_db_count"] == 3
    assert manifest.to_json()["sample_rate_count"] == 3
    assert manifest.to_json()["clip_duration_count"] == 3
    assert manifest.work_units[0].fields.keys() == {"cuts", "recordings"}


def test_write_shar_work_manifest_is_filter_independent(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[1.0, 3.0], [10.0]])

    manifest = write_shar_work_manifest(shar)
    reloaded = read_shar_work_manifest(shar)

    assert (shar / SHAR_WORK_MANIFEST_FILE).is_file()
    assert manifest.to_json()["total_cut_count"] == 3
    assert reloaded.to_json()["total_cut_count"] == 3
    assert reloaded.to_json()["total_duration_sec"] == 14.0


def test_load_or_build_prefers_existing_manifest_without_filter_rescan(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[1.0, 3.0], [10.0]])
    write_shar_work_manifest(shar)

    manifest, source = load_or_build_shar_work_manifest(
        str(shar),
        tokenize_filter=TokenizeFilter(min_duration=2.0),
    )

    assert source == "manifest"
    assert manifest.to_json()["total_cut_count"] == 3
    assert manifest.to_json()["total_duration_sec"] == 14.0


def test_min_rms_filter_requires_conversion_rms_metadata(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[3.0]], include_rms=False)

    with pytest.raises(ValueError, match="rms_db_count=0/1"):
        build_shar_work_manifest(
            str(shar),
            tokenize_filter=TokenizeFilter(min_rms_db=-50.0),
        )


def test_interleave_id_coverage_required_only_for_interleaved_plan(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[3.0]], include_interleave_ids=False)

    manifest = build_shar_work_manifest(str(shar), tokenize_filter=TokenizeFilter())
    assert manifest.to_json()["source_id_count"] == 0

    with pytest.raises(ValueError, match="source_id_count=0/1"):
        build_shar_work_manifest(
            str(shar),
            tokenize_filter=TokenizeFilter(),
            require_interleave_ids=True,
        )


def test_clip_num_only_interleave_plan_does_not_require_timestamps(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[3.0]], include_interleave_ids=True, include_timestamps=False)

    manifest = build_shar_work_manifest(
        str(shar),
        tokenize_filter=TokenizeFilter(),
        require_interleave_ids=True,
    )

    assert manifest.to_json()["source_id_count"] == 1
    assert manifest.to_json()["clip_num_count"] == 1
    assert manifest.to_json()["clip_start_count"] == 0
    assert manifest.to_json()["clip_duration_count"] == 0


def test_tokenize_assignment_is_duration_balanced_and_marks_inactive_ranks(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[100.0], [60.0], [40.0]])
    manifest = build_shar_work_manifest(str(shar), tokenize_filter=TokenizeFilter())

    assignment = build_tokenize_assignment(manifest, world_size=5)

    assert assignment.world_size == 5
    assert assignment.active_ranks == 3
    assert [a.active for a in assignment.assignments] == [True, True, True, False, False]
    assert assignment.assignment_for_rank(3).fields == {}
    assert sorted(
        round(a.duration_sec, 1)
        for a in assignment.assignments
        if a.active
    ) == [40.0, 60.0, 100.0]


def test_shar_work_manifest_rejects_unaligned_index_fields(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[1.0], [2.0]])
    (shar / "shar_index.json").write_text(
        json.dumps(
            {
                "fields": {
                    "cuts": ["cuts.000000.jsonl.gz", "cuts.000001.jsonl.gz"],
                    "recordings": ["recordings.000000.jsonl.gz"],
                }
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="expected 2"):
        build_shar_work_manifest(str(shar), tokenize_filter=TokenizeFilter())


@pytest.mark.parametrize("input_kind", ["list", "parent", "glob"])
def test_partial_durable_manifest_never_omits_requested_roots(tmp_path, input_kind):
    root = tmp_path / "shar"
    first, second = root / "node_0", root / "node_1"
    _write_shar(first, [[1.0]])
    _write_shar(second, [[2.0, 3.0]])
    write_shar_work_manifest(first)
    requested = {
        "list": [str(first), str(second)],
        "parent": str(root),
        "glob": str(root / "node_*"),
    }[input_kind]

    manifest, source = load_or_build_shar_work_manifest(requested)

    assert manifest.to_json()["total_cut_count"] == 3
    assert manifest.input_shar_dirs == [str(first), str(second)]
    assert {unit.shar_dir for unit in manifest.work_units} == {str(first), str(second)}
    assert source == "scan"


@pytest.mark.parametrize("keep_original", [True, False])
def test_copied_durable_manifest_resolves_only_requested_root(tmp_path, keep_original):
    original, copied = tmp_path / "original", tmp_path / "copied"
    _write_shar(original, [[1.0], [2.0]])
    write_shar_work_manifest(original)
    shutil.copytree(original, copied)
    if not keep_original:
        shutil.rmtree(original)

    manifest, source = load_or_build_shar_work_manifest(str(copied))

    assert source == "manifest"
    assert manifest.input_shar_dirs == [str(copied)]
    assert {unit.shar_dir for unit in manifest.work_units} == {str(copied)}
    assert all(
        Path(path).parent == copied
        for unit in manifest.work_units
        for paths in unit.fields.values()
        for path in paths
    )
    assert manifest.fingerprint == build_shar_work_manifest(str(copied)).fingerprint


def test_partitioned_durable_manifest_relocates_and_covers_all_children(tmp_path):
    original, copied = tmp_path / "original", tmp_path / "copied"
    _write_shar(original / "node_0", [[1.0]])
    _write_shar(original / "node_1", [[2.0, 3.0]])
    write_shar_work_manifest(original)
    shutil.copytree(original, copied)

    manifest, source = load_or_build_shar_work_manifest(str(copied))

    assert source == "manifest"
    assert manifest.input_shar_dirs == [str(copied / "node_0"), str(copied / "node_1")]
    assert manifest.to_json()["total_cut_count"] == 3


def test_legacy_absolute_manifest_is_rescanned_after_copy(tmp_path):
    original, copied = tmp_path / "original", tmp_path / "copied"
    _write_shar(original, [[1.0]])
    legacy = build_shar_work_manifest(str(original)).to_json()
    legacy["schema_version"] = 2
    (original / SHAR_WORK_MANIFEST_FILE).write_text(json.dumps(legacy))
    shutil.copytree(original, copied)

    manifest, source = load_or_build_shar_work_manifest(str(copied))

    assert source == "scan"
    assert manifest.input_shar_dirs == [str(copied)]
    assert manifest.work_units[0].fields["cuts"] == [str(copied / "cuts.000000.jsonl.gz")]


def test_changed_index_invalidates_durable_manifest(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[1.0]])
    write_shar_work_manifest(shar)
    _write_shar(shar, [[1.0], [2.0]])

    manifest, source = load_or_build_shar_work_manifest(str(shar))

    assert source == "scan"
    assert manifest.to_json()["total_cut_count"] == 2


def test_partial_parent_manifest_is_rescanned_after_new_partition(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar / "node_0", [[1.0]])
    write_shar_work_manifest(shar)
    _write_shar(shar / "node_1", [[2.0]])

    manifest, source = load_or_build_shar_work_manifest(str(shar))

    assert source == "scan"
    assert manifest.to_json()["total_cut_count"] == 2


def test_tokenize_assignment_preserves_nonlexical_companion_pairs(tmp_path):
    shar = tmp_path / "shar"
    _write_shar(shar, [[1.0], [2.0]])
    index_path = shar / "shar_index.json"
    index = json.loads(index_path.read_text())
    index["fields"]["recordings"].reverse()
    index_path.write_text(json.dumps(index))

    manifest = build_shar_work_manifest(str(shar))
    assignment = build_tokenize_assignment(manifest, world_size=1).assignment_for_rank(0)

    assert list(zip(assignment.fields["cuts"], assignment.fields["recordings"])) == [
        (str(shar / "cuts.000000.jsonl.gz"), str(shar / "recordings.000001.jsonl.gz")),
        (str(shar / "cuts.000001.jsonl.gz"), str(shar / "recordings.000000.jsonl.gz")),
    ]
    by_id = {unit.work_unit_id: unit for unit in manifest.work_units}
    assert [by_id[uid].fields["cuts"][0] for uid in assignment.work_unit_ids] == assignment.fields["cuts"]
