from __future__ import annotations

import gzip
import json

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
