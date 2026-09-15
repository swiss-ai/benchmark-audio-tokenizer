from __future__ import annotations

import json
import logging

import pytest
from lhotse import CutSet, MonoCut
from lhotse.indexing import create_jsonl_index

from audio_tokenization.pipelines.lhotse.data import _load_shar_cutset


def _write_paired_shar(root, *, indexed=False):
    root.mkdir()
    fields = {"cuts": [], "tag": []}
    for shard, tag_name in enumerate(["tag.z.jsonl", "tag.a.jsonl"]):
        cut_path = root / f"cuts.{shard}.jsonl"
        tag_path = root / tag_name
        cuts = [MonoCut(id=f"cut-{shard}-{i}", start=0, duration=i + 1, channel=0) for i in range(3)]
        CutSet.from_cuts(cuts).to_file(cut_path)
        tag_path.write_text("".join(json.dumps({"cut_id": cut.id, "tag": cut.id}) + "\n" for cut in cuts))
        fields["cuts"].append(str(cut_path))
        fields["tag"].append(str(tag_path))
        if indexed:
            create_jsonl_index(cut_path)
            create_jsonl_index(tag_path)
    (root / "shar_index.json").write_text(json.dumps({
        "fields": {name: [path.rsplit("/", 1)[-1] for path in paths] for name, paths in fields.items()}
    }))
    return fields


@pytest.mark.parametrize("planned", [True, False])
def test_loader_preserves_declared_companion_pairs(tmp_path, planned):
    root = tmp_path / "shar"
    fields = _write_paired_shar(root)

    cuts = _load_shar_cutset(
        input_shar_dirs=[str(root)],
        planned_shar_fields=fields if planned else None,
        index_name="shar_index.json",
        rank=0,
    )

    assert [(cut.id, cut.tag) for cut in cuts] == [
        ("cut-1-0", "cut-1-0"), ("cut-1-1", "cut-1-1"), ("cut-1-2", "cut-1-2"),
        ("cut-0-0", "cut-0-0"), ("cut-0-1", "cut-0-1"), ("cut-0-2", "cut-0-2"),
    ]


@pytest.mark.parametrize("indexed", [True, False])
def test_loader_reports_native_reader_without_changing_order_or_rank_ownership(tmp_path, caplog, monkeypatch, indexed):
    root = tmp_path / "shar"
    fields = _write_paired_shar(root, indexed=indexed)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.delenv("LHOTSE_USE_WORKER_PARTITION", raising=False)

    with caplog.at_level(logging.INFO):
        cuts = _load_shar_cutset(
            input_shar_dirs=[str(root)], planned_shar_fields=fields,
            index_name="shar_index.json", rank=1, world_size=4,
        )

    # Preserve the existing native auto reader's ordering, including its
    # distinct streaming shard shuffle and indexed cut shuffle semantics.
    reference = CutSet.from_shar(fields=fields, split_for_dataloading=False, shuffle_shards=True)
    actual = [(cut.id, cut.tag) for cut in cuts]
    assert actual == [(cut.id, cut.tag) for cut in reference]
    assert len(actual) == 6
    assert cuts.is_indexed is indexed
    assert f"reader_mode={'indexed' if indexed else 'streaming'}" in caplog.text
