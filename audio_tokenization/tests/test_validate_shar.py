"""Tests for validate_shar.py."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest


def _write_test_shar(
    tmp_path: Path,
    num_cuts: int = 2,
    *,
    shard_size: int | None = None,
    cut_ids: list[str] | None = None,
) -> Path:
    """Write a tiny SHAR (one shard, num_cuts cuts) plus shar_index.json.

    Builds cuts from real on-disk wav files so SharWriter doesn't trip on
    inline-bytes serialization that the dummy_cut(with_data=True) fixtures
    exercise via custom fields.
    """
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, MonoCut, Recording
    from lhotse.shar.writers.shar import SharWriter

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sr = 16000
    duration = 0.5
    samples = np.zeros(int(sr * duration), dtype=np.float32)

    if cut_ids is not None:
        num_cuts = len(cut_ids)

    cuts = []
    for i in range(num_cuts):
        wav_path = audio_dir / f"src-{i:04d}.wav"
        sf.write(str(wav_path), samples, sr)
        rec = Recording.from_file(wav_path, recording_id=f"rec-{i:04d}")
        cuts.append(
            MonoCut(
                id=cut_ids[i] if cut_ids is not None else f"cut-{i:04d}",
                start=0.0,
                duration=duration,
                channel=0,
                recording=rec,
            )
        )

    out_dir = tmp_path / "shar"
    out_dir.mkdir(parents=True, exist_ok=True)
    with SharWriter(
        str(out_dir),
        fields={"recording": "wav"},
        shard_size=num_cuts if shard_size is None else shard_size,
    ) as writer:
        for cut in CutSet.from_cuts(cuts):
            writer.write(cut)

    cut_shards = sorted(p.name for p in out_dir.glob("cuts.*.jsonl.gz"))
    rec_shards = sorted(p.name for p in out_dir.glob("recording.*.tar"))
    index = {
        "version": 1,
        "fields": {"cuts": cut_shards, "recording": rec_shards},
    }
    (out_dir / "shar_index.json").write_text(json.dumps(index, indent=2))
    return out_dir


def _read_cuts(cuts_path: Path) -> list[dict]:
    with gzip.open(cuts_path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_cuts(cuts_path: Path, cuts: list[dict]) -> None:
    with gzip.open(cuts_path, "wt") as f:
        for c in cuts:
            f.write(json.dumps(c) + "\n")


def _add_jsonl_sidecar(
    shar_dir: Path,
    *,
    field_name: str = "caption",
    rows: list[dict] | None = None,
) -> Path:
    sidecar_path = shar_dir / f"{field_name}.000000.jsonl.gz"
    if rows is None:
        rows = [
            {"cut_id": "cut-0000", field_name: "caption 0"},
            {"cut_id": "cut-0001", field_name: "caption 1"},
        ]
    with gzip.open(sidecar_path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    index_path = shar_dir / "shar_index.json"
    index = json.loads(index_path.read_text())
    index["fields"][field_name] = [sidecar_path.name]
    index_path.write_text(json.dumps(index, indent=2))
    return sidecar_path


# ---------------------------------------------------------------------------
# Library API
# ---------------------------------------------------------------------------


def test_validate_passes_on_clean_shar(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    counts = validate_shar_directory(shar)

    assert sum(counts.values()) == 2
    assert len(counts) == 1
    assert next(iter(counts.keys())).endswith("cuts.000000.jsonl.gz")


def test_validate_passes_on_clean_shar_with_jsonl_sidecar(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    _add_jsonl_sidecar(shar)

    counts = validate_shar_directory(shar)
    assert sum(counts.values()) == 2


def test_validate_passes_on_clean_shar_with_path_like_cut_ids(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        validate_shar_directory,
    )

    shar = _write_test_shar(
        tmp_path,
        cut_ids=["speaker_a/cut-0000", "speaker_b/cut-0001"],
    )

    counts = validate_shar_directory(shar)
    assert sum(counts.values()) == 2


def test_validate_raises_on_id_mismatch(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    cuts_path = shar / "cuts.000000.jsonl.gz"
    cuts = _read_cuts(cuts_path)
    cuts[0]["id"] = "definitely-not-the-tar-stem"
    _write_cuts(cuts_path, cuts)

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert ei.value.shard_name.endswith("cuts.000000.jsonl.gz")
    # Structural lockstep catches this on the first pair.
    msg = str(ei.value)
    assert "out of lockstep" in msg


def test_validate_raises_on_dropped_cut(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    cuts_path = shar / "cuts.000000.jsonl.gz"
    cuts = _read_cuts(cuts_path)
    _write_cuts(cuts_path, cuts[:1])

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    # First cut iterates fine; mismatch is detected when reading a 2-entry
    # tar against a 1-entry cuts manifest. Either an extra-tar-entry error or
    # an id-mismatch on the second pair surfaces as a validation failure.
    assert ei.value.shard_name.endswith("cuts.000000.jsonl.gz")


def test_validate_runs_structural_check_on_every_shard(tmp_path, monkeypatch):
    """Contract: structural check is exhaustive — no sampling, no skipped
    shards. Regression guard against re-introducing deep-validation sampling
    that weakens the _SUCCESS guarantee."""
    import audio_tokenization.prepare.validate_shar as validate_mod

    shar = _write_test_shar(tmp_path, num_cuts=10, shard_size=1)
    seen = []

    real = validate_mod._validate_structural_shard
    def spy(*, shard_name, slice_fields):
        seen.append(Path(shard_name).name)
        return real(shard_name=shard_name, slice_fields=slice_fields)

    monkeypatch.setattr(validate_mod, "_validate_structural_shard", spy)
    # num_workers=1 forces the inline single-process path so the spy
    # patched into the parent module is observable.
    counts = validate_mod.validate_shar_directory(shar, num_workers=1)

    assert len(counts) == 10
    assert len(seen) == 10


def test_validate_parallel_path_returns_same_counts(tmp_path):
    """Parallel-pool path must produce the same per-shard counts as the
    inline path. Guards against the multiprocessing wrapper dropping or
    reordering results.
    """
    from audio_tokenization.prepare.validate_shar import validate_shar_directory

    shar = _write_test_shar(tmp_path, num_cuts=12, shard_size=2)
    inline = validate_shar_directory(shar, num_workers=1)
    parallel = validate_shar_directory(shar, num_workers=4)
    assert inline == parallel
    assert sum(inline.values()) == 12


def test_validate_wraps_malformed_jsonl_in_validation_error(tmp_path):
    """P2 regression: a malformed cuts.*.jsonl.gz used to leak raw
    json.JSONDecodeError out of validate_shar_directory, bypassing the
    CLI's `except SharValidationError` handler. Now wrapped with shard
    context preserved."""
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    cuts_path = shar / "cuts.000000.jsonl.gz"
    with gzip.open(cuts_path, "wt") as f:
        f.write('{"id": "broken_cut"\n')  # missing closing brace

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert ei.value.shard_name.endswith("cuts.000000.jsonl.gz")
    # Walk __cause__ chain to find the root cause: SharValidationError ←
    # _StructuralReadError ← JSONDecodeError. The intermediate wrap exists
    # because inner readers don't have shard context.
    root = ei.value.original
    while getattr(root, "__cause__", None) is not None:
        root = root.__cause__
    assert isinstance(root, json.JSONDecodeError)


def test_validate_raises_on_cut_manifest_with_unknown_type(tmp_path):
    """Catches malformed cut manifests that pass JSON parsing but fail
    Lhotse's deserialize dispatch — a row with an unrecognized 'type' or
    missing required Cut fields. Without full deserialization this used to
    pass validation and crash later in CutSet.from_shar()."""
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    cuts_path = shar / "cuts.000000.jsonl.gz"
    cuts = _read_cuts(cuts_path)
    # Valid JSON with bogus discriminator → deserialize_item dispatch fails.
    cuts[0] = {"id": "x", "type": "NotARealCutType"}
    _write_cuts(cuts_path, cuts)

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert ei.value.shard_name.endswith("cuts.000000.jsonl.gz")
    assert "deserialize" in str(ei.value).lower()


def test_validate_cuts_only_shar_still_deserializes_cut_manifests(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    index_path = shar / "shar_index.json"
    index = json.loads(index_path.read_text())
    index["fields"] = {"cuts": index["fields"]["cuts"]}
    index_path.write_text(json.dumps(index, indent=2))

    cuts_path = shar / "cuts.000000.jsonl.gz"
    cuts = _read_cuts(cuts_path)
    cuts[0] = {"id": "x", "type": "NotARealCutType"}
    _write_cuts(cuts_path, cuts)

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert "deserialize" in str(ei.value).lower()


def test_validate_wraps_nested_cut_manifest_key_error(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    cuts_path = shar / "cuts.000000.jsonl.gz"
    cuts = _read_cuts(cuts_path)
    del cuts[0]["recording"]["sources"]
    _write_cuts(cuts_path, cuts)

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    root = ei.value.original
    while getattr(root, "__cause__", None) is not None:
        root = root.__cause__
    assert isinstance(root, KeyError)


def test_validate_raises_on_invalid_tar_metadata_json(tmp_path):
    """Catches a recording.*.tar where the per-cut JSON metadata blob is
    structurally tar-valid but its JSON payload is malformed. Without
    metadata deserialization this used to pass validation and crash later
    in CutSet.from_shar() downstream. Distinct from the truncated-tar test,
    which exercises tar-header corruption."""
    import tarfile as tarfile_mod
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    tar_path = shar / "recording.000000.tar"

    # Rebuild the tar with the .json metadata members corrupted.
    with tarfile_mod.open(tar_path, mode="r:") as src:
        members = list(src)
        member_data = [(m, src.extractfile(m).read()) for m in members]

    with tarfile_mod.open(tar_path, mode="w") as dst:
        for m, data in member_data:
            if m.name.endswith(".json"):
                data = b"{not valid json"
                m.size = len(data)
            dst.addfile(m, BytesIO(data))

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert ei.value.shard_name.endswith("cuts.000000.jsonl.gz")
    assert "deserialize" in str(ei.value).lower()


def test_validate_raises_on_invalid_shar_placeholder_metadata(tmp_path):
    """Regression: deserialize_item alone is not enough for tar metadata.
    The real reader also calls fill_shar_placeholder(), which rejects some
    manifest shapes that are valid JSON + valid Lhotse objects but illegal as
    SHAR placeholders (e.g. Recording with multiple sources)."""
    import tarfile as tarfile_mod
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    tar_path = shar / "recording.000000.tar"

    with tarfile_mod.open(tar_path, mode="r:") as src:
        members = list(src)
        member_data = [(m, src.extractfile(m).read()) for m in members]

    with tarfile_mod.open(tar_path, mode="w") as dst:
        for m, data in member_data:
            if m.name.endswith(".json"):
                payload = json.loads(data)
                payload["sources"].append(dict(payload["sources"][0]))
                data = json.dumps(payload).encode("utf-8")
                m.size = len(data)
            dst.addfile(m, BytesIO(data))

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert "placeholder" in str(ei.value).lower() or "single" in str(ei.value).lower()


def test_validate_raises_on_two_metadata_members_per_pair(tmp_path):
    """Regression: a tar pair where both members are metadata (e.g. two
    .json members for the same cut, or .json + .nometa) used to silently
    yield a stem because the old check accepted "at least one side is
    meta." The pair-invariant assert now rejects it."""
    import tarfile as tarfile_mod
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    tar_path = shar / "recording.000000.tar"

    # Replace the data side (.wav) of cut 0 with a second .json,
    # producing a corrupt .json + .json pair.
    with tarfile_mod.open(tar_path, mode="r:") as src:
        member_data = [(m, src.extractfile(m).read()) for m in list(src)]
    rewritten = []
    for m, data in member_data:
        if m.name == "cut-0000.wav":
            new_info = tarfile_mod.TarInfo("cut-0000.json")
            new_info.size = len(b"{}")
            rewritten.append((new_info, b"{}"))
        else:
            rewritten.append((m, data))
    with tarfile_mod.open(tar_path, mode="w") as dst:
        for m, data in rewritten:
            dst.addfile(m, BytesIO(data))

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert "both/neither metadata" in str(ei.value)


def test_validate_raises_on_tar_pair_metadata_before_data(tmp_path):
    import tarfile as tarfile_mod
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    tar_path = shar / "recording.000000.tar"

    with tarfile_mod.open(tar_path, mode="r:") as src:
        member_data = [(m, src.extractfile(m).read()) for m in list(src)]

    by_name = {m.name: (m, data) for m, data in member_data}
    reordered = [
        by_name["cut-0000.json"],
        by_name["cut-0000.wav"],
        *[
            pair
            for pair in member_data
            if pair[0].name not in {"cut-0000.json", "cut-0000.wav"}
        ],
    ]
    with tarfile_mod.open(tar_path, mode="w") as dst:
        for m, data in reordered:
            dst.addfile(m, BytesIO(data))

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert "data-then-metadata" in str(ei.value)


def test_validate_raises_on_tar_pair_with_wrong_parent_directory(tmp_path):
    import tarfile as tarfile_mod
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(
        tmp_path,
        cut_ids=["a/cut-0000", "b/cut-0001"],
    )
    tar_path = shar / "recording.000000.tar"

    with tarfile_mod.open(tar_path, mode="r:") as src:
        member_data = [(m, src.extractfile(m).read()) for m in list(src)]

    rewritten = []
    for m, data in member_data:
        if m.name in {"a/cut-0000.wav", "a/cut-0000.json"}:
            new_name = m.name.replace("a/", "wrong/")
            new_info = tarfile_mod.TarInfo(new_name)
            new_info.size = len(data)
            rewritten.append((new_info, data))
        else:
            rewritten.append((m, data))
    with tarfile_mod.open(tar_path, mode="w") as dst:
        for m, data in rewritten:
            dst.addfile(m, BytesIO(data))

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert "out of lockstep" in str(ei.value)


def test_validate_passes_on_gzipped_tar_field(tmp_path):
    """SHAR fields can be stored as recording.*.tar.gz (e.g. after
    merge_shar.py with kind='tar.gz'). The validator must read them via
    mode='r:gz', not the default 'r:' which trips ReadError on a valid
    gzipped tar."""
    import tarfile as tarfile_mod
    from audio_tokenization.prepare.validate_shar import validate_shar_directory

    shar = _write_test_shar(tmp_path, num_cuts=2)
    plain_tar = shar / "recording.000000.tar"
    gz_tar = shar / "recording.000000.tar.gz"

    # Recompress in-place: read the plain tar, write a gzipped equivalent.
    with tarfile_mod.open(plain_tar, mode="r:") as src:
        members = list(src)
        member_data = [(m, src.extractfile(m).read()) for m in members]
    with tarfile_mod.open(gz_tar, mode="w:gz") as dst:
        for m, data in member_data:
            dst.addfile(m, BytesIO(data))
    plain_tar.unlink()

    # Update the index to point at the .tar.gz field.
    index_path = shar / "shar_index.json"
    index = json.loads(index_path.read_text())
    index["fields"]["recording"] = ["recording.000000.tar.gz"]
    index_path.write_text(json.dumps(index, indent=2))

    counts = validate_shar_directory(shar)
    assert sum(counts.values()) == 2


def test_validate_raises_on_jsonl_sidecar_out_of_lockstep(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    _add_jsonl_sidecar(
        shar,
        rows=[
            {"cut_id": "cut-0001", "caption": "caption 1"},
            {"cut_id": "cut-0000", "caption": "caption 0"},
        ],
    )

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert "out of lockstep" in str(ei.value)


def test_validate_raises_on_jsonl_sidecar_missing_cut_id(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    _add_jsonl_sidecar(
        shar,
        rows=[
            {"caption": "caption 0"},
            {"cut_id": "cut-0001", "caption": "caption 1"},
        ],
    )

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert "no usable 'cut_id'" in str(ei.value)


def test_validate_raises_on_jsonl_sidecar_length_mismatch(tmp_path):
    """Sidecar has fewer rows than the cuts manifest — symmetric to the
    tar-side dropped-cut test. Catches producer bugs that drop sidecar
    rows or merge an under-sized sidecar against a fuller cuts file."""
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    _add_jsonl_sidecar(
        shar,
        rows=[{"cut_id": "cut-0000", "caption": "caption 0"}],
    )

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert "lockstep broke" in str(ei.value)


def test_validate_wraps_truncated_tar_in_validation_error(tmp_path):
    """P2 regression: a truncated recording.*.tar used to surface a raw
    tarfile.ReadError; now wrapped as SharValidationError."""
    from audio_tokenization.prepare.validate_shar import (
        SharValidationError,
        validate_shar_directory,
    )

    shar = _write_test_shar(tmp_path, num_cuts=2)
    tar_path = shar / "recording.000000.tar"
    # Truncate to 100 bytes — well under the first tar header (512 B).
    with open(tar_path, "r+b") as f:
        f.truncate(100)

    with pytest.raises(SharValidationError) as ei:
        validate_shar_directory(shar)
    assert ei.value.shard_name.endswith("cuts.000000.jsonl.gz")


def test_count_jsonl_entries_handles_gzipped(tmp_path):
    """Regression for P2: jsonl sidecar fields (.jsonl.gz) must be counted
    via gzip-aware reader, not sent through tarfile.open."""
    from audio_tokenization.prepare.validate_shar import (
        _count_jsonl_entries,
    )
    p = tmp_path / "captions.000000.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write('{"a":1}\n{"b":2}\n{"c":3}\n')
    assert _count_jsonl_entries(p) == 3


def test_count_jsonl_entries_handles_plain(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        _count_jsonl_entries,
    )
    p = tmp_path / "captions.000000.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n')
    assert _count_jsonl_entries(p) == 2


def test_validate_raises_on_missing_index(tmp_path):
    from audio_tokenization.prepare.validate_shar import (
        validate_shar_directory,
    )

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="shar_index.json"):
        validate_shar_directory(empty)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_module(module: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *extra],
        capture_output=True,
        text=True,
    )


def test_validate_cli_prints_per_shard_counts(tmp_path):
    shar = _write_test_shar(tmp_path, num_cuts=2)
    result = _run_module(
        "audio_tokenization.prepare.validate_shar",
        "--shar-dir", str(shar),
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout
    assert "2 cuts across 1 shards" in result.stdout
    assert "cuts.000000.jsonl.gz" in result.stdout


def test_validate_cli_accepts_custom_index_filename(tmp_path):
    """prepare_lhotse_recipe_to_shar can emit a non-default index name; the
    CLI must let operators validate those SHARs."""
    shar = _write_test_shar(tmp_path, num_cuts=2)
    (shar / "shar_index.json").rename(shar / "custom_index.json")

    result = _run_module(
        "audio_tokenization.prepare.validate_shar",
        "--shar-dir", str(shar),
        "--index-filename", "custom_index.json",
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_validate_cli_exits_nonzero_on_corruption(tmp_path):
    shar = _write_test_shar(tmp_path, num_cuts=2)
    cuts_path = shar / "cuts.000000.jsonl.gz"
    cuts = _read_cuts(cuts_path)
    cuts[0]["id"] = "broken"
    _write_cuts(cuts_path, cuts)

    result = _run_module(
        "audio_tokenization.prepare.validate_shar",
        "--shar-dir", str(shar),
    )
    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "cuts.000000.jsonl.gz" in result.stderr
