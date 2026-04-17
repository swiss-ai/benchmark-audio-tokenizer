"""Tests for validate_shar.py and the postprocess tombstones."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_test_shar(tmp_path: Path, num_cuts: int = 2) -> Path:
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

    cuts = []
    for i in range(num_cuts):
        wav_path = audio_dir / f"src-{i:04d}.wav"
        sf.write(str(wav_path), samples, sr)
        rec = Recording.from_file(wav_path, recording_id=f"rec-{i:04d}")
        cuts.append(
            MonoCut(
                id=f"cut-{i:04d}",
                start=0.0,
                duration=duration,
                channel=0,
                recording=rec,
            )
        )

    out_dir = tmp_path / "shar"
    out_dir.mkdir(parents=True, exist_ok=True)
    with SharWriter(
        str(out_dir), fields={"recording": "wav"}, shard_size=num_cuts
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
    # Two valid failure paths depending on Lhotse variant:
    #   - upstream Lhotse: hard assert "Mismatched IDs" during iteration
    #     (wrapped as the original error; cuts_consumed == 0).
    #   - dev Lhotse (patched at lazy.py:294-300): warn + skip the bad cut,
    #     so the iterator yields cut-0001 only. Validator's consumed-vs-
    #     expected check then fires with "silently skipping" in the message.
    msg = str(ei.value)
    assert ("Mismatched IDs" in msg) or ("silently skipping" in msg)


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


# ---------------------------------------------------------------------------
# Tombstones — execution-time, not import-time
# ---------------------------------------------------------------------------


# Only patch_universal_ids is tombstoned on this branch. add_captions_to_shar
# carries its own boundary check (_assert_id_stability_for_symlinked_recordings)
# that rejects the unsafe rewrite path while still allowing the safe
# stable-IDs use case, so it stays runnable.
_TOMBSTONED_MODULES = [
    "audio_tokenization.prepare.postprocess.patch_universal_ids",
]


@pytest.mark.parametrize("module", _TOMBSTONED_MODULES)
def test_tombstone_imports_cleanly(module):
    """Importing must succeed so sibling tooling and test discovery don't break."""
    __import__(module)


@pytest.mark.parametrize("module", _TOMBSTONED_MODULES)
def test_tombstone_execution_exits_nonzero(module):
    result = _run_module(module)
    assert result.returncode != 0
    assert "deprecated" in result.stderr.lower()
    assert "validate_shar" in result.stderr
