"""Focused tests for the shared stage-output contract."""

from __future__ import annotations

import json
import logging

import pytest

from audio_tokenization.stages._stage_runner import (
    MANIFEST_FILE,
    check_stage_output,
    run_stage,
    write_stage_manifest,
)


_LOG = logging.getLogger(__name__)


def test_run_stage_skips_when_success_present(tmp_path):
    output_dir = tmp_path / "stage"
    output_dir.mkdir()
    (output_dir / "_SUCCESS").write_text("ok\n")
    called: list[bool] = []

    result = run_stage(
        stage="tokenize",
        output_dir=output_dir,
        fingerprint={"k": "v"},
        work=lambda: called.append(True) or {},
        overwrite=False,
        logger=_LOG,
    )

    assert called == []
    assert result["skipped"] is True
    assert result["reason"].endswith("_SUCCESS present")


def test_run_stage_refuses_partial_without_overwrite(tmp_path):
    output_dir = tmp_path / "stage"
    output_dir.mkdir()
    (output_dir / "junk.bin").write_text("partial\n")

    with pytest.raises(RuntimeError, match=r"missing _SUCCESS"):
        run_stage(
            stage="tokenize",
            output_dir=output_dir,
            fingerprint={},
            work=lambda: {},
            overwrite=False,
            logger=_LOG,
        )
    # Output left intact when refusing — destructive cleanup needs overwrite=True.
    assert (output_dir / "junk.bin").is_file()


def test_run_stage_force_rebuilds_with_overwrite(tmp_path):
    output_dir = tmp_path / "stage"
    output_dir.mkdir()
    (output_dir / "stale.bin").write_text("stale\n")
    (output_dir / "_SUCCESS").write_text("ok\n")

    result = run_stage(
        stage="tokenize",
        output_dir=output_dir,
        fingerprint={"k": "v"},
        work=lambda: {"ran": True},
        overwrite=True,
        logger=_LOG,
    )

    assert result["skipped"] is False
    assert result["ran"] is True
    assert not (output_dir / "stale.bin").is_file()
    assert (output_dir / "_SUCCESS").is_file()
    assert (output_dir / MANIFEST_FILE).is_file()


def test_run_stage_writes_manifest_with_audit_fields(tmp_path):
    output_dir = tmp_path / "stage"
    fingerprint = {"a": 1, "b": [2, 3]}

    run_stage(
        stage="materialize",
        output_dir=output_dir,
        fingerprint=fingerprint,
        work=lambda: {"chunks_written": 7},
        overwrite=False,
        logger=_LOG,
    )

    manifest = json.loads((output_dir / MANIFEST_FILE).read_text())
    assert manifest["version"] == 1
    assert manifest["stage"] == "materialize"
    assert manifest["spec_fingerprint"] == fingerprint
    assert isinstance(manifest["wallclock_sec"], (int, float))
    assert manifest["wallclock_sec"] >= 0
    assert "started_at" in manifest
    assert "completed_at" in manifest
    # git_sha may be None outside a repo but the key is always present.
    assert "git_sha" in manifest


def test_run_stage_runs_finalize_before_success_marker(tmp_path):
    output_dir = tmp_path / "stage"
    observed: list[tuple[str, bool]] = []

    def work() -> dict:
        observed.append(("work", (output_dir / "_SUCCESS").exists()))
        return {"value": 3}

    def finalize(result: dict) -> None:
        observed.append(("finalize", (output_dir / "_SUCCESS").exists()))
        assert result["value"] == 3
        (output_dir / "terminal.json").write_text(json.dumps({"ok": True}))

    result = run_stage(
        stage="tokenize",
        output_dir=output_dir,
        fingerprint={},
        work=work,
        finalize=finalize,
        overwrite=False,
        logger=_LOG,
    )

    assert result["value"] == 3
    assert observed == [("work", False), ("finalize", False)]
    assert (output_dir / "terminal.json").is_file()
    assert (output_dir / "_SUCCESS").is_file()


@pytest.mark.parametrize("phase", ["partial", "preflight", "work", "finalize"])
def test_run_stage_on_failure_covers_every_phase(tmp_path, phase):
    output_dir = tmp_path / "stage"
    seen: list[str] = []

    def on_failure(exc: Exception) -> None:
        seen.append(f"{type(exc).__name__}: {exc}")

    kwargs = {
        "stage": "tokenize",
        "output_dir": output_dir,
        "fingerprint": {},
        "work": lambda: {},
        "overwrite": False,
        "logger": _LOG,
        "on_failure": on_failure,
    }
    if phase == "partial":
        output_dir.mkdir()
        (output_dir / "partial.bin").write_text("x")
        match = "missing _SUCCESS"
    elif phase == "preflight":
        kwargs["preflight"] = lambda: (_ for _ in ()).throw(FileNotFoundError("no input"))
        match = "no input"
    elif phase == "work":
        kwargs["work"] = lambda: (_ for _ in ()).throw(RuntimeError("work failed"))
        match = "work failed"
    else:
        kwargs["finalize"] = lambda _result: (_ for _ in ()).throw(ValueError("finalize failed"))
        match = "finalize failed"

    with pytest.raises(Exception, match=match):
        run_stage(**kwargs)

    assert len(seen) == 1
    assert match in seen[0]


def test_run_stage_skip_path_does_not_overwrite_manifest(tmp_path):
    output_dir = tmp_path / "stage"
    output_dir.mkdir()
    (output_dir / "_SUCCESS").write_text("ok\n")
    write_stage_manifest(
        output_dir=output_dir,
        stage="tokenize",
        fingerprint={"prior": "fp"},
    )
    original = (output_dir / MANIFEST_FILE).read_text()

    run_stage(
        stage="tokenize",
        output_dir=output_dir,
        fingerprint={"new": "fp"},
        work=lambda: {"ran": True},
        overwrite=False,
        logger=_LOG,
    )
    assert (output_dir / MANIFEST_FILE).read_text() == original


def test_run_stage_preflights_before_destructive_overwrite(tmp_path):
    """overwrite=True must not delete a usable artifact before inputs validate."""

    output_dir = tmp_path / "stage"
    output_dir.mkdir()
    (output_dir / "_SUCCESS").write_text("ok\n")
    kept = output_dir / "kept.bin"
    kept.write_text("keep\n")

    def preflight() -> None:
        raise FileNotFoundError("missing input")

    def work() -> dict:
        raise AssertionError("work should not run when preflight fails")

    with pytest.raises(FileNotFoundError, match="missing input"):
        run_stage(
            stage="tokenize",
            output_dir=output_dir,
            fingerprint={},
            work=work,
            overwrite=True,
            logger=_LOG,
            preflight=preflight,
        )

    assert kept.read_text() == "keep\n"
    assert (output_dir / "_SUCCESS").is_file()


def test_check_stage_output_clean_dir_returns_none(tmp_path):
    assert check_stage_output(
        stage="test",
        output_dir=tmp_path / "new_stage",
        overwrite=False,
    ) is None
