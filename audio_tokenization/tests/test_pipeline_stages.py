"""Tests for the unified stage dispatcher at ``audio_tokenization.stages``.

Scoped to the control-plane shape: ``run_stages`` routing, placeholder
behavior for not-yet-wired stages, disabled-section short-circuits.
Per-stage logic (convert internals, tokenize adapter, materialize adapter)
lives in its own test file.
"""

from __future__ import annotations

import copy
import json

import pytest

from audio_tokenization.config import load_dataset_spec
from audio_tokenization.stages import clean_stages, plan_stages, run_stages, status_stages


def _disabled_parquet_spec():
    return load_dataset_spec(
        {
            "name": "disabled_parquet",
            "convert": {
                "family": "parquet",
                "enabled": False,
                "input": {"parquet_dir": "/tmp/in"},
                "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
            },
        }
    )


def test_schema_rejects_typoed_yaml_keys():
    """extra='forbid' on every spec model: a kebab-case-looking typo
    must raise at load time rather than silently substitute a default."""
    with pytest.raises(ValueError, match=r"(extra_forbidden|parquet_dirr)"):
        load_dataset_spec(
            {
                "name": "typo",
                "convert": {
                    "family": "parquet",
                    "input": {"parquet_dirr": "/tmp/in"},  # typo
                    "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
                },
            }
        )


def test_run_stages_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown stage 'bogus'"):
        run_stages(_disabled_parquet_spec(), stage="bogus")


def test_run_stages_convert_dispatches_to_run_convert():
    result = run_stages(_disabled_parquet_spec(), stage="convert")
    assert result == {"convert": {"skipped": True, "reason": "convert.disabled"}}


def test_plan_and_status_for_disabled_stage_are_explicit():
    assert plan_stages(_disabled_parquet_spec(), stage="convert")["convert"]["status"] == "disabled"
    assert status_stages(_disabled_parquet_spec(), stage="convert")["convert"]["action"] == "skip"


def test_run_stages_materialize_disabled_by_default():
    """A spec without any enabled product should run the materialize stage
    as a no-op, not raise. Replaces the old step-5 placeholder test."""
    result = run_stages(_disabled_parquet_spec(), stage="materialize")
    assert result == {
        "materialize": {"interleave": {"skipped": True, "reason": "interleave.disabled"}}
    }


def test_clean_stages_removes_resolved_tokenize_output_dir(tmp_path):
    spec = load_dataset_spec(
        {
            "name": "clean_me",
            "convert": _base_convert_payload(),
            "tokenize": {"tokenizer": {"path": "/t"}, "output": {"output_dir": str(tmp_path / "tok")}},
        }
    )
    planned = plan_stages(spec, stage="tokenize")["tokenize"]
    output_dir = tmp_path / "tok" / "audio_only" / "clean_me"
    assert planned["paths"]["output_dir"] == str(output_dir)

    output_dir.mkdir(parents=True)
    (output_dir / "junk.bin").write_text("x")

    cleaned = clean_stages(spec, stage="tokenize")
    assert cleaned["tokenize"]["removed"] is True
    assert not output_dir.exists()


@pytest.mark.parametrize("alias", ["prepare", "products"])
def test_run_stages_rejects_legacy_stage_names(alias):
    with pytest.raises(ValueError, match=f"Unknown stage {alias!r}"):
        run_stages(_disabled_parquet_spec(), stage=alias)


# ---------------------------------------------------------------------------
# TokenizeSpec contract + cross-section invariant (DatasetSpec.__post_init__)
# ---------------------------------------------------------------------------


def _base_convert_payload():
    return {
        "family": "parquet",
        "input": {"parquet_dir": "/tmp/in"},
        "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
    }


def test_run_stages_convert_short_circuits_when_marker_and_state_match(tmp_path, monkeypatch):
    """Convert must skip the runner entirely when _SUCCESS + _PREPARE_STATE.json
    are present and the on-disk fingerprint matches the spec. Without this,
    every restart re-enters the runner and pays the validate_shar_directory
    pass — slow on large datasets.
    """
    from audio_tokenization.prepare import prepare_parquet_to_shar
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        write_prepare_state_for_spec,
    )

    # Note: parquet_dir is INTENTIONALLY missing on disk. After the P1 fix
    # the skip check happens before resolve_convert_plan globs raw inputs,
    # so a completed convert must short-circuit even when the upstream
    # parquets aren't mounted on this node (the resume-on-different-node
    # scenario).
    shar_dir = tmp_path / "out"
    shar_dir.mkdir()
    spec = load_dataset_spec({
        "name": "d",
        "convert": {
            "family": "parquet",
            "input": {"parquet_dir": str(tmp_path / "missing")},
            "output": {"shar_dir": str(shar_dir), "shard_size": 2000},
        },
    })

    # Use the production helpers to write the on-disk state — guards against
    # silent format drift between the test fixture and the real writer.
    write_prepare_state_for_spec(spec.convert)
    mark_partition_success(shar_dir)

    # Prove the runner isn't invoked. If the short-circuit regresses, the
    # convert dispatch will land here and raise.
    def _boom(_spec):
        raise AssertionError("convert short-circuit failed: parquet runner.run() invoked")
    monkeypatch.setattr(prepare_parquet_to_shar, "run", _boom)

    result = run_stages(spec, stage="convert")
    assert result["convert"]["skipped"] is True
    assert "fingerprint match" in result["convert"]["reason"]


def test_tokenize_section_absent_yields_none():
    spec = load_dataset_spec({"name": "d", "convert": _base_convert_payload()})
    assert spec.tokenize is None


def test_tokenize_minimal_requires_tokenizer_path_and_output_dir():
    with pytest.raises(ValueError, match="path"):
        load_dataset_spec(
            {
                "name": "d",
                "convert": _base_convert_payload(),
                "tokenize": {"tokenizer": {}, "output": {"output_dir": "/o"}},
            }
        )
    with pytest.raises(ValueError, match="output_dir"):
        load_dataset_spec(
            {
                "name": "d",
                "convert": _base_convert_payload(),
                "tokenize": {"tokenizer": {"path": "/t"}, "output": {}},
            }
        )


def test_tokenize_defaults_canonical():
    spec = load_dataset_spec(
        {
            "name": "d",
            "convert": _base_convert_payload(),
            "tokenize": {"tokenizer": {"path": "/t"}, "output": {"output_dir": "/o"}},
        }
    )
    tok = spec.tokenize
    assert tok.mode == "audio_only"
    assert tok.audio_text_format == "direct"
    assert tok.audio_text_task == "transcribe"
    assert tok.input_shar_dir is None
    assert tok.filter.min_duration == 1.0
    assert tok.filter.max_duration == 200.0
    assert tok.dataloader.num_buckets == 20
    assert tok.dataloader.sampler_seed == 42
    assert tok.tokenizer.sampling_rate == 24000
    assert tok.tokenizer.trim_last_tokens == 5


def test_tokenize_accepts_translate_task():
    spec = load_dataset_spec(
        {
            "name": "d",
            "convert": _base_convert_payload(),
            "tokenize": {
                "tokenizer": {"path": "/t"},
                "output": {"output_dir": "/o"},
                "mode": "audio_text",
                "audio_text_task": "translate",
            },
        }
    )
    assert spec.tokenize.audio_text_task == "translate"


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("mode", "audio_image"),
        ("audio_text_format", "novel"),
        ("audio_text_task", "sing"),
    ],
)
def test_tokenize_literal_fields_reject_unknown_values(key, bad):
    with pytest.raises(ValueError, match=key):
        load_dataset_spec(
            {
                "name": "d",
                "convert": _base_convert_payload(),
                "tokenize": {
                    "tokenizer": {"path": "/t"},
                    "output": {"output_dir": "/o"},
                    key: bad,
                },
            }
        )


def test_tokenize_sampling_rate_null_preserved_as_none():
    spec = load_dataset_spec(
        {
            "name": "d",
            "convert": _base_convert_payload(),
            "tokenize": {
                "tokenizer": {"path": "/t", "sampling_rate": None},
                "output": {"output_dir": "/o"},
            },
        }
    )
    assert spec.tokenize.tokenizer.sampling_rate is None


def test_tokenize_fingerprint_excludes_operational_knobs():
    spec = load_dataset_spec(
        {
            "name": "d",
            "convert": _base_convert_payload(),
            "tokenize": {
                "tokenizer": {"path": "/t"},
                "output": {"output_dir": "/o"},
                "dataloader": {"num_workers": 99, "prefetch_factor": 7, "checkpoint_interval_batches": 5},
                "wandb": {"project": "x"},
            },
        }
    )
    fp = spec.tokenize.fingerprint_payload()
    # Operational knobs must NOT appear (they don't affect output content).
    for excluded in ("num_workers", "prefetch_factor", "checkpoint_interval_batches", "wandb"):
        assert excluded not in fp, f"operational knob {excluded!r} leaked into fingerprint"
    # Output-shaping knobs MUST appear.
    for required in ("tokenizer_path", "mode", "sampler_seed", "num_buckets", "filter_min_duration"):
        assert required in fp


def test_interleave_derivation_requires_tokenize_section():
    with pytest.raises(ValueError, match="requires a tokenize section"):
        load_dataset_spec(
            {
                "name": "d",
                "convert": _base_convert_payload(),
                "materialize": {"interleave": {"enabled": True, "output_dir": "/i"}},
            }
        )


@pytest.mark.parametrize(
    ("mode", "fmt"),
    [("audio_only", "direct"), ("audio_text", "direct")],
)
def test_interleave_derivation_requires_interleaved_mode(mode, fmt):
    with pytest.raises(ValueError, match="audio_text_format='interleaved'"):
        load_dataset_spec(
            {
                "name": "d",
                "convert": _base_convert_payload(),
                "tokenize": {
                    "tokenizer": {"path": "/t"},
                    "output": {"output_dir": "/o"},
                    "mode": mode,
                    "audio_text_format": fmt,
                },
                "materialize": {"interleave": {"enabled": True, "output_dir": "/i"}},
            }
        )


def test_interleave_with_explicit_cache_dir_bypasses_cross_section_check():
    """An explicit cache_dir means the user is consuming an externally
    produced cache; the pipeline has nothing to derive, so the
    tokenize-mode constraint shouldn't apply."""
    spec = load_dataset_spec(
        {
            "name": "d",
            "convert": _base_convert_payload(),
            "materialize": {
                "interleave": {
                    "enabled": True,
                    "cache_dir": "/external/cache",
                    "output_dir": "/i",
                }
            },
        }
    )
    assert spec.materialize.interleave.enabled is True
    assert spec.materialize.interleave.cache_dir == "/external/cache"


# ---------------------------------------------------------------------------
# stages/tokenize.py adapter: resume gating + dependency checks
# ---------------------------------------------------------------------------


def _tokenize_spec_payload(shar_dir: str, output_dir: str) -> dict:
    return {
        "name": "smoke",
        "convert": {
            "family": "parquet",
            "input": {"parquet_dir": "/tmp/in"},
            "output": {"shar_dir": shar_dir, "shard_size": 2000},
        },
        "tokenize": {
            "tokenizer": {"path": "/tmp/tokenizer"},
            "output": {"output_dir": output_dir},
        },
    }


def test_run_tokenize_short_circuits_when_section_absent():
    from audio_tokenization.stages.tokenize import run_tokenize

    spec = load_dataset_spec(
        {
            "name": "d",
            "convert": _base_convert_payload(),
        }
    )
    assert run_tokenize(spec) == {"skipped": True, "reason": "tokenize.disabled"}


def test_run_tokenize_requires_prepare_success_marker(tmp_path):
    from audio_tokenization.stages.tokenize import run_tokenize

    # shar_dir exists but no _SUCCESS marker
    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()
    spec = load_dataset_spec(
        _tokenize_spec_payload(str(shar_dir), str(tmp_path / "out"))
    )
    with pytest.raises(RuntimeError, match=r"missing _SUCCESS"):
        run_tokenize(spec)


def test_run_tokenize_skips_prepare_check_when_input_shar_dir_explicit(tmp_path, monkeypatch):
    """An explicitly supplied input_shar_dir means the user is consuming
    an externally built SHAR — this pipeline's _SUCCESS convention doesn't
    apply there. The adapter must not require our marker on external
    paths.
    """
    from audio_tokenization.stages.tokenize import run_tokenize

    external_shar = tmp_path / "external_shar"
    external_shar.mkdir()  # no _SUCCESS; adapter should NOT complain

    payload = _tokenize_spec_payload(str(tmp_path / "our_prepare"), str(tmp_path / "out"))
    payload["tokenize"]["input_shar_dir"] = [str(external_shar)]
    spec = load_dataset_spec(payload)

    # Stub out the heavy pipeline so we only exercise control flow.
    captured: dict = {}

    def fake_run(pipeline_cfg):
        captured["cfg"] = pipeline_cfg
        return {"samples_processed": 0, "tokens_generated": 0, "output_dir": pipeline_cfg["output_dir"]}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline", fake_run
    )

    result = run_tokenize(spec, resume=False)
    assert result["skipped"] is False
    assert captured["cfg"]["shar_dir"] == [str(external_shar)]


def test_run_tokenize_explicit_missing_shar_does_not_delete_existing_output(tmp_path, monkeypatch):
    from audio_tokenization.pipelines.lhotse.core import _build_output_subdir
    from audio_tokenization.stages.tokenize import run_tokenize

    output_dir = tmp_path / "out"
    payload = _tokenize_spec_payload(str(tmp_path / "our_prepare"), str(output_dir))
    payload["tokenize"]["input_shar_dir"] = [str(tmp_path / "missing_shar")]
    spec = load_dataset_spec(payload)

    subdir = _build_output_subdir({
        "output_name": spec.name,
        "mode": spec.tokenize.mode,
        "audio_text_format": spec.tokenize.audio_text_format,
        "audio_text_task": spec.tokenize.audio_text_task,
    })
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    stale_file = final_dir / "stale.bin"
    stale_file.write_text("stale\n")

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline",
        lambda _: (_ for _ in ()).throw(AssertionError("pipeline should not run")),
    )

    with pytest.raises(FileNotFoundError, match=r"Tokenize input SHAR dir not found"):
        run_tokenize(spec, resume=False)
    assert stale_file.exists()


def test_run_tokenize_skips_on_marker_and_state_match(tmp_path, monkeypatch):
    """Second invocation against a completed output must short-circuit,
    never calling the pipeline."""
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.pipelines.lhotse.core import _build_output_subdir
    from audio_tokenization.stages._provenance import build_tokenize_resume_fingerprint
    from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE, run_tokenize

    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()
    mark_partition_success(shar_dir)

    output_dir = tmp_path / "out"
    spec = load_dataset_spec(_tokenize_spec_payload(str(shar_dir), str(output_dir)))
    fingerprint = build_tokenize_resume_fingerprint(
        spec,
        input_shar_dirs=[str(shar_dir)],
    )

    # Emulate a successful previous run: final output dir + _SUCCESS + state file.
    subdir = _build_output_subdir({
        "output_name": spec.name,
        "mode": spec.tokenize.mode,
        "audio_text_format": spec.tokenize.audio_text_format,
        "audio_text_task": spec.tokenize.audio_text_task,
    })
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    validate_or_write_prepare_state(
        final_dir / TOKENIZE_STATE_FILE,
        expected=fingerprint,
        invariant_keys=tuple(fingerprint.keys()),
        guidance="test",
    )
    mark_partition_success(final_dir)

    def boom(cfg):  # pragma: no cover — should NEVER fire on the skip path
        raise AssertionError("pipeline invoked on resume-skip path")

    monkeypatch.setattr("audio_tokenization.stages.tokenize._invoke_pipeline", boom)

    result = run_tokenize(spec, resume=True)
    assert result["skipped"] is True
    assert "state fingerprint match" in result["reason"]
    assert result["output_dir"] == str(final_dir)


def test_run_tokenize_rejects_resume_on_state_drift(tmp_path, monkeypatch):
    """If the on-disk state doesn't match the current spec, resume must
    fail loudly — never silently overwriting or falsely skipping."""
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.pipelines.lhotse.core import _build_output_subdir
    from audio_tokenization.stages._provenance import build_tokenize_resume_fingerprint
    from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE, run_tokenize

    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()
    mark_partition_success(shar_dir)

    output_dir = tmp_path / "out"
    payload_old = _tokenize_spec_payload(str(shar_dir), str(output_dir))
    payload_old["tokenize"]["filter"] = {"min_duration": 1.0}
    spec_old = load_dataset_spec(payload_old)

    subdir = _build_output_subdir({
        "output_name": spec_old.name,
        "mode": spec_old.tokenize.mode,
        "audio_text_format": spec_old.tokenize.audio_text_format,
        "audio_text_task": spec_old.tokenize.audio_text_task,
    })
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    fp_old = build_tokenize_resume_fingerprint(
        spec_old,
        input_shar_dirs=[str(shar_dir)],
    )
    validate_or_write_prepare_state(
        final_dir / TOKENIZE_STATE_FILE,
        expected=fp_old,
        invariant_keys=tuple(fp_old.keys()),
        guidance="test",
    )
    mark_partition_success(final_dir)

    # New spec: same output_dir but different filter threshold → drift.
    payload_new = _tokenize_spec_payload(str(shar_dir), str(output_dir))
    payload_new["tokenize"]["filter"] = {"min_duration": 5.0}
    spec_new = load_dataset_spec(payload_new)

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline",
        lambda _: (_ for _ in ()).throw(AssertionError("pipeline should not run on drift"))
    )

    with pytest.raises(AssertionError, match=r"Unsafe resume"):
        run_tokenize(spec_new, resume=True)


def test_run_tokenize_rejects_resume_when_upstream_prepare_state_changes(tmp_path, monkeypatch):
    from audio_tokenization.prepare.constants import CURRENT_PREPARE_STATE_VERSION
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.pipelines.lhotse.core import _build_output_subdir
    from audio_tokenization.stages._provenance import (
        build_tokenize_resume_fingerprint,
        read_prepare_provenance,
    )
    from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE, run_tokenize

    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()
    mark_partition_success(shar_dir)

    output_dir = tmp_path / "out"
    payload_old = _tokenize_spec_payload(str(shar_dir), str(output_dir))
    payload_old["convert"]["metadata"] = {"text_column": "text_old"}
    spec_old = load_dataset_spec(payload_old)

    prepare_state_path = shar_dir / "_PREPARE_STATE.json"
    validate_or_write_prepare_state(
        prepare_state_path,
        expected=spec_old.convert.fingerprint_payload(),
        invariant_keys=tuple(spec_old.convert.fingerprint_payload().keys()),
        guidance="test",
    )

    subdir = _build_output_subdir({
        "output_name": spec_old.name,
        "mode": spec_old.tokenize.mode,
        "audio_text_format": spec_old.tokenize.audio_text_format,
        "audio_text_task": spec_old.tokenize.audio_text_task,
    })
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    fp_old = build_tokenize_resume_fingerprint(
        spec_old,
        input_shar_dirs=[str(shar_dir)],
        prepare_provenance=read_prepare_provenance([str(shar_dir)]),
    )
    validate_or_write_prepare_state(
        final_dir / TOKENIZE_STATE_FILE,
        expected=fp_old,
        invariant_keys=tuple(fp_old.keys()),
        guidance="test",
    )
    mark_partition_success(final_dir)

    payload_new = _tokenize_spec_payload(str(shar_dir), str(output_dir))
    payload_new["convert"]["metadata"] = {"text_column": "text_new"}
    spec_new = load_dataset_spec(payload_new)
    prepare_state_path.write_text(
        json.dumps(
            {
                "version": CURRENT_PREPARE_STATE_VERSION,
                **spec_new.convert.fingerprint_payload(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline",
        lambda _: (_ for _ in ()).throw(AssertionError("pipeline should not run on drift"))
    )

    with pytest.raises(AssertionError, match=r"Unsafe resume"):
        run_tokenize(spec_new, resume=True)


def test_run_tokenize_rejects_resume_when_upstream_audio_dir_prepare_state_changes(
    tmp_path, monkeypatch
):
    from audio_tokenization.prepare.constants import CURRENT_PREPARE_STATE_VERSION
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
        write_prepare_state_for_spec as write_audio_dir_prepare_state,
    )
    from audio_tokenization.pipelines.lhotse.core import _build_output_subdir
    from audio_tokenization.stages._provenance import (
        build_tokenize_resume_fingerprint,
        read_prepare_provenance,
    )
    from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE, run_tokenize

    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()
    mark_partition_success(shar_dir)
    output_dir = tmp_path / "out"

    payload_old = {
        "name": "audio_dir_tok",
        "convert": {
            "family": "audio_dir",
            "input": {
                "audio_root": "/tmp/audio",
                "jsonl_files": ["/tmp/vad.jsonl"],
                "vad_max_chunk_sec": 200.0,
            },
            "output": {"shar_dir": str(shar_dir), "shard_size": 2000},
        },
        "tokenize": {
            "tokenizer": {"path": "/tmp/tokenizer"},
            "output": {"output_dir": str(output_dir)},
        },
    }
    spec_old = load_dataset_spec(payload_old)
    write_audio_dir_prepare_state(spec_old.convert)

    subdir = _build_output_subdir({
        "output_name": spec_old.name,
        "mode": spec_old.tokenize.mode,
        "audio_text_format": spec_old.tokenize.audio_text_format,
        "audio_text_task": spec_old.tokenize.audio_text_task,
    })
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    fp_old = build_tokenize_resume_fingerprint(
        spec_old,
        input_shar_dirs=[str(shar_dir)],
        prepare_provenance=read_prepare_provenance([str(shar_dir)]),
    )
    validate_or_write_prepare_state(
        final_dir / TOKENIZE_STATE_FILE,
        expected=fp_old,
        invariant_keys=tuple(fp_old.keys()),
        guidance="test",
    )
    mark_partition_success(final_dir)

    payload_new = copy.deepcopy(payload_old)
    payload_new["convert"]["input"]["vad_max_chunk_sec"] = 123.0
    spec_new = load_dataset_spec(payload_new)
    (shar_dir / "_PREPARE_STATE.json").write_text(
        json.dumps(
            {
                "version": CURRENT_PREPARE_STATE_VERSION,
                **spec_new.convert.fingerprint_payload(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline",
        lambda _: (_ for _ in ()).throw(AssertionError("pipeline should not run on drift"))
    )

    with pytest.raises(AssertionError, match=r"Unsafe resume"):
        run_tokenize(spec_new, resume=True)


def test_run_tokenize_derives_output_name_from_dataset_name(tmp_path, monkeypatch):
    from audio_tokenization.prepare.runtime import mark_partition_success
    from audio_tokenization.stages.tokenize import run_tokenize

    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()
    mark_partition_success(shar_dir)

    payload = _tokenize_spec_payload(str(shar_dir), str(tmp_path / "out"))
    payload["name"] = "infore2"
    spec = load_dataset_spec(payload)

    captured: dict = {}

    def fake_run(pipeline_cfg):
        captured["cfg"] = pipeline_cfg
        return {"output_dir": pipeline_cfg["output_dir"]}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline", fake_run
    )

    run_tokenize(spec, resume=False)
    assert captured["cfg"]["output_name"] == "infore2"
    assert captured["cfg"]["dataset_name"] == "infore2"


def test_run_tokenize_clears_partial_output_dir_before_rerun(tmp_path, monkeypatch):
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.pipelines.lhotse.core import _build_output_subdir
    from audio_tokenization.stages._provenance import build_tokenize_resume_fingerprint
    from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE, run_tokenize

    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()
    mark_partition_success(shar_dir)

    output_dir = tmp_path / "out"
    spec = load_dataset_spec(_tokenize_spec_payload(str(shar_dir), str(output_dir)))
    subdir = _build_output_subdir({
        "output_name": spec.name,
        "mode": spec.tokenize.mode,
        "audio_text_format": spec.tokenize.audio_text_format,
        "audio_text_task": spec.tokenize.audio_text_task,
    })
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    stale_file = final_dir / "rank_0000_chunk_000001.bin"
    stale_file.write_text("stale\n")

    fingerprint = build_tokenize_resume_fingerprint(
        spec,
        input_shar_dirs=[str(shar_dir)],
    )
    validate_or_write_prepare_state(
        final_dir / TOKENIZE_STATE_FILE,
        expected=fingerprint,
        invariant_keys=tuple(fingerprint.keys()),
        guidance="test",
    )

    def fake_run(_cfg):
        assert not stale_file.exists(), "partial rerun leaked stale tokenize artifact"
        return {"samples_processed": 0, "tokens_generated": 0}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline", fake_run
    )

    result = run_tokenize(spec, resume=True)
    assert result["skipped"] is False


def test_run_stages_all_runs_in_order_and_short_circuits():
    """stage='all' runs convert → tokenize → materialize in order; each
    short-circuits cleanly when its section is disabled or absent."""
    result = run_stages(_disabled_parquet_spec(), stage="all")
    assert result == {
        "convert": {"skipped": True, "reason": "convert.disabled"},
        "tokenize": {"skipped": True, "reason": "tokenize.disabled"},
        "materialize": {"interleave": {"skipped": True, "reason": "interleave.disabled"}},
    }


# ---------------------------------------------------------------------------
# stages/materialize.py adapter: interleave wiring
# ---------------------------------------------------------------------------


def _interleave_spec_payload(
    *, tokenize_output_dir: str, interleave_output_dir: str
) -> dict:
    return {
        "name": "ds",
        "convert": {
            "family": "parquet",
            "input": {"parquet_dir": "/tmp/in"},
            "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
        },
        "tokenize": {
            "tokenizer": {"path": "/tmp/tokenizer"},
            "output": {"output_dir": tokenize_output_dir},
            "mode": "audio_text",
            "audio_text_format": "interleaved",
        },
        "materialize": {
            "interleave": {
                "enabled": True,
                "output_dir": interleave_output_dir,
            }
        },
    }


def _materialize_tokenize_stage_success(cache_dir, spec, *, input_shar_dirs=None):
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.stages._provenance import build_tokenize_resume_fingerprint
    from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE

    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = build_tokenize_resume_fingerprint(
        spec,
        input_shar_dirs=input_shar_dirs or ["/tmp/shar"],
    )
    validate_or_write_prepare_state(
        cache_dir / TOKENIZE_STATE_FILE,
        expected=fingerprint,
        invariant_keys=tuple(fingerprint.keys()),
        guidance="test",
    )
    mark_partition_success(cache_dir)


def test_run_materialize_interleave_disabled_returns_skip():
    from audio_tokenization.stages.materialize import run_materialize as run_materialize_impl

    spec = load_dataset_spec(
        {"name": "d", "convert": _base_convert_payload()}
    )
    assert run_materialize_impl(spec) == {
        "interleave": {"skipped": True, "reason": "interleave.disabled"}
    }


def test_run_materialize_interleave_invokes_shift_by_one(tmp_path, monkeypatch):
    from audio_tokenization.stages.materialize import run_materialize as run_materialize_impl

    tokenize_out = tmp_path / "tokenize"
    interleave_out = tmp_path / "interleave"
    spec = load_dataset_spec(
        _interleave_spec_payload(
            tokenize_output_dir=str(tokenize_out),
            interleave_output_dir=str(interleave_out),
        )
    )
    _materialize_tokenize_stage_success(
        tokenize_out / "interleave_cache" / spec.name,
        spec,
    )

    captured: dict = {}

    def fake_shift(argv):
        captured["argv"] = argv

    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one", fake_shift
    )

    result = run_materialize_impl(spec, resume=False)
    assert result["interleave"]["skipped"] is False

    # argv carries the derived parquet_dir (tokenize interleave_cache) and
    # the inherited tokenizer_path from the tokenize section.
    argv = captured["argv"]
    assert "--parquet-dir" in argv
    parquet_dir = argv[argv.index("--parquet-dir") + 1]
    assert parquet_dir.endswith("/interleave_cache/ds"), parquet_dir
    assert "--tokenizer-path" in argv
    assert argv[argv.index("--tokenizer-path") + 1] == "/tmp/tokenizer"
    assert "--output-dir" in argv
    assert argv[argv.index("--output-dir") + 1] == str(interleave_out)


def test_run_materialize_interleave_rejects_unsupported_strategy(tmp_path, monkeypatch):
    from audio_tokenization.stages.materialize import run_materialize as run_materialize_impl

    payload = _interleave_spec_payload(
        tokenize_output_dir=str(tmp_path / "tokenize"),
        interleave_output_dir=str(tmp_path / "interleave"),
    )
    payload["materialize"]["interleave"]["strategy"] = "pattern"
    spec = load_dataset_spec(payload)

    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one",
        lambda _: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    with pytest.raises(NotImplementedError, match=r"strategy='pattern'"):
        run_materialize_impl(spec)


def test_run_materialize_interleave_skips_on_marker_and_state_match(tmp_path, monkeypatch):
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.stages._provenance import build_interleave_resume_fingerprint
    from audio_tokenization.stages.materialize import (
        INTERLEAVE_STATE_FILE,
        run_materialize as run_materialize_impl,
    )

    tokenize_out = tmp_path / "tokenize"
    interleave_out = tmp_path / "interleave"
    interleave_out.mkdir(parents=True)

    spec = load_dataset_spec(
        _interleave_spec_payload(
            tokenize_output_dir=str(tokenize_out),
            interleave_output_dir=str(interleave_out),
        )
    )
    cache_dir = tokenize_out / "interleave_cache" / spec.name
    _materialize_tokenize_stage_success(cache_dir, spec)
    fp = build_interleave_resume_fingerprint(
        spec.materialize.interleave,
        cache_dir=cache_dir,
        tokenizer_path=spec.tokenize.tokenizer.path,
        tokenize_provenance={
            str(cache_dir.resolve()): json.loads(
                (cache_dir / "tokenize_state.json").read_text()
            )
        },
    )
    validate_or_write_prepare_state(
        interleave_out / INTERLEAVE_STATE_FILE,
        expected=fp,
        invariant_keys=tuple(fp.keys()),
        guidance="test",
    )
    mark_partition_success(interleave_out)

    def boom(argv):  # pragma: no cover
        raise AssertionError("shift_by_one invoked on resume-skip path")

    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one", boom
    )

    result = run_materialize_impl(spec, resume=True)
    assert result["interleave"]["skipped"] is True
    assert "state fingerprint match" in result["interleave"]["reason"]


def test_run_materialize_interleave_rejects_resume_on_state_drift(tmp_path, monkeypatch):
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.stages._provenance import build_interleave_resume_fingerprint
    from audio_tokenization.stages.materialize import (
        INTERLEAVE_STATE_FILE,
        run_materialize as run_materialize_impl,
    )

    tokenize_out = tmp_path / "tokenize"
    interleave_out = tmp_path / "interleave"
    interleave_out.mkdir(parents=True)

    payload_old = _interleave_spec_payload(
        tokenize_output_dir=str(tokenize_out),
        interleave_output_dir=str(interleave_out),
    )
    payload_old["materialize"]["interleave"]["max_seq_len"] = 131072
    spec_old = load_dataset_spec(payload_old)
    cache_dir = tokenize_out / "interleave_cache" / spec_old.name
    _materialize_tokenize_stage_success(cache_dir, spec_old)
    fp_old = build_interleave_resume_fingerprint(
        spec_old.materialize.interleave,
        cache_dir=cache_dir,
        tokenizer_path=spec_old.tokenize.tokenizer.path,
        tokenize_provenance={
            str(cache_dir.resolve()): json.loads(
                (cache_dir / "tokenize_state.json").read_text()
            )
        },
    )
    validate_or_write_prepare_state(
        interleave_out / INTERLEAVE_STATE_FILE,
        expected=fp_old,
        invariant_keys=tuple(fp_old.keys()),
        guidance="test",
    )
    mark_partition_success(interleave_out)

    payload_new = _interleave_spec_payload(
        tokenize_output_dir=str(tokenize_out),
        interleave_output_dir=str(interleave_out),
    )
    payload_new["materialize"]["interleave"]["max_seq_len"] = 262144
    spec_new = load_dataset_spec(payload_new)

    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one",
        lambda _: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    with pytest.raises(AssertionError, match=r"Unsafe resume"):
        run_materialize_impl(spec_new, resume=True)


def test_run_materialize_interleave_rejects_resume_when_derived_tokenizer_path_changes(
    tmp_path, monkeypatch
):
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.stages._provenance import build_interleave_resume_fingerprint
    from audio_tokenization.stages.materialize import (
        INTERLEAVE_STATE_FILE,
        run_materialize as run_materialize_impl,
    )

    tokenize_out = tmp_path / "tokenize"
    interleave_out = tmp_path / "interleave"
    interleave_out.mkdir(parents=True)

    payload_old = _interleave_spec_payload(
        tokenize_output_dir=str(tokenize_out),
        interleave_output_dir=str(interleave_out),
    )
    payload_old["tokenize"]["tokenizer"]["path"] = "/tmp/tokenizer_old"
    spec_old = load_dataset_spec(payload_old)
    cache_dir = tokenize_out / "interleave_cache" / spec_old.name
    _materialize_tokenize_stage_success(cache_dir, spec_old)
    fp_old = build_interleave_resume_fingerprint(
        spec_old.materialize.interleave,
        cache_dir=cache_dir,
        tokenizer_path=spec_old.tokenize.tokenizer.path,
        tokenize_provenance={
            str(cache_dir.resolve()): json.loads(
                (cache_dir / "tokenize_state.json").read_text()
            )
        },
    )
    validate_or_write_prepare_state(
        interleave_out / INTERLEAVE_STATE_FILE,
        expected=fp_old,
        invariant_keys=tuple(fp_old.keys()),
        guidance="test",
    )
    mark_partition_success(interleave_out)

    payload_new = _interleave_spec_payload(
        tokenize_output_dir=str(tokenize_out),
        interleave_output_dir=str(interleave_out),
    )
    payload_new["tokenize"]["tokenizer"]["path"] = "/tmp/tokenizer_new"
    spec_new = load_dataset_spec(payload_new)

    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one",
        lambda _: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    with pytest.raises(AssertionError, match=r"Unsafe resume"):
        run_materialize_impl(spec_new, resume=True)


def test_run_materialize_interleave_rejects_resume_when_derived_cache_dir_changes(
    tmp_path, monkeypatch
):
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.stages._provenance import build_interleave_resume_fingerprint
    from audio_tokenization.stages.materialize import (
        INTERLEAVE_STATE_FILE,
        run_materialize as run_materialize_impl,
    )

    tokenize_out_old = tmp_path / "tokenize_old"
    tokenize_out_new = tmp_path / "tokenize_new"
    interleave_out = tmp_path / "interleave"
    interleave_out.mkdir(parents=True)

    payload_old = _interleave_spec_payload(
        tokenize_output_dir=str(tokenize_out_old),
        interleave_output_dir=str(interleave_out),
    )
    spec_old = load_dataset_spec(payload_old)
    cache_dir_old = tokenize_out_old / "interleave_cache" / spec_old.name
    _materialize_tokenize_stage_success(cache_dir_old, spec_old)
    fp_old = build_interleave_resume_fingerprint(
        spec_old.materialize.interleave,
        cache_dir=cache_dir_old,
        tokenizer_path=spec_old.tokenize.tokenizer.path,
        tokenize_provenance={
            str(cache_dir_old.resolve()): json.loads(
                (cache_dir_old / "tokenize_state.json").read_text()
            )
        },
    )
    validate_or_write_prepare_state(
        interleave_out / INTERLEAVE_STATE_FILE,
        expected=fp_old,
        invariant_keys=tuple(fp_old.keys()),
        guidance="test",
    )
    mark_partition_success(interleave_out)

    payload_new = _interleave_spec_payload(
        tokenize_output_dir=str(tokenize_out_new),
        interleave_output_dir=str(interleave_out),
    )
    spec_new = load_dataset_spec(payload_new)
    _materialize_tokenize_stage_success(
        tokenize_out_new / "interleave_cache" / spec_new.name,
        spec_new,
    )

    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one",
        lambda _: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    with pytest.raises(AssertionError, match=r"Unsafe resume"):
        run_materialize_impl(spec_new, resume=True)


def test_run_materialize_interleave_requires_tokenize_success_for_derived_cache(
    tmp_path, monkeypatch
):
    from audio_tokenization.stages.materialize import run_materialize as run_materialize_impl

    tokenize_out = tmp_path / "tokenize"
    interleave_out = tmp_path / "interleave"
    spec = load_dataset_spec(
        _interleave_spec_payload(
            tokenize_output_dir=str(tokenize_out),
            interleave_output_dir=str(interleave_out),
        )
    )

    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one",
        lambda _: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    with pytest.raises(RuntimeError, match=r"missing _SUCCESS"):
        run_materialize_impl(spec, resume=False)


def test_run_materialize_explicit_missing_cache_dir_does_not_delete_existing_output(
    tmp_path, monkeypatch
):
    from audio_tokenization.stages.materialize import run_materialize as run_materialize_impl

    interleave_out = tmp_path / "interleave"
    interleave_out.mkdir(parents=True, exist_ok=True)
    stale_file = interleave_out / "stale.parquet"
    stale_file.write_text("stale\n")

    payload = _interleave_spec_payload(
        tokenize_output_dir=str(tmp_path / "tokenize"),
        interleave_output_dir=str(interleave_out),
    )
    payload["materialize"]["interleave"]["cache_dir"] = str(tmp_path / "missing_cache")
    payload["materialize"]["interleave"]["tokenizer_path"] = "/tmp/tokenizer"
    spec = load_dataset_spec(payload)

    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one",
        lambda _: (_ for _ in ()).throw(AssertionError("shift_by_one should not run")),
    )

    with pytest.raises(FileNotFoundError, match=r"Explicit interleave cache_dir not found"):
        run_materialize_impl(spec, resume=False)
    assert stale_file.exists()


def test_run_materialize_interleave_rejects_resume_when_upstream_tokenize_state_changes(
    tmp_path, monkeypatch
):
    from audio_tokenization.prepare.constants import CURRENT_PREPARE_STATE_VERSION
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.stages._provenance import build_interleave_resume_fingerprint
    from audio_tokenization.stages.materialize import (
        INTERLEAVE_STATE_FILE,
        run_materialize as run_materialize_impl,
    )
    from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE

    tokenize_out = tmp_path / "tokenize"
    interleave_out = tmp_path / "interleave"
    interleave_out.mkdir(parents=True)

    payload_old = _interleave_spec_payload(
        tokenize_output_dir=str(tokenize_out),
        interleave_output_dir=str(interleave_out),
    )
    payload_old["tokenize"]["filter"] = {"min_duration": 1.0}
    spec_old = load_dataset_spec(payload_old)
    cache_dir = tokenize_out / "interleave_cache" / spec_old.name
    _materialize_tokenize_stage_success(cache_dir, spec_old)
    fp_old = build_interleave_resume_fingerprint(
        spec_old.materialize.interleave,
        cache_dir=cache_dir,
        tokenizer_path=spec_old.tokenize.tokenizer.path,
        tokenize_provenance={
            str(cache_dir.resolve()): json.loads(
                (cache_dir / TOKENIZE_STATE_FILE).read_text()
            )
        },
    )
    validate_or_write_prepare_state(
        interleave_out / INTERLEAVE_STATE_FILE,
        expected=fp_old,
        invariant_keys=tuple(fp_old.keys()),
        guidance="test",
    )
    mark_partition_success(interleave_out)

    payload_new = _interleave_spec_payload(
        tokenize_output_dir=str(tokenize_out),
        interleave_output_dir=str(interleave_out),
    )
    payload_new["tokenize"]["filter"] = {"min_duration": 5.0}
    spec_new = load_dataset_spec(payload_new)
    (cache_dir / TOKENIZE_STATE_FILE).write_text(
        json.dumps(
            {
                "version": CURRENT_PREPARE_STATE_VERSION,
                **spec_new.tokenize.fingerprint_payload(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one",
        lambda _: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    with pytest.raises(AssertionError, match=r"Unsafe resume"):
        run_materialize_impl(spec_new, resume=True)
