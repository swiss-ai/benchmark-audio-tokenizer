"""Tests for the unified stage dispatcher at ``audio_tokenization.stages``.

Scoped to the control-plane shape: ``run_stages`` routing, placeholder
behavior for not-yet-wired stages, disabled-section short-circuits.
Per-stage logic (convert internals, tokenize adapter, materialize adapter)
lives in its own test file.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from audio_tokenization.config import load_dataset_spec
from audio_tokenization.pipelines.shard_io import INTERLEAVE_CACHE_OUTPUT_STEM
from audio_tokenization.stages._plans import ResolvedStagePlan
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


def test_run_stages_materialize_without_enabled_product_raises():
    """Under explicit single-stage run, asking to materialize a spec with no
    enabled product is a configuration error — not a silent no-op. A Slurm
    job that did nothing while reporting success is the failure mode this
    guard prevents."""
    with pytest.raises(ValueError, match="no materialize section"):
        run_stages(_disabled_parquet_spec(), stage="materialize")


def test_run_stages_tokenize_without_section_raises():
    """Same guard for tokenize: a spec without a tokenize section (no
    outputs.tokenized_dir authored) must fail loudly, not silent-skip."""
    with pytest.raises(ValueError, match="no tokenize section"):
        run_stages(_disabled_parquet_spec(), stage="tokenize")


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


def test_run_with_resume_false_removes_drifted_state_before_validation(tmp_path):
    from audio_tokenization.prepare.runtime import validate_or_write_prepare_state
    from audio_tokenization.stages._resume import run_with_resume

    output_dir = tmp_path / "stage_out"
    output_dir.mkdir()
    stale_file = output_dir / "stale.bin"
    stale_file.write_text("stale\n")
    validate_or_write_prepare_state(
        output_dir / "state.json",
        expected={"num_buckets": 20},
        invariant_keys=("num_buckets",),
        guidance="test",
    )

    result = run_with_resume(
        output_dir=output_dir,
        state_filename="state.json",
        fingerprint={"num_buckets": 4},
        guidance="test",
        stage_label="tokenize",
        resume=False,
        work=lambda: {"ran": True},
        logger=__import__("logging").getLogger(__name__),
    )

    assert result["skipped"] is False
    assert result["ran"] is True
    assert not stale_file.exists()
    assert json.loads((output_dir / "state.json").read_text())["num_buckets"] == 4


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


def _write_cut_shar_index(shar_dir, durations=(10.0,)):
    shar_dir.mkdir(parents=True, exist_ok=True)
    cut_paths = []
    for idx, duration in enumerate(durations):
        name = f"cuts.{idx:06d}.jsonl.gz"
        cut_paths.append(name)
        with gzip.open(shar_dir / name, "wt") as f:
            f.write(
                json.dumps(
                    {
                        "id": f"cut-{idx}",
                        "duration": duration,
                        "recording": {"sampling_rate": 24000},
                        "custom": {
                            "rms_db": -20.0,
                            "interleave": {
                                "source_id": "source",
                                "clip_num": idx,
                                "clip_start": float(idx),
                                "clip_duration": duration,
                            },
                        },
                    }
                )
                + "\n"
            )
    (shar_dir / "shar_index.json").write_text(
        json.dumps({"fields": {"cuts": cut_paths}}) + "\n"
    )


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
    _write_cut_shar_index(shar_dir)
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


def test_partial_tokenize_spec_does_not_require_convert_section():
    spec = load_dataset_spec(
        {
            "name": "tokenize_only",
            "tokenize": {
                "input_shar_dir": ["/tmp/shar"],
                "tokenizer": {"path": "/tmp/tokenizer"},
                "output": {"output_dir": "/tmp/out"},
            },
        }
    )

    assert spec.convert is None
    assert spec.tokenize.input_shar_dir == ["/tmp/shar"]


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


def test_tokenize_rejects_legacy_normalize_rms_db_key():
    with pytest.raises(ValueError, match="normalize_rms_db"):
        load_dataset_spec(
            {
                "name": "d",
                "convert": _base_convert_payload(),
                "tokenize": {
                    "tokenizer": {"path": "/t"},
                    "output": {"output_dir": "/o"},
                    "filter": {"normalize_rms_db": -3},
                },
            }
        )


def test_interleaved_tokenize_uses_structured_cache_partitioning():
    spec = load_dataset_spec(
        {
            "name": "d",
            "convert": _base_convert_payload(),
            "tokenize": {
                "tokenizer": {"path": "/t"},
                "output": {"output_dir": "/o"},
                "mode": "audio_text",
                "audio_text_format": "interleaved",
            },
        }
    )

    fp = spec.tokenize.fingerprint_payload()
    assert "cache_layout_version" not in fp
    assert fp["partitioning"] == {
        "type": "hash",
        "field": "source_id",
        "num_buckets": 16,
    }


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


def test_run_tokenize_raises_when_section_absent():
    """Under explicit single-stage run, an absent tokenize section is a
    user error (typoed stage or missing outputs.tokenized_dir), not a
    silent no-op."""
    from audio_tokenization.stages.tokenize import run_tokenize

    spec = load_dataset_spec(
        {
            "name": "d",
            "convert": _base_convert_payload(),
        }
    )
    with pytest.raises(ValueError, match="no tokenize section"):
        run_tokenize(spec)


def test_run_tokenize_does_not_call_plan_preflight_directly(tmp_path, monkeypatch):
    from audio_tokenization.stages import tokenize as tokenize_stage

    spec = load_dataset_spec(
        _tokenize_spec_payload(str(tmp_path / "shar"), str(tmp_path / "out"))
    )
    plan = ResolvedStagePlan(
        stage="tokenize",
        enabled=True,
        reason=None,
        inputs={},
        outputs={},
        effective={},
        fingerprint={},
        output_dir=tmp_path / "out",
        state_path=tmp_path / "out" / "tokenize_state.json",
        success_marker=tmp_path / "out" / "_SUCCESS",
        preflight=lambda: (_ for _ in ()).throw(
            AssertionError("run_tokenize should not call plan.preflight()")
        ),
        execute=lambda resume: {"resume": resume},
    )

    monkeypatch.setattr(tokenize_stage, "resolve_tokenize_plan", lambda _spec: plan)

    assert tokenize_stage.run_tokenize(spec, resume=False) == {"resume": False}


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
    _write_cut_shar_index(external_shar)  # no _SUCCESS; adapter should NOT complain

    payload = _tokenize_spec_payload(str(tmp_path / "our_prepare"), str(tmp_path / "out"))
    payload["tokenize"]["input_shar_dir"] = [str(external_shar)]
    spec = load_dataset_spec(payload)

    # Stub out the heavy pipeline so we only exercise control flow.
    captured: dict = {}

    def fake_run(tokenize_spec, **kwargs):
        captured["spec"] = tokenize_spec
        captured["kwargs"] = kwargs
        return {"samples_processed": 0, "tokens_generated": 0, "output_dir": str(kwargs["final_output_dir"])}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline", fake_run
    )

    result = run_tokenize(spec, resume=False)
    assert result["skipped"] is False
    assert captured["kwargs"]["input_shar_dirs"] == [str(external_shar)]


def test_run_tokenize_expands_explicit_shar_dir_glob(tmp_path, monkeypatch):
    from audio_tokenization.stages.tokenize import run_tokenize

    node_02 = tmp_path / "node_02"
    node_01 = tmp_path / "node_01"
    _write_cut_shar_index(node_02)
    _write_cut_shar_index(node_01)

    payload = _tokenize_spec_payload(str(tmp_path / "our_prepare"), str(tmp_path / "out"))
    payload["tokenize"]["input_shar_dir"] = [str(tmp_path / "node_*")]
    spec = load_dataset_spec(payload)

    captured: dict = {}

    def fake_run(tokenize_spec, **kwargs):
        captured["spec"] = tokenize_spec
        captured["kwargs"] = kwargs
        return {"samples_processed": 0, "tokens_generated": 0, "output_dir": str(kwargs["final_output_dir"])}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline", fake_run
    )

    result = run_tokenize(spec, resume=False)
    assert result["skipped"] is False
    assert captured["kwargs"]["input_shar_dirs"] == [str(node_01), str(node_02)]


def test_run_tokenize_expands_partitioned_convert_root(tmp_path, monkeypatch):
    from audio_tokenization.prepare.runtime import mark_partition_success
    from audio_tokenization.stages.tokenize import run_tokenize

    root = tmp_path / "partitioned_shar"
    node_00 = root / "node_00"
    node_01 = root / "node_01"
    _write_cut_shar_index(node_00)
    _write_cut_shar_index(node_01)
    mark_partition_success(node_00)
    mark_partition_success(node_01)

    spec = load_dataset_spec(_tokenize_spec_payload(str(root), str(tmp_path / "out")))

    captured: dict = {}

    def fake_run(tokenize_spec, **kwargs):
        captured["spec"] = tokenize_spec
        captured["kwargs"] = kwargs
        return {"samples_processed": 0, "tokens_generated": 0, "output_dir": str(kwargs["final_output_dir"])}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline", fake_run
    )

    result = run_tokenize(spec, resume=False)
    assert result["skipped"] is False
    assert captured["kwargs"]["input_shar_dirs"] == [str(node_00), str(node_01)]


def test_run_tokenize_distributed_rank0_owns_cleanup_and_success(tmp_path, monkeypatch):
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.output_layout import build_tokenize_output_subdir
    from audio_tokenization.stages.tokenize import (
        TOKENIZE_START_FILE,
        TOKENIZE_STATE_FILE,
        run_tokenize,
    )

    shar_dir = tmp_path / "shar"
    _write_cut_shar_index(shar_dir, durations=(10.0, 11.0, 12.0, 13.0))
    mark_partition_success(shar_dir)

    output_dir = tmp_path / "out"
    spec = load_dataset_spec(_tokenize_spec_payload(str(shar_dir), str(output_dir)))
    subdir = build_tokenize_output_subdir(spec.tokenize, dataset_name=spec.name)
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True)
    stale_file = final_dir / "stale.bin"
    stale_file.write_text("stale\n")
    validate_or_write_prepare_state(
        final_dir / TOKENIZE_STATE_FILE,
        expected={"num_buckets": 999},
        invariant_keys=("num_buckets",),
        guidance="test",
    )

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")

    def stub_invoke(_spec, **kwargs):
        # Simulate the convoy-leader contract: each rank writes its stats, the
        # last rank publishes terminal artifacts (stats_summary + _SUCCESS).
        # Here rank 0 plays the convoy leader by pre-populating the other
        # ranks' stats first, then writing its own stats and triggering publish.
        from audio_tokenization.pipelines.lhotse.stats_reducer import (
            maybe_publish_terminal_artifacts,
            write_rank_stats,
        )

        rank = kwargs["rank"]
        world_size = kwargs["world_size"]
        out = Path(kwargs["final_output_dir"])
        for other in range(world_size):
            if other == rank:
                continue
            write_rank_stats(out, {"rank": other, "success": True})
        write_rank_stats(out, {"rank": rank, "success": True})
        maybe_publish_terminal_artifacts(out, expected_ranks=world_size)
        return {"rank": rank, "output_dir": str(out), "success": True}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline", stub_invoke
    )

    result = run_tokenize(spec, resume=False)

    assert result["skipped"] is False
    assert not stale_file.exists()
    assert (final_dir / TOKENIZE_STATE_FILE).is_file()
    assert json.loads((final_dir / TOKENIZE_STATE_FILE).read_text())["num_buckets"] == 20
    assert (final_dir / TOKENIZE_START_FILE).is_file()
    assert (final_dir / "_SUCCESS").is_file()


def test_run_tokenize_resume_restarts_incomplete_output_after_state_validation(
    tmp_path, monkeypatch
):
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.output_layout import build_tokenize_output_subdir
    from audio_tokenization.stages._provenance import build_tokenize_resume_fingerprint
    from audio_tokenization.stages.tokenize import (
        TOKENIZE_START_FILE,
        TOKENIZE_STATE_FILE,
        run_tokenize,
    )

    shar_dir = tmp_path / "shar"
    _write_cut_shar_index(shar_dir, durations=(10.0, 11.0))
    mark_partition_success(shar_dir)

    output_dir = tmp_path / "out"
    spec = load_dataset_spec(_tokenize_spec_payload(str(shar_dir), str(output_dir)))
    final_dir = output_dir / build_tokenize_output_subdir(
        spec.tokenize,
        dataset_name=spec.name,
    )
    final_dir.mkdir(parents=True)
    partial_chunk = final_dir / "rank_0000_chunk_0000.bin"
    partial_chunk.write_bytes(b"partial")
    stale_stats = final_dir / "rank_0000_stats.json"
    stale_stats.write_text(json.dumps({"rank": 0, "success": False}) + "\n")
    (final_dir / "stats_summary.json").write_text("{}\n")
    (final_dir / TOKENIZE_START_FILE).write_text("{}\n")

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

    def fake_run(_spec, **kwargs):
        assert not partial_chunk.exists()
        assert not stale_stats.exists()
        assert not (final_dir / "stats_summary.json").exists()
        assert not (final_dir / TOKENIZE_START_FILE).exists()
        return {"rank": 0, "success": True, "output_dir": str(kwargs["final_output_dir"])}

    monkeypatch.setattr("audio_tokenization.stages.tokenize._invoke_pipeline", fake_run)

    result = run_tokenize(spec, resume=True)

    assert result["skipped"] is False
    assert not partial_chunk.exists()


def test_run_tokenize_distributed_nonzero_rank_does_not_cleanup(tmp_path, monkeypatch):
    from audio_tokenization.prepare.runtime import mark_partition_success
    from audio_tokenization.output_layout import build_tokenize_output_subdir
    from audio_tokenization.pipelines.lhotse.planning import (
        TokenizeFilter,
        build_shar_work_manifest,
        build_tokenize_assignment,
        write_tokenize_plan_artifacts,
    )
    from audio_tokenization.stages._provenance import (
        build_tokenize_resume_fingerprint,
        read_prepare_provenance,
    )
    from audio_tokenization.stages.tokenize import (
        TOKENIZE_START_FILE,
        _build_start_marker_payload,
        run_tokenize,
    )

    shar_dir = tmp_path / "shar"
    _write_cut_shar_index(shar_dir, durations=(10.0, 11.0, 12.0, 13.0))
    mark_partition_success(shar_dir)

    output_dir = tmp_path / "out"
    spec = load_dataset_spec(_tokenize_spec_payload(str(shar_dir), str(output_dir)))
    subdir = build_tokenize_output_subdir(spec.tokenize, dataset_name=spec.name)
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True)
    stale_file = final_dir / "rank0-owned.bin"
    stale_file.write_text("keep\n")
    manifest = build_shar_work_manifest(
        str(shar_dir),
        tokenize_filter=TokenizeFilter(),
    )
    assignment = build_tokenize_assignment(manifest, world_size=4)
    write_tokenize_plan_artifacts(
        final_dir,
        manifest=manifest,
        assignment=assignment,
    )
    fingerprint = build_tokenize_resume_fingerprint(
        spec,
        input_shar_dirs=[str(shar_dir)],
        prepare_provenance=read_prepare_provenance([str(shar_dir)]),
    )
    (final_dir / TOKENIZE_START_FILE).write_text(
        json.dumps(_build_start_marker_payload(fingerprint, world_size=4)) + "\n"
    )

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")

    def fake_run(_spec, **kwargs):
        assert stale_file.exists()
        return {"rank": 2, "output_dir": str(kwargs["final_output_dir"])}

    monkeypatch.setattr("audio_tokenization.stages.tokenize._invoke_pipeline", fake_run)

    result = run_tokenize(spec, resume=False)

    assert result["skipped"] is False
    assert stale_file.exists()
    assert not (final_dir / "_SUCCESS").exists()


def test_read_prepare_provenance_rejects_legacy_unversioned_state(tmp_path):
    from audio_tokenization.stages._provenance import read_prepare_provenance

    shar_dir = tmp_path / "legacy_shar"
    shar_dir.mkdir()
    legacy_state = {
        "recipe": "commonvoice",
        "corpus_dir": "/data/commonvoice24",
        "split": "other",
        "language": "de",
    }
    (shar_dir / "_PREPARE_STATE.json").write_text(json.dumps(legacy_state) + "\n")

    with pytest.raises(RuntimeError, match="has no version"):
        read_prepare_provenance([str(shar_dir)])


def test_nonzero_rank_ignores_stale_tokenize_start_marker(tmp_path, monkeypatch):
    from audio_tokenization.stages.tokenize import (
        TOKENIZE_START_FILE,
        _build_start_marker_payload,
        _wait_for_rank0_tokenize_start,
    )

    final_dir = tmp_path / "out"
    final_dir.mkdir()
    start_marker = final_dir / TOKENIZE_START_FILE
    fingerprint = {"tokenizer_path": "/tmp/tok", "resolved_input_shar_dirs": ["/tmp/shar"]}
    monkeypatch.setenv("AUDIO_TOKENIZATION_RUN_ID", "fresh-run")
    stale = _build_start_marker_payload(fingerprint, world_size=4)
    stale["run_id"] = "old-run"
    start_marker.write_text(json.dumps(stale) + "\n")
    calls = {"sleep": 0}

    def fake_sleep(_seconds):
        calls["sleep"] += 1
        if calls["sleep"] == 1:
            fresh = _build_start_marker_payload(fingerprint, world_size=4)
            assert fresh["run_id"] == "fresh-run"
            (final_dir / "_tokenize_assignment.json").write_text("{}\n")
            start_marker.write_text(json.dumps(fresh) + "\n")
        else:  # pragma: no cover - would indicate we failed to observe fresh marker
            raise AssertionError("nonzero rank did not accept fresh start marker")

    monkeypatch.setattr("audio_tokenization.stages.tokenize.time.sleep", fake_sleep)

    result = _wait_for_rank0_tokenize_start(
        final_output_dir=final_dir,
        state_filename="tokenize_state.json",
        fingerprint=fingerprint,
        start_marker=start_marker,
        world_size=4,
        resume=False,
        rank=2,
    )

    assert result is None
    assert calls["sleep"] == 1


def test_run_tokenize_allows_more_ranks_than_cut_shards_with_planned_assignment(
    tmp_path, monkeypatch
):
    from audio_tokenization.prepare.runtime import mark_partition_success
    from audio_tokenization.pipelines.lhotse.planning import TOKENIZE_ASSIGNMENT_FILE
    from audio_tokenization.stages.tokenize import run_tokenize

    shar_dir = tmp_path / "shar"
    _write_cut_shar_index(shar_dir, durations=(10.0,))
    mark_partition_success(shar_dir)

    spec = load_dataset_spec(_tokenize_spec_payload(str(shar_dir), str(tmp_path / "out")))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    captured = {}
    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline",
        lambda _spec, **kwargs: captured.setdefault("kwargs", kwargs) or {"rank": 0, "output_dir": str(kwargs["final_output_dir"])},
    )

    result = run_tokenize(spec, resume=False)

    assert result["skipped"] is False
    assert len(captured["kwargs"]["planned_shar_fields"]["cuts"]) == 1
    assert captured["kwargs"]["assigned_cut_count"] == 1
    assignment = json.loads(
        (Path(result["output_dir"]) / TOKENIZE_ASSIGNMENT_FILE).read_text()
    )
    assert assignment["world_size"] == 4
    assert assignment["active_ranks"] == 1


def test_run_tokenize_explicit_missing_shar_does_not_delete_existing_output(tmp_path, monkeypatch):
    from audio_tokenization.output_layout import build_tokenize_output_subdir
    from audio_tokenization.stages.tokenize import run_tokenize

    output_dir = tmp_path / "out"
    payload = _tokenize_spec_payload(str(tmp_path / "our_prepare"), str(output_dir))
    payload["tokenize"]["input_shar_dir"] = [str(tmp_path / "missing_shar")]
    spec = load_dataset_spec(payload)

    subdir = build_tokenize_output_subdir(spec.tokenize, dataset_name=spec.name)
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    stale_file = final_dir / "stale.bin"
    stale_file.write_text("stale\n")

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline",
        lambda _spec, **_kwargs: (_ for _ in ()).throw(AssertionError("pipeline should not run")),
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
    from audio_tokenization.output_layout import build_tokenize_output_subdir
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
    subdir = build_tokenize_output_subdir(spec.tokenize, dataset_name=spec.name)
    final_dir = output_dir / subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    validate_or_write_prepare_state(
        final_dir / TOKENIZE_STATE_FILE,
        expected=fingerprint,
        invariant_keys=tuple(fingerprint.keys()),
        guidance="test",
    )
    mark_partition_success(final_dir)

    def boom(_spec, **_kwargs):  # pragma: no cover — should NEVER fire on the skip path
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
    from audio_tokenization.output_layout import build_tokenize_output_subdir
    from audio_tokenization.stages._provenance import build_tokenize_resume_fingerprint
    from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE, run_tokenize

    shar_dir = tmp_path / "shar"
    shar_dir.mkdir()
    mark_partition_success(shar_dir)

    output_dir = tmp_path / "out"
    payload_old = _tokenize_spec_payload(str(shar_dir), str(output_dir))
    payload_old["tokenize"]["filter"] = {"min_duration": 1.0}
    spec_old = load_dataset_spec(payload_old)

    subdir = build_tokenize_output_subdir(spec_old.tokenize, dataset_name=spec_old.name)
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
        lambda _spec, **_kwargs: (_ for _ in ()).throw(AssertionError("pipeline should not run on drift"))
    )

    with pytest.raises(AssertionError, match=r"Unsafe resume"):
        run_tokenize(spec_new, resume=True)


def test_run_tokenize_rejects_resume_when_upstream_prepare_state_changes(tmp_path, monkeypatch):
    from audio_tokenization.prepare.constants import CURRENT_PREPARE_STATE_VERSION
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.output_layout import build_tokenize_output_subdir
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
        state_version=CURRENT_PREPARE_STATE_VERSION,
        state_label="prepare state",
    )

    subdir = build_tokenize_output_subdir(spec_old.tokenize, dataset_name=spec_old.name)
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
        lambda _spec, **_kwargs: (_ for _ in ()).throw(AssertionError("pipeline should not run on drift"))
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
    from audio_tokenization.output_layout import build_tokenize_output_subdir
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

    subdir = build_tokenize_output_subdir(spec_old.tokenize, dataset_name=spec_old.name)
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
        lambda _spec, **_kwargs: (_ for _ in ()).throw(AssertionError("pipeline should not run on drift"))
    )

    with pytest.raises(AssertionError, match=r"Unsafe resume"):
        run_tokenize(spec_new, resume=True)


def test_run_tokenize_derives_output_name_from_dataset_name(tmp_path, monkeypatch):
    from audio_tokenization.prepare.runtime import mark_partition_success
    from audio_tokenization.stages.tokenize import run_tokenize

    shar_dir = tmp_path / "shar"
    _write_cut_shar_index(shar_dir)
    mark_partition_success(shar_dir)

    payload = _tokenize_spec_payload(str(shar_dir), str(tmp_path / "out"))
    payload["name"] = "infore2"
    spec = load_dataset_spec(payload)

    captured: dict = {}

    def fake_run(tokenize_spec, **kwargs):
        captured["spec"] = tokenize_spec
        captured["kwargs"] = kwargs
        return {"output_dir": str(kwargs["final_output_dir"])}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline", fake_run
    )

    run_tokenize(spec, resume=False)
    assert captured["spec"].output.output_name is None
    assert captured["kwargs"]["dataset_name"] == "infore2"


def test_run_tokenize_resume_false_clears_partial_output_dir_before_rerun(
    tmp_path, monkeypatch
):
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        validate_or_write_prepare_state,
    )
    from audio_tokenization.output_layout import build_tokenize_output_subdir
    from audio_tokenization.stages._provenance import build_tokenize_resume_fingerprint
    from audio_tokenization.stages.tokenize import TOKENIZE_STATE_FILE, run_tokenize

    shar_dir = tmp_path / "shar"
    _write_cut_shar_index(shar_dir)
    mark_partition_success(shar_dir)

    output_dir = tmp_path / "out"
    spec = load_dataset_spec(_tokenize_spec_payload(str(shar_dir), str(output_dir)))
    subdir = build_tokenize_output_subdir(spec.tokenize, dataset_name=spec.name)
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

    def fake_run(_spec, **_kwargs):
        assert not stale_file.exists(), "partial rerun leaked stale tokenize artifact"
        return {"samples_processed": 0, "tokens_generated": 0}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline", fake_run
    )

    result = run_tokenize(spec, resume=False)
    assert result["skipped"] is False


def test_run_stages_requires_single_stage():
    """``run`` rejects multi-stage selection so convert/tokenize/materialize
    cannot accidentally share a Slurm allocation."""
    with pytest.raises(ValueError, match="stage="):
        run_stages(_disabled_parquet_spec(), stage=None)
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stages(_disabled_parquet_spec(), stage="convert,tokenize")


def test_stage_chain_end_to_end_control_plane(tmp_path, monkeypatch):
    """Convert -> tokenize -> materialize works as an artifact-driven graph.

    Each stage is launched as its own ``run_stages`` call (mirroring the
    production Slurm model where each stage is a separate job). Heavy
    decode/GPU/product work is faked; the real state files and success
    markers are still written so downstream stages consume upstream artifacts
    exactly as they do in production.
    """
    from audio_tokenization.prepare.runtime import (
        mark_partition_success,
        write_prepare_state_for_spec,
    )
    from audio_tokenization.output_layout import build_tokenize_output_subdir

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    (parquet_dir / "data.parquet").write_text("stub\n")
    shar_dir = tmp_path / "shar"
    tokenized_dir = tmp_path / "tokenized"
    materialized_dir = tmp_path / "materialized"

    spec = load_dataset_spec(
        {
            "name": "e2e",
            "convert": {
                "family": "parquet",
                "input": {"parquet_dir": str(parquet_dir)},
                "output": {"shar_dir": str(shar_dir), "shard_size": 2000},
            },
            "tokenize": {
                "tokenizer": {"path": "/tmp/tokenizer"},
                "output": {"output_dir": str(tokenized_dir)},
                "mode": "audio_text",
                "audio_text_format": "interleaved",
            },
            "materialize": {
                "interleave": {
                    "enabled": True,
                    "output_dir": str(materialized_dir),
                }
            },
        }
    )

    def fake_convert(_prepare_spec, **_kwargs):
        _write_cut_shar_index(shar_dir)
        write_prepare_state_for_spec(spec.convert)
        mark_partition_success(shar_dir)
        return {"converted": True, "output_dir": str(shar_dir)}

    monkeypatch.setattr("audio_tokenization.stages.convert.validate_prepare_runtime", lambda **_: None)
    monkeypatch.setattr(
        "audio_tokenization.stages.convert.get_prepare_runner",
        lambda _spec: type("FakePrepareModule", (), {"run": staticmethod(fake_convert)}),
    )
    def fake_tokenize(_spec, **kwargs):
        mark_partition_success(kwargs["final_output_dir"])
        return {"tokenized": True, "output_dir": str(kwargs["final_output_dir"])}

    monkeypatch.setattr(
        "audio_tokenization.stages.tokenize._invoke_pipeline",
        fake_tokenize,
    )
    monkeypatch.setattr(
        "audio_tokenization.stages.materialize._invoke_shift_by_one",
        lambda argv: None,
    )

    convert_result = run_stages(spec, stage="convert", resume=False)
    tokenize_result = run_stages(spec, stage="tokenize", resume=False)
    materialize_result = run_stages(spec, stage="materialize", resume=False)

    cache_dir = tokenized_dir / build_tokenize_output_subdir(spec.tokenize, dataset_name=spec.name)
    assert convert_result["convert"]["converted"] is True
    assert tokenize_result["tokenize"]["tokenized"] is True
    assert materialize_result["materialize"]["interleave"]["skipped"] is False
    assert (shar_dir / "_SUCCESS").is_file()
    assert (cache_dir / "_SUCCESS").is_file()
    assert (materialized_dir / "_SUCCESS").is_file()


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


def test_run_materialize_raises_when_no_product_enabled():
    """Under explicit single-stage run, materialize with no enabled product
    is a user error, not a silent no-op."""
    from audio_tokenization.stages.materialize import run_materialize as run_materialize_impl

    spec = load_dataset_spec(
        {"name": "d", "convert": _base_convert_payload()}
    )
    with pytest.raises(ValueError, match="no materialize section"):
        run_materialize_impl(spec)


def test_run_materialize_does_not_call_plan_preflight_directly(tmp_path, monkeypatch):
    from audio_tokenization.stages import materialize as materialize_stage

    spec = load_dataset_spec(
        _interleave_spec_payload(
            tokenize_output_dir=str(tmp_path / "tokenize"),
            interleave_output_dir=str(tmp_path / "interleave"),
        )
    )
    plan = ResolvedStagePlan(
        stage="materialize",
        enabled=True,
        reason=None,
        inputs={},
        outputs={},
        effective={},
        fingerprint={},
        output_dir=tmp_path / "interleave",
        state_path=tmp_path / "interleave" / "products_interleave_state.json",
        success_marker=tmp_path / "interleave" / "_SUCCESS",
        preflight=lambda: (_ for _ in ()).throw(
            AssertionError("run_materialize should not call plan.preflight()")
        ),
        execute=lambda resume: {"interleave": {"resume": resume}},
    )

    monkeypatch.setattr(materialize_stage, "resolve_materialize_plan", lambda _spec: plan)

    assert materialize_stage.run_materialize(spec, resume=False) == {
        "interleave": {"resume": False}
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
        tokenize_out / INTERLEAVE_CACHE_OUTPUT_STEM / spec.name,
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

    # argv carries the derived parquet_dir (tokenize interleave cache) and
    # the inherited tokenizer_path from the tokenize section.
    argv = captured["argv"]
    assert "--parquet-dir" in argv
    parquet_dir = argv[argv.index("--parquet-dir") + 1]
    assert parquet_dir.endswith(f"/{INTERLEAVE_CACHE_OUTPUT_STEM}/ds"), parquet_dir
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
    cache_dir = tokenize_out / INTERLEAVE_CACHE_OUTPUT_STEM / spec.name
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
    cache_dir = tokenize_out / INTERLEAVE_CACHE_OUTPUT_STEM / spec_old.name
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


def test_materialize_max_gap_sec_affects_only_materialize_fingerprint(tmp_path):
    from audio_tokenization.stages._provenance import build_interleave_resume_fingerprint

    payload_old = _interleave_spec_payload(
        tokenize_output_dir=str(tmp_path / "tokenize"),
        interleave_output_dir=str(tmp_path / "interleave"),
    )
    payload_new = copy.deepcopy(payload_old)
    payload_new["materialize"]["interleave"]["max_gap_sec"] = 5.0

    spec_old = load_dataset_spec(payload_old)
    spec_new = load_dataset_spec(payload_new)

    assert spec_old.tokenize.fingerprint_payload() == spec_new.tokenize.fingerprint_payload()

    fp_old = build_interleave_resume_fingerprint(
        spec_old.materialize.interleave,
        cache_dir=tmp_path / "cache",
        tokenizer_path=spec_old.tokenize.tokenizer.path,
    )
    fp_new = build_interleave_resume_fingerprint(
        spec_new.materialize.interleave,
        cache_dir=tmp_path / "cache",
        tokenizer_path=spec_new.tokenize.tokenizer.path,
    )

    assert fp_old["max_gap_sec"] is None
    assert fp_new["max_gap_sec"] == 5.0
    assert fp_old != fp_new


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
    cache_dir = tokenize_out / INTERLEAVE_CACHE_OUTPUT_STEM / spec_old.name
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
    cache_dir_old = tokenize_out_old / INTERLEAVE_CACHE_OUTPUT_STEM / spec_old.name
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
        tokenize_out_new / INTERLEAVE_CACHE_OUTPUT_STEM / spec_new.name,
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
    from audio_tokenization.prepare.constants import CURRENT_STAGE_STATE_VERSION
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
    cache_dir = tokenize_out / INTERLEAVE_CACHE_OUTPUT_STEM / spec_old.name
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
                "version": CURRENT_STAGE_STATE_VERSION,
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
