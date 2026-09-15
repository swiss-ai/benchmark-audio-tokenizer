import pytest

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.pipelines.lhotse.core import (
    _build_sampler_kwargs,
    _build_wandb_config,
    _build_wandb_tags,
    _cap_sampler_buckets_to_cut_count,
    run_lhotse_pipeline,
)


def _tokenize_spec(**overrides):
    payload = {
        "tokenizer": {"path": "/tmp/tokenizer"},
        "output": {"output_dir": "/tmp/out"},
    }
    payload.update(overrides)
    return TokenizeSpec.model_validate(payload)


def test_build_sampler_kwargs_uses_explicit_optional_config():
    spec = _tokenize_spec(
        dataloader={
            "max_batch_duration": 2400.0,
            "num_buckets": 32,
            "bucket_buffer_size": 50000,
            "sampler_shuffle": False,
            "sampler_seed": 7,
            "max_batch_cuts": 64,
            "quadratic_duration": 20.0,
        }
    )

    kwargs = _build_sampler_kwargs(spec)

    assert kwargs["max_duration"] == 2400.0
    assert kwargs["num_buckets"] == 32
    assert kwargs["buffer_size"] == 50000
    assert kwargs["shuffle"] is False
    assert kwargs["seed"] == 7
    assert kwargs["max_cuts"] == 64
    assert kwargs["quadratic_duration"] == 20.0
    assert kwargs["world_size"] == 1
    assert kwargs["rank"] == 0
    assert kwargs["drop_last"] is False


def test_build_sampler_kwargs_omits_optional_keys_by_default():
    kwargs = _build_sampler_kwargs(_tokenize_spec())

    assert kwargs["max_duration"] == 2000.0
    assert kwargs["num_buckets"] == 20
    assert kwargs["buffer_size"] == 20000
    assert kwargs["shuffle"] is True
    assert kwargs["seed"] == 42
    assert "max_cuts" not in kwargs
    assert "quadratic_duration" not in kwargs


def test_cap_sampler_buckets_to_rank_local_cut_count():
    kwargs = _build_sampler_kwargs(
        _tokenize_spec(dataloader={"num_buckets": 20, "max_batch_duration": 100.0})
    )

    capped = _cap_sampler_buckets_to_cut_count(kwargs, cut_count=2, rank=1)

    assert capped["num_buckets"] == 2
    assert kwargs["num_buckets"] == 20


def test_cap_sampler_uses_single_bucket_for_single_cut_rank():
    kwargs = _build_sampler_kwargs(
        _tokenize_spec(dataloader={"num_buckets": 20, "max_batch_duration": 100.0})
    )

    capped = _cap_sampler_buckets_to_cut_count(kwargs, cut_count=1, rank=1)

    assert capped["duration_bins"] == []
    assert kwargs["num_buckets"] == 20


def test_cap_sampler_buckets_keeps_unknown_cut_count():
    kwargs = _build_sampler_kwargs(_tokenize_spec(dataloader={"num_buckets": 20}))

    capped = _cap_sampler_buckets_to_cut_count(kwargs, cut_count=None, rank=0)

    assert capped is kwargs
    assert capped["num_buckets"] == 20


def test_wandb_config_includes_resolved_hydra_and_effective_prefetch(tmp_path):
    spec = _tokenize_spec(
        dataloader={
            "num_workers": 99,
            "prefetch_factor": 7,
            "checkpoint_interval_batches": 5,
            "max_batch_duration": 123.0,
        },
        filter={"min_duration": 2.0, "max_duration": 30.0},
        wandb={"enabled": True},
    )
    sampler_kwargs = _build_sampler_kwargs(spec)

    config = _build_wandb_config(
        spec,
        dataset_name="test_dataset",
        input_shar_dirs=["/input/shar"],
        final_output_dir=tmp_path,
        rank=0,
        world_size=4,
        local_rank=0,
        assigned_cut_count=1234,
        sampler_kwargs=sampler_kwargs,
        effective_num_workers=32,
        effective_prefetch_factor=7,
        max_workers_per_rank=72,
        dataloader_timeout=300,
        output_name="test_output",
    )

    assert config["tokenize.dataloader.num_workers"] == 99
    assert config["tokenize.dataloader.prefetch_factor"] == 7
    assert config["tokenize.dataloader.checkpoint_interval_batches"] == 5
    assert config["effective.dataloader.num_workers"] == 32
    assert config["effective.dataloader.prefetch_factor"] == 7
    assert config["effective.dataloader.max_workers_per_rank"] == 72
    assert config["tokenize.filter.min_duration"] == 2.0
    assert config["dataset_name"] == "test_dataset"
    assert config["world_size"] == 4


def test_wandb_tags_include_dataset_stage_and_mode():
    spec = _tokenize_spec(
        mode="audio_text",
        audio_text_format="interleaved",
        audio_text_task="translate",
    )

    tags = _build_wandb_tags(
        ["manual", "dataset:test_dataset"],
        dataset_name="test_dataset",
        spec=spec,
    )

    assert tags == [
        "manual",
        "dataset:test_dataset",
        "stage:tokenize",
        "mode:audio_text",
        "format:interleaved",
        "task:translate",
    ]


def test_run_lhotse_pipeline_requires_stage_assignment(monkeypatch, tmp_path):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)

    with pytest.raises(RuntimeError, match="stage-created SHAR assignment"):
        run_lhotse_pipeline(
            _tokenize_spec(output={"output_dir": str(tmp_path)}),
            dataset_name="test",
            input_shar_dirs=[],
            planned_shar_fields=None,
            rank=0,
            world_size=1,
            local_rank=0,
            final_output_dir=tmp_path,
        )
