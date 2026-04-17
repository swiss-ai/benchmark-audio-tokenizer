from audio_tokenization.pipelines.lhotse.core import _build_sampler_kwargs


def test_build_sampler_kwargs_uses_explicit_optional_config():
    cfg = {
        "max_batch_duration": 2400.0,
        "num_buckets": 32,
        "bucket_buffer_size": 50000,
        "sampler_shuffle": False,
        "sampler_seed": 7,
        "max_batch_cuts": 64,
        "quadratic_duration": 20.0,
        "duration_bins": [3.0, 6.0, 9.0],
        "num_cuts_for_bins_estimate": 1234,
        "sampler_concurrent": True,
        "sampler_sync_buckets": True,
    }

    kwargs = _build_sampler_kwargs(cfg)

    assert kwargs["max_duration"] == 2400.0
    assert kwargs["num_buckets"] == 32
    assert kwargs["buffer_size"] == 50000
    assert kwargs["shuffle"] is False
    assert kwargs["seed"] == 7
    assert kwargs["max_cuts"] == 64
    assert kwargs["quadratic_duration"] == 20.0
    assert kwargs["duration_bins"] == [3.0, 6.0, 9.0]
    assert kwargs["num_cuts_for_bins_estimate"] == 1234
    assert kwargs["concurrent"] is True
    assert kwargs["sync_buckets"] is True
    assert kwargs["world_size"] == 1
    assert kwargs["rank"] == 0
    assert kwargs["drop_last"] is False


def test_build_sampler_kwargs_omits_optional_keys_by_default():
    kwargs = _build_sampler_kwargs({})

    assert kwargs["max_duration"] == 1500.0
    assert kwargs["num_buckets"] == 20
    assert kwargs["buffer_size"] == 20000
    assert kwargs["shuffle"] is True
    assert kwargs["seed"] == 42
    assert "max_cuts" not in kwargs
    assert "quadratic_duration" not in kwargs
    assert "duration_bins" not in kwargs
    assert "num_cuts_for_bins_estimate" not in kwargs
    assert "concurrent" not in kwargs
    assert "sync_buckets" not in kwargs
