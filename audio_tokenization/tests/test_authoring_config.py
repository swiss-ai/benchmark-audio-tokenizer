import pytest

from audio_tokenization.config import load_dataset_spec


def _authoring_interleaved_payload():
    return {
        "name": "authoring_ds",
        "language": "en",
        "recipe": {
            "mode": "audio_text_interleaved",
            "source_type": "parquet",
            "tokenize": True,
            "materialize_interleave": True,
            "audio_text_task": "transcribe",
        },
        "source": {
            "type": "parquet",
            "path": "/raw/parquet",
            "files": "*.parquet",
        },
        "columns": {
            "audio": "audio",
            "text": "text",
            "duration": "duration",
            "id": "id",
            "language_column": None,
            "external_metadata": None,
            "id_prefix": None,
            "custom_fields": [],
            "id_field": "id",
            "text_field": "text",
            "keep": [],
            "constant": {},
            "derived": {},
            "text_tokenize": [],
        },
        "timeline": {
            "parser": None,
            "source_id": None,
            "clip_num": None,
            "clip_start": None,
            "clip_end": None,
            "clip_duration": None,
        },
        "outputs": {
            "shar_dir": "/out/shar",
            "tokenized_dir": "/out/tokenized",
            "name": "authoring_ds",
            "interleaved_dir": "/out/interleaved",
        },
        "tokenizer": {
            "path": "/tokenizer",
            "text_tokenizer": "/tokenizer/tokenizer.json",
            "sampling_rate": 24000,
            "torch_compile": False,
            "trim_last_tokens": 5,
        },
        "conversion": {
            "enabled": True,
            "shard_size": 5000,
            "shar_format": "flac",
            "target_sr": 24000,
            "text_tokenizer": "/tokenizer/tokenizer.json",
            "num_workers": None,
            "resampling_backend": "soxr",
            "mp_start_method": "forkserver",
            "read_batch_size": 256,
        },
        "tokenization": {
            "enabled": True,
            "input_shar_dir": None,
            "partitioning": None,
            "resampling_backend": "soxr",
            "audio_text_task": "transcribe",
            "min_duration": 1.0,
            "max_duration": 200.0,
            "min_sample_rate": 16000,
            "min_rms_db": -50,
            "normalize_peak_db": -3,
            "num_workers": 32,
            "prefetch_factor": 4,
            "max_batch_duration": 2000.0,
            "max_batch_cuts": None,
            "checkpoint_interval_batches": 1000,
            "num_buckets": 20,
            "bucket_buffer_size": 20000,
            "sampler_shuffle": True,
            "sampler_seed": 42,
            "quadratic_duration": None,
            "shar_index_filename": "shar_index.json",
            "wandb": {},
        },
        "materialization": {
            "max_seq_len": 8192,
            "max_gap_sec": 5.0,
            "transcribe_ratio": 0.5,
            "interleave": {
                "enabled": True,
                "strategy": "shift_by_one",
                "cache_dir": None,
                "output_dir": None,
                "tokenizer_path": None,
                "max_seq_len": 262144,
                "max_gap_sec": None,
                "seq_threshold": None,
                "transcribe_ratio": None,
                "num_workers": None,
                "tmp_dir": None,
            },
        },
    }


def test_authoring_materialize_null_interleave_defaults_do_not_suppress_overrides():
    spec = load_dataset_spec(_authoring_interleaved_payload())

    interleave = spec.materialize.interleave
    assert interleave.enabled is True
    assert interleave.output_dir == "/out/interleaved"
    assert interleave.tokenizer_path == "/tokenizer"
    assert interleave.max_seq_len == 8192
    assert interleave.max_gap_sec == 5.0
    assert interleave.transcribe_ratio == 0.5


def test_authoring_sft_materialization_section_requires_conversations_dir():
    payload = _authoring_interleaved_payload()
    payload["recipe"]["materialize_interleave"] = False
    payload["materialization"]["interleave"]["enabled"] = False
    payload["materialization"]["sft"] = {}

    with pytest.raises(ValueError, match=r"materialize\.sft requires conversations_dir"):
        load_dataset_spec(payload)


def test_authoring_accepts_sft_materialization_section():
    payload = _authoring_interleaved_payload()
    payload["recipe"]["mode"] = "sft_audio"
    payload["recipe"]["materialize_interleave"] = False
    payload["outputs"]["sft_dir"] = "/out/sft"
    payload["materialization"] = {
        "sft": {
            "conversations_dir": "/processed/sft/conversations",
            "cache_dir": "/processed/audio_cache",
            "tokenizer_path": "/tokenizer",
            "max_seq_len": 4096,
            "seq_threshold": 2048,
            "num_workers": 3,
        }
    }

    spec = load_dataset_spec(payload)

    assert spec.tokenize.mode == "audio_cache"
    assert spec.materialize.sft.enabled is True
    assert spec.materialize.sft.conversations_dir == "/processed/sft/conversations"
    assert spec.materialize.sft.output_dir == "/out/sft"
    assert spec.materialize.sft.max_seq_len == 4096
    assert spec.materialize.sft.seq_threshold == 2048
    assert spec.materialize.sft.num_workers == 3


def test_authoring_sft_defaults_tokenizer_path_to_tokenizer_dir():
    payload = _authoring_interleaved_payload()
    payload["recipe"]["mode"] = "sft_audio"
    payload["recipe"]["materialize_interleave"] = False
    payload["outputs"]["sft_dir"] = "/out/sft"
    payload["materialization"] = {
        "sft": {
            "conversations_dir": "/processed/sft/conversations",
            "cache_dir": "/processed/audio_cache",
        }
    }

    spec = load_dataset_spec(payload)

    assert spec.materialize.sft.tokenizer_path == "/tokenizer"
