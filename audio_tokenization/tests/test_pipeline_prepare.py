import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from audio_tokenization.config import load_dataset_spec
from audio_tokenization.stages import prepare as prepare_stage
from audio_tokenization.utils.prepare_data import prepare_parquet_to_shar


_PIPELINE_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "pipeline" / "dataset"


def _load_dataset_cfg(name: str):
    return OmegaConf.load(_PIPELINE_CONFIG_DIR / f"{name}.yaml")


def _namespace_to_plain_dict(ns):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(ns).items()
    }


def test_load_dataset_spec_infore2_and_aozora():
    infore2 = load_dataset_spec(_load_dataset_cfg("infore2"))
    aozora = load_dataset_spec(_load_dataset_cfg("aozora_hurigana"))

    assert infore2.name == "infore2"
    assert infore2.prepare.family == "parquet"
    assert infore2.prepare.input.parquet_glob == "train-*.parquet"
    assert infore2.products.interleave.enabled is True

    assert aozora.name == "aozora_hurigana"
    assert aozora.prepare.metadata.id_column == "sample_id"
    assert aozora.prepare.metadata.custom_columns == ["author", "work", "rendition", "line_num"]


def test_load_dataset_spec_rejects_missing_required_prepare_input():
    with pytest.raises(ValueError, match="parquet_dir"):
        load_dataset_spec(
            {
                "name": "broken",
                "prepare": {
                    "family": "parquet",
                    "input": {},
                    "output": {"shar_dir": "/tmp/out"},
                },
            }
        )


def test_load_dataset_spec_coerces_numeric_string_fields():
    spec = load_dataset_spec(
        {
            "name": "typed",
            "prepare": {
                "family": "parquet",
                "input": {"parquet_dir": "/tmp/data"},
                "output": {
                    "shar_dir": "/tmp/out",
                    "shard_size": "2000",
                    "target_sr": "24000",
                    "read_batch_size": "64",
                },
            },
        }
    )

    assert spec.prepare.output.shard_size == 2000
    assert spec.prepare.output.target_sr == 24000
    assert spec.prepare.output.read_batch_size == 64


@pytest.mark.parametrize(
    ("dataset_name", "argv"),
    [
        (
            "infore2",
            [
                "--parquet-dir",
                "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/doof-ferb___infore2_audiobooks/data",
                "--parquet-glob",
                "train-*.parquet",
                "--shar-dir",
                "/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/infore2_audiobooks_train_fixed_ids_v4",
                "--text-column",
                "transcription",
                "--id-column",
                "audio.path",
                "--language",
                "vi",
                "--input-clip-id-parser",
                "trailing_number_basename",
                "--target-sr",
                "24000",
                "--text-tokenizer",
                "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json",
                "--shard-size",
                "5000",
                "--resampling-backend",
                "soxr",
            ],
        ),
        (
            "aozora_hurigana",
            [
                "--parquet-dir",
                "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/ndl___aozora_hurigana_speech_v2/parquet",
                "--parquet-glob",
                "train-*.parquet",
                "--shar-dir",
                "/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/cooldown/aozora_hurigana_train",
                "--text-column",
                "transcription",
                "--id-column",
                "sample_id",
                "--language",
                "ja",
                "--custom-columns",
                "author",
                "work",
                "rendition",
                "line_num",
                "--input-clip-id-parser",
                "trailing_number",
                "--target-sr",
                "24000",
                "--text-tokenizer",
                "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json",
                "--shard-size",
                "5000",
                "--resampling-backend",
                "soxr",
            ],
        ),
    ],
)
def test_prepare_stage_namespace_matches_legacy_parquet_cli(dataset_name, argv):
    spec = load_dataset_spec(_load_dataset_cfg(dataset_name))

    actual = _namespace_to_plain_dict(prepare_stage.build_prepare_namespace(spec))
    expected = _namespace_to_plain_dict(prepare_parquet_to_shar.build_parser().parse_args(argv))

    assert actual == expected


def test_prepare_stage_rejects_legacy_state_without_version(tmp_path):
    state_path = tmp_path / "_PREPARE_STATE.json"
    state_path.write_text(json.dumps({"parquet_dir": "/tmp/data"}) + "\n")

    with pytest.raises(RuntimeError, match="Unsupported legacy prepare state"):
        prepare_stage._validate_existing_prepare_state(tmp_path)
