import json
import copy
import gzip
import subprocess
import sys
import types
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from audio_tokenization import __main__ as audio_main
from audio_tokenization.__main__ import _compose_pipeline_cfg, _split_command
from audio_tokenization.config import load_dataset_spec
from audio_tokenization.config.schema import (
    _INPUT_SPEC_BY_FAMILY,
    PrepareMetadataSpec,
    PrepareSpec,
)
from audio_tokenization.prepare.columnar import ColumnarWorkerArgs
from audio_tokenization.prepare.cli import expand_path_patterns
from audio_tokenization.stages import convert as convert_stage
from audio_tokenization.prepare import (
    prepare_audio_dir_to_shar,
    prepare_hf_to_shar,
    prepare_lhotse_recipe_to_shar,
    prepare_parquet_to_shar,
    prepare_wds_to_shar,
)
from audio_tokenization.prepare.runtime import (
    preflight_prepare_spec,
    resolve_prepare_inputs,
)


def _write_cut_shar_index(shar_dir: Path, durations=(10.0,)) -> None:
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
                        "custom": {"rms_db": -20.0},
                    }
                )
                + "\n"
            )
    (shar_dir / "shar_index.json").write_text(
        json.dumps({"fields": {"cuts": cut_paths}}) + "\n"
    )


_PIPELINE_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "pipeline"


def _load_dataset_cfg(name: str):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(_PIPELINE_CONFIG_DIR.resolve()),
    ):
        cfg = compose(config_name="config", overrides=[f"dataset={name}"])
    return cfg.dataset


def _namespace_to_plain_dict(ns):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(ns).items()
    }


def test_load_dataset_spec_infore2_and_aozora():
    infore2 = load_dataset_spec(_load_dataset_cfg("cooldown/infore2"))
    aozora = load_dataset_spec(_load_dataset_cfg("cooldown/aozora_hurigana"))

    assert infore2.name == "cooldown_infore2"
    assert infore2.convert.family == "parquet"
    assert infore2.convert.input.parquet_glob == "train-*.parquet"
    assert infore2.materialize.interleave.enabled is True

    assert aozora.name == "aozora_hurigana"
    assert aozora.convert.metadata.id_column == "sample_id"
    assert aozora.convert.metadata.custom_columns == [
        "sample_id",
        "author",
        "work",
        "rendition",
        "line_num",
    ]


def test_load_nested_sft_dataset_spec():
    spec = load_dataset_spec(_load_dataset_cfg("sft/teleantifraud_matching"))

    assert spec.name == "sft_teleantifraud_matching"
    assert spec.convert.family == "parquet"
    assert spec.tokenize.mode == "audio_cache"
    assert spec.materialize.sft.enabled is True


def test_load_nested_stage2_and_internal_specs():
    stage2 = load_dataset_spec(_load_dataset_cfg("stage2/libriheavy_large"))
    internal = load_dataset_spec(_load_dataset_cfg("internal/srg_apertus"))

    assert stage2.name == "libriheavy_large"
    assert stage2.convert.output.shar_dir.endswith("/SHAR/stage_2/libriheavy_large")
    assert stage2.tokenize is None

    assert internal.name == "internal_srg_apertus"
    assert internal.convert.output.shar_dir.endswith("/SHAR/internal-only/srg_apertus")
    assert internal.materialize.interleave.enabled is True


def test_load_dataset_spec_does_not_reconfigure_logging(monkeypatch):
    import logging

    def _unexpected_basic_config(*args, **kwargs):
        raise AssertionError("logging.basicConfig should not be called during schema load")

    monkeypatch.setattr(logging, "basicConfig", _unexpected_basic_config)

    spec = load_dataset_spec(
        {
            "name": "hf_minimal",
            "convert": {
                "family": "hf",
                "input": {"arrow_dir": "/tmp/in"},
                "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
            },
        }
    )

    assert spec.convert.family == "hf"


def test_prepare_family_registry_is_owned_by_input_specs():
    expected_modules = {
        "parquet": "audio_tokenization.prepare.prepare_parquet_to_shar",
        "hf": "audio_tokenization.prepare.prepare_hf_to_shar",
        "wds": "audio_tokenization.prepare.prepare_wds_to_shar",
        "audio_dir": "audio_tokenization.prepare.prepare_audio_dir_to_shar",
        "lhotse_recipe": "audio_tokenization.prepare.prepare_lhotse_recipe_to_shar",
    }

    assert set(_INPUT_SPEC_BY_FAMILY) == set(expected_modules)
    for family, input_cls in _INPUT_SPEC_BY_FAMILY.items():
        assert input_cls.FAMILY == family
        assert input_cls.RUNNER_MODULE == expected_modules[family]


def test_prepare_runtime_delegates_resolution_to_family_runner(monkeypatch):
    from audio_tokenization.prepare import runtime

    spec = PrepareSpec.from_mapping(
        {
            "family": "parquet",
            "input": {"parquet_dir": "/tmp/in"},
            "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
        }
    )

    class _Runner:
        @staticmethod
        def resolve(received):
            assert received is spec
            return ["resolved.parquet"], {"family": "parquet"}

    monkeypatch.setattr(runtime, "get_prepare_runner", lambda received: _Runner)

    assert runtime.resolve_prepare_inputs(spec) == (
        ["resolved.parquet"],
        {"family": "parquet"},
    )


def test_source_lhotse_runtime_defaults_repo_dir_to_repo_root():
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "utils" / "source_lhotse_runtime.sh"

    proc = subprocess.run(
        [
            "bash",
            "-lc",
            (
                "unset REPO_DIR; "
                "INSTALL_TORCHCODEC=0 INSTALL_TORCHAUDIO=0 PRINT_LHOTSE_RUNTIME=0 "
                f"source {script}; "
                'printf "%s" "$REPO_DIR"'
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout == str(repo)


def test_prepare_runtime_requires_dev_lhotse_shar_features(monkeypatch):
    from audio_tokenization.prepare.runtime import require_lhotse_shar_features

    fake_lhotse = types.ModuleType("lhotse")
    fake_shar = types.ModuleType("lhotse.shar")

    class _CutSet:
        def to_shar(self, output_dir):  # pragma: no cover - inspected only
            raise AssertionError

    class _SharWriter:
        def __init__(self, output_dir):  # pragma: no cover - inspected only
            raise AssertionError

    fake_lhotse.CutSet = _CutSet
    fake_shar.SharWriter = _SharWriter
    monkeypatch.setitem(sys.modules, "lhotse", fake_lhotse)
    monkeypatch.setitem(sys.modules, "lhotse.shar", fake_shar)

    with pytest.raises(RuntimeError, match="requires dev Lhotse"):
        require_lhotse_shar_features()


def test_prepare_family_runners_canary_resolve_and_preflight(tmp_path):
    from audio_tokenization.prepare.runtime import get_prepare_runner

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    parquet_file = parquet_dir / "data.parquet"
    parquet_file.write_text("stub\n")

    arrow_dir = tmp_path / "arrow"
    arrow_dir.mkdir()
    arrow_file = arrow_dir / "data.arrow"
    arrow_file.write_text("stub\n")

    wds_dir = tmp_path / "wds"
    wds_dir.mkdir()
    wds_file = wds_dir / "data.tar"
    wds_file.write_text("stub\n")

    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    vad_jsonl = tmp_path / "vad.jsonl"
    vad_jsonl.write_text("{}\n")

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    cases = {
        "parquet": (
            {
                "family": "parquet",
                "input": {"parquet_dir": str(parquet_dir)},
                "output": {"shar_dir": str(tmp_path / "out_parquet"), "shard_size": 2000},
            },
            [str(parquet_file)],
        ),
        "hf": (
            {
                "family": "hf",
                "input": {"arrow_dir": str(arrow_dir)},
                "output": {"shar_dir": str(tmp_path / "out_hf"), "shard_size": 2000},
            },
            [str(arrow_file)],
        ),
        "wds": (
            {
                "family": "wds",
                "input": {"wds_shards": [str(wds_dir / "*.tar")]},
                "output": {"shar_dir": str(tmp_path / "out_wds"), "shard_size": 2000},
            },
            [str(wds_file)],
        ),
        "audio_dir": (
            {
                "family": "audio_dir",
                "input": {"audio_root": str(audio_root), "jsonl_files": [str(vad_jsonl)]},
                "output": {"shar_dir": str(tmp_path / "out_audio_dir"), "shard_size": 2000},
            },
            [str(vad_jsonl)],
        ),
        "lhotse_recipe": (
            {
                "family": "lhotse_recipe",
                "input": {
                    "recipe": "librispeech",
                    "corpus_dir": str(corpus_dir),
                    "split": "test-clean",
                },
                "output": {"shar_dir": str(tmp_path / "out_lhotse"), "shard_size": 2000},
            },
            [],
        ),
    }

    for family, (payload, expected_resolved) in cases.items():
        spec = PrepareSpec.from_mapping(payload)
        runner = get_prepare_runner(spec)
        assert callable(runner.resolve)
        assert callable(runner.preflight)
        assert callable(runner.run)

        resolved, summary = resolve_prepare_inputs(spec)
        assert resolved == expected_resolved
        assert summary["family"] == family
        preflight_prepare_spec(spec, runtime_validator=lambda **_kwargs: None)


def test_load_dataset_spec_does_not_import_prepare_modules():
    """Architectural-boundary lock-in: schema parsing must not transitively
    drag in any heavy ``audio_tokenization.prepare.prepare_*_to_shar`` module.
    Stronger than the logging-basicConfig guard alone — catches any
    reintroduction of parser-introspection even if the reintroducer
    remembers to suppress logging.
    """
    import sys

    minimal_yamls = {
        "parquet": {
            "input": {"parquet_dir": "/tmp/in"},
            "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
        },
        "hf": {
            "input": {"arrow_dir": "/tmp/in"},
            "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
        },
        "wds": {
            "input": {"wds_shards": ["/tmp/in/*.tar"]},
            "output": {"shar_dir": "/tmp/out", "shard_size": 5000},
        },
        "audio_dir": {
            "input": {"audio_root": "/tmp/in", "jsonl_files": ["/tmp/vad.jsonl"]},
            "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
        },
        "lhotse_recipe": {
            "input": {"recipe": "librispeech", "corpus_dir": "/tmp/in", "split": "test-clean"},
            "output": {"shar_dir": "/tmp/out", "shard_size": 1000},
        },
    }
    # Drop any prepare modules already imported by sibling tests; we only
    # care whether THIS load_dataset_spec call adds new ones.
    for k in [k for k in sys.modules if "audio_tokenization.prepare.prepare_" in k]:
        del sys.modules[k]

    for family, payload in minimal_yamls.items():
        load_dataset_spec({"name": f"{family}_min", "convert": {"family": family, **payload}})

    leaked = [k for k in sys.modules if "audio_tokenization.prepare.prepare_" in k]
    assert not leaked, (
        f"load_dataset_spec imported prepare modules: {sorted(leaked)}. "
        "Schema parsing must remain decoupled from standalone prepare scripts."
    )


def test_load_dataset_spec_rejects_missing_required_prepare_input():
    with pytest.raises(ValueError, match="parquet_dir"):
        load_dataset_spec(
            {
                "name": "broken",
                "convert": {
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
            "convert": {
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

    assert spec.convert.output.shard_size == 2000
    assert spec.convert.output.target_sr == 24000
    assert spec.convert.output.read_batch_size == 64


# ---------------------------------------------------------------------------
# text_column: canonical default is None ("do not extract"); omit and explicit
# null share that semantic. Setters opt in by writing a column name.
# ---------------------------------------------------------------------------


def test_text_column_defaults_to_none_when_key_omitted():
    """Canonical schema: omitting text_column means "do not extract text".
    No CLI-default substitution; setters opt in explicitly.
    """
    spec = load_dataset_spec(
        {
            "name": "omitted_text",
            "convert": {
                "family": "parquet",
                "input": {"parquet_dir": "/tmp/data"},
                "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
                "metadata": {},
            },
        }
    )
    assert spec.convert.metadata.text_column is None


def test_text_column_explicit_null_means_disabled():
    """Truly unsupervised datasets can opt out with `text_column: null`."""
    spec = load_dataset_spec(
        {
            "name": "unsupervised",
            "convert": {
                "family": "parquet",
                "input": {"parquet_dir": "/tmp/data"},
                "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
                "metadata": {"text_column": None},
            },
        }
    )
    assert spec.convert.metadata.text_column is None


# ---------------------------------------------------------------------------
# id_column must accept str, list[str], or null
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yaml_value", "expected"),
    [
        (None, None),
        ("sample_id", "sample_id"),
        (["session", "seg"], ["session", "seg"]),
        (["single_list_entry"], ["single_list_entry"]),
    ],
)
def test_id_column_accepts_str_list_or_null(yaml_value, expected):
    spec = load_dataset_spec(
        {
            "name": "id_shape",
            "convert": {
                "family": "parquet",
                "input": {"parquet_dir": "/tmp/data"},
                "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
                "metadata": {"id_column": yaml_value},
            },
        }
    )
    assert spec.convert.metadata.id_column == expected


def test_id_column_rejects_invalid_shape():
    with pytest.raises((ValueError, TypeError), match="id_column"):
        load_dataset_spec(
            {
                "name": "bad_id",
                "convert": {
                    "family": "parquet",
                    "input": {"parquet_dir": "/tmp/data"},
                    "output": {"shar_dir": "/tmp/out", "shard_size": 2000},
                    "metadata": {"id_column": 42},
                },
            }
        )


# ---------------------------------------------------------------------------
# Schema-semantic contract: canonical defaults, null-rejection, semantic-null
# ---------------------------------------------------------------------------


def _minimal_payload(family: str, **output_overrides):
    """Smallest valid convert payload per family. Always includes shard_size
    (now a required canonical field)."""
    bases = {
        "parquet": {"input": {"parquet_dir": "/tmp/in"}},
        "hf": {"input": {"arrow_dir": "/tmp/in"}},
        "wds": {"input": {"wds_shards": ["/tmp/*.tar"]}},
        "audio_dir": {"input": {"audio_root": "/tmp/in", "jsonl_files": ["/tmp/v.jsonl"]}},
        "lhotse_recipe": {"input": {"recipe": "librispeech", "corpus_dir": "/tmp/in", "split": "test-clean"}},
    }
    out = {"shar_dir": "/tmp/out", "shard_size": 2000}
    out.update(output_overrides)
    return {"name": f"{family}_test", "convert": {"family": family, **bases[family], "output": out}}


def _set_convert_field(payload: dict, path: tuple[str, ...], value):
    cur = payload["convert"]
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


@pytest.mark.parametrize("family", ["parquet", "hf", "wds", "audio_dir", "lhotse_recipe"])
def test_shard_size_required_in_canonical_config(family):
    """shard_size is behavior-shaping; canonical config must state intent
    rather than fall back to an arbitrary default."""
    payload = _minimal_payload(family)
    payload["convert"]["output"].pop("shard_size")
    with pytest.raises((ValueError, TypeError), match="shard_size"):
        load_dataset_spec(payload)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("output", "shard_size"),
        ("output", "shar_format"),
        ("output", "mp_start_method"),
        ("output", "read_batch_size"),
        ("metadata", "audio_column"),
        ("metadata", "id_field"),
        ("metadata", "text_field"),
    ],
)
def test_null_rejected_for_non_semantic_fields(section, key):
    """Fields where explicit null has no semantic meaning must reject it
    rather than silently substituting a default."""
    payload = _minimal_payload("parquet")
    payload["convert"].setdefault("metadata", {})
    payload["convert"][section][key] = None
    with pytest.raises((ValueError, TypeError), match=key):
        load_dataset_spec(payload)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("output", "target_sr"),
        ("output", "text_tokenizer"),
        ("output", "num_workers"),
        ("output", "resampling_backend"),
        ("metadata", "text_column"),
        ("metadata", "duration_column"),
        ("metadata", "id_column"),
        ("metadata", "language_column"),
        ("metadata", "language"),
    ],
)
def test_null_preserved_for_semantic_fields(section, key):
    """Fields where explicit null carries semantic meaning (no resample, no
    text extraction, defer to env, ...) must surface as ``None`` on the spec."""
    payload = _minimal_payload("parquet")
    payload["convert"].setdefault("metadata", {})
    payload["convert"][section][key] = None
    spec = load_dataset_spec(payload)
    assert getattr(getattr(spec.convert, section), key) is None


def test_schema_defaults_are_canonical():
    """Lock the per-field default values from the convert spec docs/plan."""
    spec = load_dataset_spec(_minimal_payload("parquet"))
    # output bucket defaults
    assert spec.convert.output.shar_format == "flac"
    assert spec.convert.output.target_sr == 24000
    assert spec.convert.output.text_tokenizer is None
    assert spec.convert.output.num_workers is None
    assert spec.convert.output.resampling_backend == "soxr"
    assert spec.convert.output.mp_start_method == "forkserver"
    assert spec.convert.output.read_batch_size == 256
    # metadata bucket defaults
    assert spec.convert.metadata.audio_column == "audio"
    assert spec.convert.metadata.text_column is None
    assert spec.convert.metadata.duration_column is None
    assert spec.convert.metadata.source_id_column is None
    assert spec.convert.metadata.clip_num_column is None
    assert spec.convert.metadata.id_column is None
    assert spec.convert.metadata.id_field == "id"
    assert spec.convert.metadata.text_field == "text"


@pytest.mark.parametrize(
    "metadata",
    [
        {"clip_end_column": "parent_end"},
        {"clip_duration_column": "segment_duration"},
    ],
)
def test_clip_timestamp_columns_require_clip_start_column(metadata):
    payload = _minimal_payload("parquet")
    payload["convert"]["metadata"] = metadata

    with pytest.raises(ValueError, match="require clip_start_column"):
        load_dataset_spec(payload)


def test_clip_timestamp_columns_reject_end_and_duration_together():
    payload = _minimal_payload("parquet")
    payload["convert"]["metadata"] = {
        "clip_start_column": "parent_start",
        "clip_end_column": "parent_end",
        "clip_duration_column": "segment_duration",
    }

    with pytest.raises(ValueError, match="set exactly one of clip_end_column or clip_duration_column"):
        load_dataset_spec(payload)


def test_clip_num_column_requires_source_id_column():
    payload = _minimal_payload("parquet")
    payload["convert"]["metadata"] = {"clip_num_column": "segment_index"}

    with pytest.raises(ValueError, match="clip_num_column requires source_id_column"):
        load_dataset_spec(payload)


def test_source_id_column_requires_clip_num_or_clip_start_column():
    payload = _minimal_payload("parquet")
    payload["convert"]["metadata"] = {"source_id_column": "original_audio_id"}

    with pytest.raises(ValueError, match="source_id_column without clip_num_column"):
        load_dataset_spec(payload)


def test_source_id_column_rejects_input_clip_id_parser():
    payload = _minimal_payload("parquet")
    payload["convert"]["metadata"] = {
        "source_id_column": "original_audio_id",
        "input_clip_id_parser": "trailing_number",
    }

    with pytest.raises(ValueError, match="set either source_id_column"):
        load_dataset_spec(payload)


@pytest.mark.parametrize(
    ("family", "payload_mutator", "required_keys", "normalized_assertions"),
    [
        (
            "parquet",
            lambda payload: _set_convert_field(
                payload,
                ("metadata", "custom_columns"),
                ["work", "author", "line_num", "rendition"],
            ),
            ("family", "input.parquet_dir", "input.parquet_glob", "metadata.custom_columns"),
            {"metadata.custom_columns": ["author", "line_num", "rendition", "work"]},
        ),
        (
            "hf",
            lambda payload: payload["convert"]["input"].update(
                {"arrow_dir": None, "arrow_files": ["/tmp/b.arrow", "/tmp/a.arrow"]}
            ),
            ("family", "input.arrow_dir", "input.arrow_glob", "input.arrow_files"),
            {"input.arrow_files": ["/tmp/a.arrow", "/tmp/b.arrow"]},
        ),
        (
            "wds",
            lambda payload: _set_convert_field(
                payload, ("input", "wds_shards"), ["/tmp/z.tar", "/tmp/a.tar"]
            ),
            ("family", "input.wds_shards", "input.vad_max_chunk_sec"),
            {"input.wds_shards": ["/tmp/a.tar", "/tmp/z.tar"]},
        ),
        (
            "audio_dir",
            lambda payload: _set_convert_field(
                payload, ("input", "jsonl_files"), ["/tmp/z.jsonl", "/tmp/a.jsonl"]
            ),
            ("family", "input.audio_root", "input.audio_ext", "input.vad_max_chunk_sec"),
            {"input.jsonl_files": ["/tmp/a.jsonl", "/tmp/z.jsonl"]},
        ),
        (
            "lhotse_recipe",
            lambda payload: None,
            ("family", "input.recipe", "input.recipe_kwargs", "input.shar_index_filename"),
            {},
        ),
    ],
)
def test_prepare_spec_fingerprint_payload_is_hashable_and_stable(
    family, payload_mutator, required_keys, normalized_assertions
):
    """Prepare fingerprints must be JSON-round-trippable and stable for all families."""
    payload = _minimal_payload(family)
    payload_mutator(payload)
    spec = load_dataset_spec(payload)
    fp = spec.convert.fingerprint_payload()

    dumped = json.dumps(fp, sort_keys=True)
    assert fp == json.loads(dumped)
    for key in required_keys:
        assert key in fp
    for key, expected in normalized_assertions.items():
        assert fp[key] == expected


def test_hf_preflight_honors_arrow_files_override_when_arrow_dir_missing(tmp_path):
    arrow_file = tmp_path / "explicit.arrow"
    arrow_file.write_bytes(b"not a real arrow file; preflight should not parse it")
    missing_arrow_dir = tmp_path / "missing_arrow_dir"

    payload = _minimal_payload("hf")
    payload["convert"]["input"].update(
        {
            "arrow_dir": str(missing_arrow_dir),
            "arrow_files": [str(arrow_file)],
        }
    )
    spec = PrepareSpec.from_mapping(payload["convert"])

    resolved, summary = resolve_prepare_inputs(spec)
    assert resolved == [str(arrow_file)]
    assert summary["arrow_files"] == [str(arrow_file)]

    preflight_prepare_spec(spec, runtime_validator=lambda **_kwargs: None)


@pytest.mark.parametrize("family", ["parquet", "hf"])
def test_prepare_fingerprint_includes_clip_timestamp_metadata_fields(family):
    payload = _minimal_payload(family)
    payload["convert"]["metadata"] = {
        "source_id_column": "original_audio_id",
        "clip_start_column": "parent_start",
        "clip_end_column": "parent_end",
        "clip_duration_column": None,
    }

    spec = load_dataset_spec(payload)
    fp = spec.convert.fingerprint_payload()

    assert fp["metadata.source_id_column"] == "original_audio_id"
    assert fp["metadata.clip_num_column"] is None
    assert fp["metadata.clip_start_column"] == "parent_start"
    assert fp["metadata.clip_end_column"] == "parent_end"
    assert fp["metadata.clip_duration_column"] is None


# ---------------------------------------------------------------------------
# End-to-end: spec drift detection at the parquet runner's state-file layer.
# Both Hydra and CLI paths converge on PrepareSpec; tests build the spec
# directly rather than going through argv.
# ---------------------------------------------------------------------------


def _minimal_parquet_spec(tmp_path, **metadata_overrides) -> PrepareSpec:
    """PrepareSpec for parquet with shar_dir under tmp_path. metadata
    overrides go into the metadata section (text_column, id_column, ...)."""
    return PrepareSpec.from_mapping({
        "family": "parquet",
        "input": {"parquet_dir": str(tmp_path / "in")},
        "output": {"shar_dir": str(tmp_path / "out"), "shard_size": 2000},
        "metadata": metadata_overrides,
    })


def test_audio_dir_prepare_does_not_create_output_before_audio_index_succeeds(
    tmp_path, monkeypatch
):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    jsonl_path = tmp_path / "vad.jsonl"
    jsonl_path.write_text("{}\n")
    out = tmp_path / "out"

    spec = PrepareSpec.from_mapping(
        {
            "family": "audio_dir",
            "input": {
                "audio_root": str(audio_root),
                "jsonl_files": [str(jsonl_path)],
            },
            "output": {
                "shar_dir": str(out),
                "shard_size": 2000,
            },
        }
    )

    monkeypatch.setattr(prepare_audio_dir_to_shar, "validate_prepare_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(prepare_audio_dir_to_shar, "build_audio_index", lambda *_args, **_kwargs: {})

    with pytest.raises(FileNotFoundError, match=r"No \*\.ogg files found"):
        prepare_audio_dir_to_shar.run(spec)

    assert not out.exists()


def test_expand_path_patterns_resolves_globs_and_literals(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("{}\n")
    b.write_text("{}\n")

    assert expand_path_patterns([str(tmp_path / "*.jsonl"), str(a)]) == [
        str(a),
        str(b),
    ]


def test_expand_path_patterns_rejects_unmatched_pattern(tmp_path):
    with pytest.raises(FileNotFoundError, match="No files match pattern"):
        expand_path_patterns([str(tmp_path / "*.missing")])


def test_resolve_convert_plan_canonicalizes_lhotse_recipe_num_workers(tmp_path):
    """When num_workers is unset, the lhotse_recipe runner needs a concrete
    int (it does ``range(num_workers)`` directly). Hardcoded to 64 so the
    resolved plan stays stable across nodes with different ``SLURM_CPUS_PER_TASK``."""
    spec = load_dataset_spec(
        {
            "name": "recipe_ds",
            "convert": {
                "family": "lhotse_recipe",
                "input": {
                    "recipe": "librispeech",
                    "corpus_dir": str(tmp_path),
                    "split": "test-clean",
                },
                "output": {
                    "shar_dir": str(tmp_path / "out"),
                    "shard_size": 1000,
                },
            },
        }
    )

    plan = convert_stage.resolve_convert_plan(spec)
    assert plan.effective["effective_num_workers"] == 64
    assert plan.fingerprint["output.num_workers"] == 64


def test_parquet_run_builds_typed_columnar_worker_args(tmp_path, monkeypatch):
    in_dir = tmp_path / "parquet"
    out_dir = tmp_path / "shar"
    in_dir.mkdir()
    (in_dir / "chunk.parquet").write_text("")
    spec = PrepareSpec.from_mapping(
        {
            "family": "parquet",
            "input": {"parquet_dir": str(in_dir)},
            "output": {"shar_dir": str(out_dir), "shard_size": 2000},
        }
    )

    monkeypatch.setattr(prepare_parquet_to_shar, "_preflight_prepare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prepare_parquet_to_shar, "resolve_num_workers", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        prepare_parquet_to_shar,
        "distribute_round_robin",
        lambda resolved, _num_workers: [resolved],
    )
    monkeypatch.setattr(prepare_parquet_to_shar, "load_text_tokenizer", lambda _path: None)

    captured = {}

    def _capture(_worker_fn, worker_args, _shar_dir, _num_workers, mp_start_method="forkserver"):
        captured["worker_args"] = worker_args
        captured["mp_start_method"] = mp_start_method
        return []

    monkeypatch.setattr(prepare_parquet_to_shar, "run_pool_and_finalize", _capture)

    prepare_parquet_to_shar.run(spec)

    assert len(captured["worker_args"]) == 1
    assert isinstance(captured["worker_args"][0], ColumnarWorkerArgs)
    assert captured["worker_args"][0].input_paths == (str(in_dir / "chunk.parquet"),)


def test_hf_run_builds_typed_columnar_worker_args(tmp_path, monkeypatch):
    in_dir = tmp_path / "arrow"
    out_dir = tmp_path / "shar"
    in_dir.mkdir()
    (in_dir / "chunk.arrow").write_text("")
    spec = PrepareSpec.from_mapping(
        {
            "family": "hf",
            "input": {"arrow_dir": str(in_dir)},
            "output": {"shar_dir": str(out_dir), "shard_size": 2000},
        }
    )

    monkeypatch.setattr(prepare_hf_to_shar, "_preflight_prepare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prepare_hf_to_shar, "resolve_num_workers", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        prepare_hf_to_shar,
        "distribute_round_robin",
        lambda resolved, _num_workers: [resolved],
    )
    monkeypatch.setattr(prepare_hf_to_shar, "load_text_tokenizer", lambda _path: None)

    captured = {}

    def _capture(_worker_fn, worker_args, _shar_dir, _num_workers, mp_start_method="forkserver"):
        captured["worker_args"] = worker_args
        captured["mp_start_method"] = mp_start_method
        return []

    monkeypatch.setattr(prepare_hf_to_shar, "run_pool_and_finalize", _capture)

    prepare_hf_to_shar.run(spec)

    assert len(captured["worker_args"]) == 1
    assert isinstance(captured["worker_args"][0], ColumnarWorkerArgs)
    assert captured["worker_args"][0].input_paths == (str(in_dir / "chunk.arrow"),)


def test_audio_dir_run_uses_shared_audio_index_with_fork(tmp_path, monkeypatch):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    jsonl_path = tmp_path / "vad.jsonl"
    jsonl_path.write_text("{}\n")
    out_dir = tmp_path / "shar"
    spec = PrepareSpec.from_mapping(
        {
            "family": "audio_dir",
            "input": {
                "audio_root": str(audio_root),
                "jsonl_files": [str(jsonl_path)],
            },
            "output": {
                "shar_dir": str(out_dir),
                "shard_size": 2000,
                "mp_start_method": "fork",
            },
        }
    )

    monkeypatch.setattr(prepare_audio_dir_to_shar, "validate_prepare_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(prepare_audio_dir_to_shar, "build_audio_index", lambda *_args, **_kwargs: {"clip": "/audio.wav"})
    monkeypatch.setattr(prepare_audio_dir_to_shar, "resolve_num_workers", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        prepare_audio_dir_to_shar,
        "distribute_round_robin",
        lambda resolved, _num_workers: [resolved],
    )

    captured = {}

    def _capture(_worker_fn, worker_args, _shar_dir, _num_workers, mp_start_method="forkserver"):
        captured["worker_args"] = worker_args
        captured["mp_start_method"] = mp_start_method
        return []

    monkeypatch.setattr(prepare_audio_dir_to_shar, "run_pool_and_finalize", _capture)

    prepare_audio_dir_to_shar.run(spec)

    assert captured["mp_start_method"] == "fork"
    assert len(captured["worker_args"]) == 1
    worker_args = captured["worker_args"][0]
    assert isinstance(worker_args, prepare_audio_dir_to_shar.AudioDirWorkerArgs)
    assert worker_args.audio_index is None
    assert worker_args.jsonl_paths == (str(jsonl_path),)
    assert prepare_audio_dir_to_shar._AUDIO_INDEX is None


def test_run_convert_dispatch_skips_when_disabled():
    spec = load_dataset_spec(
        {
            "name": "disabled",
            "convert": {
                "family": "wds",
                "enabled": False,
                "input": {"wds_shards": ["/x/*.tar"]},
                "output": {"shar_dir": "/out/disabled", "shard_size": 5000},
            },
        }
    )
    result = convert_stage.run_convert(spec)
    assert result == {"skipped": True, "reason": "convert.disabled"}


def test_run_convert_overwrite_removes_existing_output_dir(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("stale\n")
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    (parquet_dir / "data.parquet").write_text("stub\n")

    spec = load_dataset_spec(
        {
            "name": "rerun",
            "convert": {
                "family": "parquet",
                "input": {"parquet_dir": str(parquet_dir)},
                "output": {"shar_dir": str(out), "shard_size": 2000},
            },
        }
    )

    captured = {}

    class _RunnerModule:
        @staticmethod
        def preflight(_prepare_spec, **_kwargs):
            return None

        @staticmethod
        def run(prepare_spec, *, resolved_inputs=None):
            del resolved_inputs
            captured["exists_on_entry"] = Path(prepare_spec.output.shar_dir).exists()
            _write_cut_shar_index(Path(prepare_spec.output.shar_dir))
            return {"ran": True}

    monkeypatch.setattr(convert_stage, "get_prepare_runner", lambda _spec: _RunnerModule)

    result = convert_stage.run_convert(spec, overwrite=True)
    assert result["ran"] is True
    assert result["skipped"] is False
    assert result["stage"] == "convert"
    assert captured["exists_on_entry"] is True
    assert not (out / "stale.txt").exists()
    assert (out / "_shar_work_manifest.json").is_file()


def test_run_convert_overwrite_preflights_before_deleting_existing_output(
    tmp_path, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    (out / "_SUCCESS").write_text("ok\n")
    kept = out / "usable_shar.bin"
    kept.write_text("keep\n")
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    parquet_path = parquet_dir / "data.parquet"
    parquet_path.write_text("stub\n")

    spec = load_dataset_spec(
        {
            "name": "validate_before_delete",
            "convert": {
                "family": "parquet",
                "input": {"parquet_dir": str(parquet_dir)},
                "output": {"shar_dir": str(out), "shard_size": 2000},
            },
        }
    )

    monkeypatch.setattr(
        convert_stage,
        "resolve_prepare_inputs",
        lambda _spec: ([str(parquet_path)], {"resolved_inputs": [str(parquet_path)]}),
    )
    class _RunnerModule:
        @staticmethod
        def preflight(_prepare_spec, **_kwargs):
            raise FileNotFoundError("missing tokenizer runtime")

        @staticmethod
        def run(_prepare_spec, *, resolved_inputs=None):
            raise AssertionError("runner must not execute when preflight fails")

    monkeypatch.setattr(convert_stage, "get_prepare_runner", lambda _spec: _RunnerModule)

    with pytest.raises(FileNotFoundError, match="missing tokenizer runtime"):
        convert_stage.run_convert(spec, overwrite=True)

    assert kept.read_text() == "keep\n"
    assert (out / "_SUCCESS").is_file()


def test_run_convert_preflights_and_runs_with_same_resolved_inputs(tmp_path, monkeypatch):
    out = tmp_path / "out"
    resolved_inputs = [str(tmp_path / "inputs" / "data.parquet")]
    spec = load_dataset_spec(
        {
            "name": "single_resolve",
            "convert": {
                "family": "parquet",
                "input": {"parquet_dir": str(tmp_path / "inputs")},
                "output": {"shar_dir": str(out), "shard_size": 2000},
            },
        }
    )
    captured = {}

    def _resolve_inputs_once(prepare_spec):
        captured["resolved_spec"] = prepare_spec
        return resolved_inputs, {
            "family": "parquet",
            "resolved_inputs": resolved_inputs,
        }

    class _RunnerModule:
        @staticmethod
        def preflight(prepare_spec, **kwargs):
            captured["preflight_spec"] = prepare_spec
            captured["preflight_resolved_inputs"] = kwargs["resolved_inputs"]

        @staticmethod
        def run(prepare_spec, *, resolved_inputs=None):
            captured["runner_spec"] = prepare_spec
            captured["runner_resolved_inputs"] = resolved_inputs
            _write_cut_shar_index(Path(prepare_spec.output.shar_dir))
            return {"ran": True}

    monkeypatch.setattr(convert_stage, "resolve_prepare_inputs", _resolve_inputs_once)
    monkeypatch.setattr(convert_stage, "get_prepare_runner", lambda _spec: _RunnerModule)

    result = convert_stage.run_convert(spec)

    assert result["ran"] is True
    assert result["skipped"] is False
    assert captured["resolved_spec"] is spec.convert
    assert captured["preflight_spec"] is spec.convert
    assert captured["preflight_resolved_inputs"] == resolved_inputs
    assert captured["runner_spec"] is spec.convert
    assert captured["runner_resolved_inputs"] == resolved_inputs
    assert (out / "_shar_work_manifest.json").is_file()


def test_convert_execution_loads_runner_from_input_spec(tmp_path, monkeypatch):
    out = tmp_path / "out"
    resolved_inputs = [str(tmp_path / "inputs" / "data.parquet")]
    spec = load_dataset_spec(
        {
            "name": "runner_from_spec",
            "convert": {
                "family": "parquet",
                "input": {"parquet_dir": str(tmp_path / "inputs")},
                "output": {"shar_dir": str(out), "shard_size": 2000},
            },
        }
    )
    captured = {}

    class _RunnerModule:
        @staticmethod
        def preflight(_prepare_spec, **_kwargs):
            return None

        @staticmethod
        def run(prepare_spec, *, resolved_inputs=None):
            captured["runner_spec"] = prepare_spec
            captured["runner_resolved_inputs"] = resolved_inputs
            _write_cut_shar_index(Path(prepare_spec.output.shar_dir))
            return {"ran": True}

    monkeypatch.setattr(convert_stage, "resolve_prepare_inputs", lambda _spec: (resolved_inputs, {}))
    monkeypatch.setattr(convert_stage, "get_prepare_runner", lambda _spec: _RunnerModule)

    result = convert_stage.run_convert(spec)

    assert result["ran"] is True
    assert result["skipped"] is False
    assert captured["runner_spec"] is spec.convert
    assert captured["runner_resolved_inputs"] == resolved_inputs


def test_run_convert_preflights_exactly_once_before_runner(tmp_path, monkeypatch):
    """Convert preflight runs once in run_stage before runner work."""
    out = tmp_path / "out"
    resolved_inputs = [str(tmp_path / "inputs" / "data.parquet")]
    spec = load_dataset_spec(
        {
            "name": "single_preflight",
            "convert": {
                "family": "parquet",
                "input": {"parquet_dir": str(tmp_path / "inputs")},
                "output": {"shar_dir": str(out), "shard_size": 2000},
            },
        }
    )
    preflight_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        convert_stage,
        "resolve_prepare_inputs",
        lambda prepare_spec: (resolved_inputs, {"family": prepare_spec.family}),
    )

    class _RunnerModule:
        @staticmethod
        def preflight(prepare_spec, **kwargs):
            preflight_calls.append({"family": prepare_spec.family, "kwargs": kwargs})

        @staticmethod
        def run(prepare_spec, *, resolved_inputs=None):
            _write_cut_shar_index(Path(prepare_spec.output.shar_dir))
            return {"ran": True}

    monkeypatch.setattr(convert_stage, "get_prepare_runner", lambda _spec: _RunnerModule)

    convert_stage.run_convert(spec)

    assert len(preflight_calls) == 1, f"expected exactly one preflight, got {len(preflight_calls)}"
    assert preflight_calls[0]["family"] == "parquet"
    assert preflight_calls[0]["kwargs"]["resolved_inputs"] == resolved_inputs


def test_split_command_defaults_to_run():
    assert _split_command(["dataset=cooldown/infore2"]) == ("run", ["dataset=cooldown/infore2"])


def test_split_command_explicit_status():
    assert _split_command(["status", "dataset=cooldown/infore2"]) == (
        "status",
        ["dataset=cooldown/infore2"],
    )


def test_root_pipeline_config_leaves_stage_missing_without_override():
    cfg = _compose_pipeline_cfg(["dataset=stage2/libriheavy_large"])

    assert OmegaConf.is_missing(cfg, "stage")
    assert cfg.dataset.name == "libriheavy_large"


def test_root_pipeline_config_requires_dataset_override():
    with pytest.raises(Exception, match="dataset"):
        _compose_pipeline_cfg([])


def test_root_pipeline_config_accepts_explicit_stage_override():
    cfg = _compose_pipeline_cfg(["dataset=stage2/libriheavy_large", "stage=convert"])

    assert cfg.stage == "convert"
    assert cfg.dataset.name == "libriheavy_large"


def test_plan_without_stage_override_inspects_all_stages(monkeypatch):
    cfg = _compose_pipeline_cfg(["dataset=stage2/libriheavy_large"])
    seen: dict[str, object] = {}

    def _fake_plan(_spec, *, stage):
        seen["stage"] = stage
        return {"convert": {}, "tokenize": {}, "materialize": {}}

    monkeypatch.setattr(audio_main, "plan_stages", _fake_plan)

    result = audio_main._execute_command("plan", cfg)

    assert seen["stage"] is None
    assert result["stage"] is None
    assert set(result["stages"]) == {"convert", "tokenize", "materialize"}


def test_run_without_stage_override_fails_before_defaulting_to_convert(monkeypatch):
    cfg = _compose_pipeline_cfg(["dataset=stage2/libriheavy_large"])

    def _fake_run(_spec, *, stage, overwrite):
        del overwrite
        if stage is None:
            raise ValueError("run requires stage=<convert|tokenize|materialize>")
        raise AssertionError(f"unexpected stage default: {stage!r}")

    monkeypatch.setattr(audio_main, "run_stages", _fake_run)

    with pytest.raises(ValueError, match="run requires stage"):
        audio_main._execute_command("run", cfg)


def test_unsupported_family_raises_at_load_time():
    with pytest.raises(ValueError, match="Unsupported convert family"):
        load_dataset_spec(
            {
                "name": "bogus",
                "convert": {
                    "family": "not_a_family",
                    "input": {},
                    "output": {"shar_dir": "/out/bogus"},
                },
            }
        )
