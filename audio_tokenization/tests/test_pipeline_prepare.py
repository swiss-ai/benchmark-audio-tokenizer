import json
import copy
import gzip
import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from audio_tokenization.__main__ import _split_command
from audio_tokenization.config import load_dataset_spec
from audio_tokenization.config.schema import PrepareMetadataSpec, PrepareSpec
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
from audio_tokenization.prepare.constants import (
    CURRENT_PREPARE_STATE_VERSION,
    CURRENT_STAGE_STATE_VERSION,
    PREPARE_STATE_FILE,
)
from audio_tokenization.prepare.runtime import (
    read_prepare_state,
    validate_or_write_prepare_state,
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
    infore2 = load_dataset_spec(_load_dataset_cfg("cooldown_infore2"))
    aozora = load_dataset_spec(_load_dataset_cfg("aozora_hurigana"))

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
# state versioning
# ---------------------------------------------------------------------------


def test_prepare_state_rejects_unversioned_payload(tmp_path):
    state_path = tmp_path / "_PREPARE_STATE.json"
    state_path.write_text(
        json.dumps(
            {
                "parquet_dir": "/tmp/data",
                "text_tokenizer": None,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="has no version"):
        read_prepare_state(state_path)


def test_prepare_state_rejects_stale_version(tmp_path):
    state_path = tmp_path / "_PREPARE_STATE.json"
    state_path.write_text(json.dumps({"version": 0, "parquet_dir": "/tmp/data"}) + "\n")

    with pytest.raises(RuntimeError, match="stale version"):
        read_prepare_state(state_path)


def test_validate_or_write_prepare_state_rejects_missing_invariant(tmp_path):
    state_path = tmp_path / "stage_state.json"
    stale_payload = {
        "version": CURRENT_STAGE_STATE_VERSION,
        "parquet_dir": "/tmp/data",
        "text_tokenizer": None,
    }
    state_path.write_text(json.dumps(stale_payload) + "\n")

    with pytest.raises(AssertionError, match="missing a required invariant"):
        validate_or_write_prepare_state(
            state_path,
            expected={**stale_payload, "metadata.text_column": "text"},
            invariant_keys=("parquet_dir", "metadata.text_column"),
            guidance="remove output dir to reset",
        )


def test_prepare_state_rejects_unknown_future_version(tmp_path):
    state_path = tmp_path / "_PREPARE_STATE.json"
    state_path.write_text(
        json.dumps({"version": CURRENT_PREPARE_STATE_VERSION + 99}) + "\n"
    )
    with pytest.raises(RuntimeError, match="only knows how to read up to"):
        read_prepare_state(state_path)


def test_validate_or_write_prepare_state_first_run_writes_versioned(tmp_path):
    state_path = tmp_path / "stage_state.json"
    expected = {"parquet_dir": "/tmp/data", "text_tokenizer": None}

    wrote = validate_or_write_prepare_state(
        state_path,
        expected=expected,
        invariant_keys=("parquet_dir", "text_tokenizer"),
        guidance="remove the output dir to restart fresh.",
    )
    assert wrote is True

    payload = json.loads(state_path.read_text())
    assert payload["version"] == CURRENT_STAGE_STATE_VERSION
    assert CURRENT_STAGE_STATE_VERSION != CURRENT_PREPARE_STATE_VERSION
    assert payload["parquet_dir"] == "/tmp/data"


def test_write_prepare_state_for_spec_uses_prepare_specific_version(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    spec = _minimal_parquet_spec(tmp_path)

    _write_state_via_runtime_helper(spec)

    payload = json.loads((out / PREPARE_STATE_FILE).read_text())
    assert payload["version"] == CURRENT_PREPARE_STATE_VERSION


def test_validate_or_write_prepare_state_detects_invariant_drift(tmp_path):
    state_path = tmp_path / "stage_state.json"
    validate_or_write_prepare_state(
        state_path,
        expected={"parquet_dir": "/tmp/data"},
        invariant_keys=("parquet_dir",),
        guidance="remove output dir to reset",
    )

    with pytest.raises(AssertionError, match="Unsafe resume"):
        validate_or_write_prepare_state(
            state_path,
            expected={"parquet_dir": "/tmp/OTHER"},
            invariant_keys=("parquet_dir",),
            guidance="remove output dir to reset",
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


def _write_state_via_runtime_helper(spec):
    """Test wrapper around the hoisted ``write_prepare_state_for_spec``.
    All five family runners now delegate to this single helper, so the
    parametrized drift tests no longer need a per-family writer map."""
    from audio_tokenization.prepare.runtime import write_prepare_state_for_spec
    return write_prepare_state_for_spec(spec)


_PREPARE_STATE_WRITERS = {
    family: _write_state_via_runtime_helper
    for family in ("parquet", "hf", "wds", "audio_dir", "lhotse_recipe")
}


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


@pytest.mark.parametrize(
    ("flag", "v1", "v2"),
    [
        ("text_column", None, "transcription"),
        ("id_column", "row_id", ["session", "seg"]),
        ("language", "ja", "vi"),
    ],
)
def test_e2e_parquet_prepare_state_rejects_drift(tmp_path, flag, v1, v2):
    """Resume with any output-affecting field changed must be rejected by
    the real parquet backend, not just the isolated helper. v1=None means
    omit the field for the first run."""
    out = tmp_path / "out"
    out.mkdir()

    spec_v1 = (
        _minimal_parquet_spec(tmp_path)
        if v1 is None
        else _minimal_parquet_spec(tmp_path, **{flag: v1})
    )
    _write_state_via_runtime_helper(spec_v1)

    spec_v2 = _minimal_parquet_spec(tmp_path, **{flag: v2})

    with pytest.raises(
        AssertionError, match=rf"(?s)Unsafe resume.*Key: metadata\.{re.escape(flag)}"
    ):
        _write_state_via_runtime_helper(spec_v2)


def test_e2e_parquet_prepare_state_rejects_missing_expanded_invariant(tmp_path):
    """Current state files must contain the full invariant set.

    If a previous run lacks a newly required field, we rebuild instead of
    backfilling it silently.
    """
    out = tmp_path / "out"
    out.mkdir()

    spec_v1 = _minimal_parquet_spec(tmp_path, text_column="transcription")
    stale_state = {
        "version": CURRENT_PREPARE_STATE_VERSION,
        **spec_v1.fingerprint_payload(),
    }
    stale_state.pop("metadata.text_column")
    (out / PREPARE_STATE_FILE).write_text(json.dumps(stale_state) + "\n")

    with pytest.raises(AssertionError, match=r"(?s)Unsafe resume.*metadata\.text_column"):
        _write_state_via_runtime_helper(spec_v1)


@pytest.mark.parametrize(
    ("family", "path", "old", "new", "expected_key"),
    [
        ("parquet", ("metadata", "text_column"), None, "transcription", "metadata.text_column"),
        ("hf", ("input", "arrow_glob"), "*.arrow", "train-*.arrow", "input.arrow_glob"),
        ("wds", ("input", "vad_max_chunk_sec"), 200.0, 123.0, "input.vad_max_chunk_sec"),
        ("audio_dir", ("input", "vad_max_chunk_sec"), 200.0, 123.0, "input.vad_max_chunk_sec"),
        ("lhotse_recipe", ("input", "recipe_kwargs"), "{}", '{"lang":"en"}', "input.recipe_kwargs"),
    ],
)
def test_prepare_state_drift_detected_for_each_family(
    tmp_path, family, path, old, new, expected_key
):
    out = tmp_path / family / "out"
    out.mkdir(parents=True)

    payload_v1 = _minimal_payload(family, shar_dir=str(out))
    if old is not None:
        _set_convert_field(payload_v1, path, old)
    spec_v1 = PrepareSpec.from_mapping(payload_v1["convert"])
    _PREPARE_STATE_WRITERS[family](spec_v1)

    payload_v2 = copy.deepcopy(payload_v1)
    _set_convert_field(payload_v2, path, new)
    spec_v2 = PrepareSpec.from_mapping(payload_v2["convert"])

    with pytest.raises(
        AssertionError,
        match=rf"(?s)Unsafe resume.*Key: {re.escape(expected_key)}",
    ):
        _PREPARE_STATE_WRITERS[family](spec_v2)


@pytest.mark.parametrize("family", ["parquet", "hf", "wds", "audio_dir"])
def test_prepare_state_ignores_operational_knobs(tmp_path, family):
    out = tmp_path / family / "out"
    out.mkdir(parents=True)

    payload_v1 = _minimal_payload(family, shar_dir=str(out))
    spec_v1 = PrepareSpec.from_mapping(payload_v1["convert"])
    _PREPARE_STATE_WRITERS[family](spec_v1)

    payload_v2 = copy.deepcopy(payload_v1)
    payload_v2["convert"]["output"]["num_workers"] = 123
    spec_v2 = PrepareSpec.from_mapping(payload_v2["convert"])

    _PREPARE_STATE_WRITERS[family](spec_v2)


def test_lhotse_recipe_prepare_state_rejects_num_workers_drift(tmp_path):
    out = tmp_path / "lhotse_recipe" / "out"
    out.mkdir(parents=True)

    payload_v1 = _minimal_payload("lhotse_recipe", shar_dir=str(out))
    spec_v1 = PrepareSpec.from_mapping(payload_v1["convert"])
    _PREPARE_STATE_WRITERS["lhotse_recipe"](spec_v1)

    payload_v2 = copy.deepcopy(payload_v1)
    payload_v2["convert"]["output"]["num_workers"] = 123
    spec_v2 = PrepareSpec.from_mapping(payload_v2["convert"])

    with pytest.raises(
        AssertionError,
        match=r"(?s)Unsafe resume.*Key: output\.num_workers",
    ):
        _PREPARE_STATE_WRITERS["lhotse_recipe"](spec_v2)


def test_audio_dir_prepare_does_not_write_state_before_audio_index_succeeds(
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

    assert not (out / PREPARE_STATE_FILE).exists()


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
    fingerprint stays stable across nodes with different ``SLURM_CPUS_PER_TASK``
    — lhotse_recipe is the only family that includes num_workers in the
    resume fingerprint."""
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


@pytest.mark.parametrize(
    ("family", "path", "value"),
    [
        ("hf", ("metadata", "duration_column"), "duration"),
        ("wds", ("metadata", "audio_column"), "audio"),
        ("audio_dir", ("metadata", "text_column"), "transcription"),
        ("lhotse_recipe", ("output", "resampling_backend"), None),
    ],
)
def test_prepare_state_ignores_unused_family_specific_knobs(tmp_path, family, path, value):
    out = tmp_path / family / "out"
    out.mkdir(parents=True)

    payload_v1 = _minimal_payload(family, shar_dir=str(out))
    spec_v1 = PrepareSpec.from_mapping(payload_v1["convert"])
    _PREPARE_STATE_WRITERS[family](spec_v1)

    payload_v2 = copy.deepcopy(payload_v1)
    _set_convert_field(payload_v2, path, value)
    spec_v2 = PrepareSpec.from_mapping(payload_v2["convert"])

    _PREPARE_STATE_WRITERS[family](spec_v2)


# ---------------------------------------------------------------------------
# Per-family CLI smoke: each script's _args_to_spec(args) produces a valid
# PrepareSpec for typical CLI invocations. Replaces the old
# "namespace-parity" tests — under typed runners there's no Namespace to
# compare; both Hydra and CLI converge on PrepareSpec via schema validation.
# ---------------------------------------------------------------------------


def test_parquet_cli_args_round_trip_to_spec():
    args = prepare_parquet_to_shar.build_parser().parse_args([
        "--parquet-dir", "/data/p",
        "--shar-dir", "/out/p",
    ])
    spec = prepare_parquet_to_shar._args_to_spec(args)
    assert spec.family == "parquet"
    assert spec.input.parquet_dir == "/data/p"
    assert spec.output.shar_dir == "/out/p"


def test_hf_cli_args_round_trip_to_spec():
    args = prepare_hf_to_shar.build_parser().parse_args([
        "--arrow-dir", "/data/a",
        "--shar-dir", "/out/a",
    ])
    spec = prepare_hf_to_shar._args_to_spec(args)
    assert spec.family == "hf"
    assert spec.input.arrow_dir == "/data/a"
    assert spec.output.shar_dir == "/out/a"


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
    monkeypatch.setattr(prepare_parquet_to_shar, "write_prepare_state_for_spec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prepare_parquet_to_shar, "ensure_worker_assignment", lambda *_args, **_kwargs: 1)
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
    monkeypatch.setattr(prepare_hf_to_shar, "write_prepare_state_for_spec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prepare_hf_to_shar, "ensure_worker_assignment", lambda *_args, **_kwargs: 1)
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
    monkeypatch.setattr(prepare_audio_dir_to_shar, "write_prepare_state_for_spec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prepare_audio_dir_to_shar, "ensure_worker_assignment", lambda *_args, **_kwargs: 1)
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
    assert len(captured["worker_args"][0]) == 15
    assert prepare_audio_dir_to_shar._AUDIO_INDEX is None


def test_wds_cli_args_round_trip_to_spec():
    args = prepare_wds_to_shar.build_parser().parse_args([
        "--wds-shards", "/data/*.tar",
        "--shar-dir", "/out/w",
    ])
    spec = prepare_wds_to_shar._args_to_spec(args)
    assert spec.family == "wds"
    assert spec.input.wds_shards == ["/data/*.tar"]
    assert spec.output.shar_dir == "/out/w"


def test_audio_dir_cli_args_round_trip_to_spec():
    args = prepare_audio_dir_to_shar.build_parser().parse_args([
        "--audio-root", "/data/audio",
        "--jsonl-files", "/data/v.jsonl",
        "--shar-dir", "/out/d",
    ])
    spec = prepare_audio_dir_to_shar._args_to_spec(args)
    assert spec.family == "audio_dir"
    assert spec.input.audio_root == "/data/audio"
    assert spec.input.jsonl_files == ["/data/v.jsonl"]
    assert spec.output.shar_dir == "/out/d"


def test_lhotse_recipe_cli_args_round_trip_to_spec(tmp_path):
    args = prepare_lhotse_recipe_to_shar.build_parser().parse_args([
        "--recipe", "librispeech",
        "--corpus_dir", str(tmp_path),
        "--split", "test-clean",
        "--shar_output_dir", str(tmp_path / "out"),
    ])
    spec = prepare_lhotse_recipe_to_shar._args_to_spec(args)
    assert spec.family == "lhotse_recipe"
    assert spec.input.recipe == "librispeech"
    assert spec.input.split == "test-clean"
    assert spec.output.shar_dir == str(tmp_path / "out")


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


def test_run_convert_resume_false_removes_existing_output_dir(tmp_path, monkeypatch):
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
        def run(prepare_spec):
            captured["exists_on_entry"] = Path(prepare_spec.output.shar_dir).exists()
            _write_cut_shar_index(Path(prepare_spec.output.shar_dir))
            return {"ran": True}

    monkeypatch.setattr(
        convert_stage.importlib,
        "import_module",
        lambda _path: _RunnerModule,
    )
    monkeypatch.setattr(convert_stage, "validate_prepare_runtime", lambda **_kwargs: None)

    result = convert_stage.run_convert(spec, resume=False)
    assert result == {"ran": True}
    assert captured["exists_on_entry"] is False
    assert (out / "_shar_work_manifest.json").is_file()


def test_split_command_defaults_to_run():
    assert _split_command(["dataset=infore2"]) == ("run", ["dataset=infore2"])


def test_split_command_explicit_status():
    assert _split_command(["status", "dataset=infore2"]) == (
        "status",
        ["dataset=infore2"],
    )


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
