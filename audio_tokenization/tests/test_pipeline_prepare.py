import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from audio_tokenization.config import load_dataset_spec
from audio_tokenization.config.schema import (
    PrepareMetadataSpec,
    build_parquet_prepare_fingerprint,
)
from audio_tokenization.stages import prepare as prepare_stage
from audio_tokenization.prepare import prepare_parquet_to_shar
from audio_tokenization.prepare.constants import (
    CURRENT_PREPARE_STATE_VERSION,
    PREPARE_STATE_FILE,
)
from audio_tokenization.prepare.runtime import (
    read_prepare_state,
    validate_or_write_prepare_state,
)


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


# ---------------------------------------------------------------------------
# text_column must default to "text" on omit, support null for unsupervised
# ---------------------------------------------------------------------------


def test_text_column_defaults_to_text_when_key_omitted():
    """Omitting text_column must match legacy CLI default of "text"; otherwise
    _convert_worker writes cuts without supervisions and silently drops all
    transcripts.
    """
    spec = load_dataset_spec(
        {
            "name": "omitted_text",
            "prepare": {
                "family": "parquet",
                "input": {"parquet_dir": "/tmp/data"},
                "output": {"shar_dir": "/tmp/out"},
                "metadata": {},
            },
        }
    )
    assert spec.prepare.metadata.text_column == "text"


def test_text_column_explicit_null_means_disabled():
    """Truly unsupervised datasets can opt out with `text_column: null`."""
    spec = load_dataset_spec(
        {
            "name": "unsupervised",
            "prepare": {
                "family": "parquet",
                "input": {"parquet_dir": "/tmp/data"},
                "output": {"shar_dir": "/tmp/out"},
                "metadata": {"text_column": None},
            },
        }
    )
    assert spec.prepare.metadata.text_column is None


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
            "prepare": {
                "family": "parquet",
                "input": {"parquet_dir": "/tmp/data"},
                "output": {"shar_dir": "/tmp/out"},
                "metadata": {"id_column": yaml_value},
            },
        }
    )
    assert spec.prepare.metadata.id_column == expected


def test_id_column_rejects_invalid_shape():
    with pytest.raises(TypeError, match="id_column"):
        load_dataset_spec(
            {
                "name": "bad_id",
                "prepare": {
                    "family": "parquet",
                    "input": {"parquet_dir": "/tmp/data"},
                    "output": {"shar_dir": "/tmp/out"},
                    "metadata": {"id_column": 42},
                },
            }
        )


def test_id_column_list_threads_through_namespace():
    """Composite id_column flows through build_prepare_namespace as a list so
    extract_row_metadata's argparse-nargs path joins with '_'.
    """
    spec = load_dataset_spec(
        {
            "name": "composite",
            "prepare": {
                "family": "parquet",
                "input": {"parquet_dir": "/tmp/data"},
                "output": {"shar_dir": "/tmp/out"},
                "metadata": {"id_column": ["session", "seg"]},
            },
        }
    )
    ns = prepare_stage.build_prepare_namespace(spec)
    assert ns.id_column == ["session", "seg"]


def test_id_column_single_string_wrapped_in_list_for_cli():
    spec = load_dataset_spec(
        {
            "name": "single",
            "prepare": {
                "family": "parquet",
                "input": {"parquet_dir": "/tmp/data"},
                "output": {"shar_dir": "/tmp/out"},
                "metadata": {"id_column": "sample_id"},
            },
        }
    )
    ns = prepare_stage.build_prepare_namespace(spec)
    # argparse nargs="*" stores a list; stage wrapper matches that shape.
    assert ns.id_column == ["sample_id"]


# ---------------------------------------------------------------------------
# state versioning + auto-upgrade of legacy v0 payloads
# ---------------------------------------------------------------------------


def test_prepare_state_auto_upgrades_legacy_v0_in_place(tmp_path):
    """Unversioned state from the pre-refactor CLI must be accepted and
    upgraded transparently so long-running resumes don't fail.
    """
    state_path = tmp_path / "_PREPARE_STATE.json"
    legacy_payload = {
        "parquet_dir": "/tmp/data",
        "text_tokenizer": None,
        "input_clip_id_parser": None,
        "external_metadata": None,
        "id_field": "id",
        "text_field": "text",
        "custom_fields": None,
    }
    state_path.write_text(json.dumps(legacy_payload) + "\n")

    upgraded = read_prepare_state(state_path)

    assert upgraded["version"] == CURRENT_PREPARE_STATE_VERSION
    # Payload is preserved key-for-key apart from the new version sentinel.
    for k, v in legacy_payload.items():
        assert upgraded[k] == v

    # The upgrade was persisted: reading again returns a versioned payload
    # directly without re-migrating.
    rewritten = json.loads(state_path.read_text())
    assert rewritten["version"] == CURRENT_PREPARE_STATE_VERSION


def test_prepare_state_rejects_unknown_future_version(tmp_path):
    state_path = tmp_path / "_PREPARE_STATE.json"
    state_path.write_text(
        json.dumps({"version": CURRENT_PREPARE_STATE_VERSION + 99}) + "\n"
    )
    with pytest.raises(RuntimeError, match="only knows how to read up to"):
        read_prepare_state(state_path)


def test_validate_or_write_prepare_state_first_run_writes_versioned(tmp_path):
    state_path = tmp_path / "_PREPARE_STATE.json"
    expected = {"parquet_dir": "/tmp/data", "text_tokenizer": None}

    wrote = validate_or_write_prepare_state(
        state_path,
        expected=expected,
        invariant_keys=("parquet_dir", "text_tokenizer"),
        guidance="remove the output dir to restart fresh.",
    )
    assert wrote is True

    payload = json.loads(state_path.read_text())
    assert payload["version"] == CURRENT_PREPARE_STATE_VERSION
    assert payload["parquet_dir"] == "/tmp/data"


def test_validate_or_write_prepare_state_resume_after_v0_upgrade(tmp_path):
    """Legacy v0 state resumes cleanly when invariants still match."""
    state_path = tmp_path / "_PREPARE_STATE.json"
    legacy_payload = {
        "parquet_dir": "/tmp/data",
        "text_tokenizer": None,
        "input_clip_id_parser": None,
        "external_metadata": None,
        "id_field": "id",
        "text_field": "text",
        "custom_fields": None,
    }
    state_path.write_text(json.dumps(legacy_payload) + "\n")

    # Resume with identical expected values succeeds (no AssertionError).
    wrote = validate_or_write_prepare_state(
        state_path,
        expected=legacy_payload,
        invariant_keys=tuple(legacy_payload.keys()),
        guidance="remove output dir to reset",
    )
    assert wrote is False

    # File now includes the version sentinel.
    payload = json.loads(state_path.read_text())
    assert payload["version"] == CURRENT_PREPARE_STATE_VERSION


def test_validate_or_write_prepare_state_detects_invariant_drift(tmp_path):
    state_path = tmp_path / "_PREPARE_STATE.json"
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
# CLI-schema parity: schema defaults must not drift from argparse defaults
# ---------------------------------------------------------------------------


def test_prepare_metadata_spec_defaults_match_legacy_parquet_cli():
    """If this test fails, the schema default for some field has drifted
    from the legacy argparse default; pick one and fix the other.
    """
    parser = prepare_parquet_to_shar.build_parser()
    cli_defaults = {action.dest: action.default for action in parser._actions}

    spec_defaults = PrepareMetadataSpec()
    # Each pairing: (argparse dest, dataclass field) — keep in lockstep.
    pairs = [
        ("audio_column", "audio_column"),
        ("text_column", "text_column"),
        ("duration_column", "duration_column"),
        ("language_column", "language_column"),
        ("language", "language"),
        ("input_clip_id_parser", "input_clip_id_parser"),
        ("external_metadata", "external_metadata"),
        ("id_field", "id_field"),
        ("text_field", "text_field"),
    ]
    mismatches = []
    for cli_name, schema_name in pairs:
        cli_val = cli_defaults.get(cli_name, "<MISSING>")
        schema_val = getattr(spec_defaults, schema_name)
        if cli_val != schema_val:
            mismatches.append(
                f"  {cli_name}: CLI default {cli_val!r} vs schema default {schema_val!r}"
            )
    assert not mismatches, (
        "schema/CLI default drift detected:\n" + "\n".join(mismatches)
    )


def test_prepare_spec_fingerprint_payload_is_hashable_and_stable():
    """The fingerprint dict must be built from JSON-round-trippable types so
    state comparison is deterministic. Lists are normalised to sorted lists.
    """
    spec = load_dataset_spec(_load_dataset_cfg("aozora_hurigana"))
    payload = spec.prepare.fingerprint_payload()
    # must JSON round-trip byte-identically so stored-vs-current compare works
    dumped = json.dumps(payload, sort_keys=True)
    assert payload == json.loads(dumped)
    assert "sample_id" in dumped
    # sorted-list normalisation for list fields
    assert payload["custom_columns"] == sorted(
        ["author", "work", "rendition", "line_num"]
    )


# ---------------------------------------------------------------------------
# End-to-end: config-driven and CLI-driven paths produce identical fingerprints
# and the expanded invariant set actually invalidates resume on drift.
# ---------------------------------------------------------------------------


def _argv_from_spec(spec):
    """Build a minimal argv list equivalent to the given DatasetSpec so the
    legacy parquet CLI parser produces the same argparse Namespace that the
    stage wrapper would."""
    m = spec.prepare.metadata
    o = spec.prepare.output
    argv = [
        "--parquet-dir", spec.prepare.input.parquet_dir,
        "--parquet-glob", spec.prepare.input.parquet_glob,
        "--shar-dir", o.shar_dir,
        "--shard-size", str(o.shard_size),
        "--shar-format", o.shar_format,
        "--target-sr", str(o.target_sr),
        "--num-workers", str(o.num_workers),
    ]
    if o.resampling_backend:
        argv += ["--resampling-backend", o.resampling_backend]
    if o.text_tokenizer:
        argv += ["--text-tokenizer", o.text_tokenizer]
    if m.text_column is not None:
        argv += ["--text-column", m.text_column]
    else:
        # argparse has no way to set --text-column to None; the CLI's default
        # is "text", so "explicit null" is not expressible on the CLI. This
        # only affects DatasetSpec-only test cases.
        pass
    if m.duration_column is not None:
        argv += ["--duration-column", m.duration_column]
    if m.language:
        argv += ["--language", m.language]
    if m.input_clip_id_parser:
        argv += ["--input-clip-id-parser", m.input_clip_id_parser]
    if isinstance(m.id_column, list):
        argv += ["--id-column", *m.id_column]
    elif isinstance(m.id_column, str):
        argv += ["--id-column", m.id_column]
    if m.custom_columns:
        argv += ["--custom-columns", *m.custom_columns]
    return argv


@pytest.mark.parametrize("dataset_name", ["infore2", "aozora_hurigana"])
def test_fingerprint_parity_config_vs_cli(dataset_name):
    """DatasetSpec-driven and argparse-driven fingerprints must agree for
    equivalent inputs. aozora exercises composite-list id_column.
    """
    spec = load_dataset_spec(_load_dataset_cfg(dataset_name))
    argv = _argv_from_spec(spec)
    cli_ns = prepare_parquet_to_shar.build_parser().parse_args(argv)

    cli_fp = build_parquet_prepare_fingerprint(cli_ns)
    spec_fp = spec.prepare.fingerprint_payload()

    assert cli_fp == spec_fp


def _minimal_argparse_ns(tmp_path, **overrides):
    """Build a parsed argparse Namespace with sensible defaults for tests
    that drive prepare_parquet_to_shar._validate_or_write_prepare_state."""
    argv = [
        "--parquet-dir", str(tmp_path / "in"),
        "--shar-dir", str(tmp_path / "out"),
    ]
    for flag, value in overrides.items():
        cli_flag = "--" + flag.replace("_", "-")
        if isinstance(value, list):
            argv += [cli_flag, *value]
        else:
            argv += [cli_flag, str(value)]
    return prepare_parquet_to_shar.build_parser().parse_args(argv)


@pytest.mark.parametrize(
    ("flag", "v1", "v2"),
    [
        ("text_column", None, "transcription"),
        ("id_column", "row_id", ["session", "seg"]),
        ("language", "ja", "vi"),
    ],
)
def test_e2e_parquet_prepare_state_rejects_drift(tmp_path, flag, v1, v2):
    """Resume with any output-affecting field changed must be rejected by the
    real parquet backend, not just the isolated helper. v1=None means rely on
    the CLI default for the first run.
    """
    out = tmp_path / "out"
    out.mkdir()

    args_v1 = (
        _minimal_argparse_ns(tmp_path)
        if v1 is None
        else _minimal_argparse_ns(tmp_path, **{flag: v1})
    )
    args_v1.shar_dir = out
    prepare_parquet_to_shar._validate_or_write_prepare_state(args_v1)

    args_v2 = _minimal_argparse_ns(tmp_path, **{flag: v2})
    args_v2.shar_dir = out

    with pytest.raises(AssertionError, match=rf"(?s)Unsafe resume.*Key: {flag}"):
        prepare_parquet_to_shar._validate_or_write_prepare_state(args_v2)


def test_e2e_parquet_prepare_state_v0_legacy_backfills_and_then_enforces(tmp_path):
    """Legacy v0 state (missing expanded invariants) must resume cleanly,
    backfill current values, and enforce drift on the NEXT run.
    """
    out = tmp_path / "out"
    out.mkdir()

    # Simulate a pre-refactor state file with only the old narrow invariants.
    legacy_state = {
        "parquet_dir": str((tmp_path / "in").resolve()),
        "text_tokenizer": None,
        "input_clip_id_parser": None,
        "external_metadata": None,
        "id_field": "id",
        "text_field": "text",
        "custom_fields": None,
    }
    (out / PREPARE_STATE_FILE).write_text(json.dumps(legacy_state) + "\n")

    # First resume with text_column="transcription" succeeds (backfill).
    args_v1 = _minimal_argparse_ns(tmp_path, text_column="transcription")
    args_v1.shar_dir = out
    prepare_parquet_to_shar._validate_or_write_prepare_state(args_v1)

    # The backfilled value is now locked in.
    stored = json.loads((out / PREPARE_STATE_FILE).read_text())
    assert stored["text_column"] == "transcription"
    assert stored["version"] == CURRENT_PREPARE_STATE_VERSION

    # Second resume with a different text_column is rejected.
    args_v2 = _minimal_argparse_ns(tmp_path, text_column="body")
    args_v2.shar_dir = out
    with pytest.raises(AssertionError, match=r"(?s)Unsafe resume.*Key: text_column"):
        prepare_parquet_to_shar._validate_or_write_prepare_state(args_v2)
