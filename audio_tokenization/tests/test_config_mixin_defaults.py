"""Mixin defaults must mirror schema defaults exactly.

Each stage profile YAML lists default values explicitly so a user can
discover them by reading YAML rather than grepping ``schema.py``. Pydantic
defaults remain authoritative for python-side construction (tests build
``DatasetSpec`` from minimal payloads). This test guards the contract: if
either side changes a default, the two must move together.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from audio_tokenization.config.schema import (
    InterleaveProductSpec,
    PrepareMetadataSpec,
    PrepareOutputSpec,
    TokenizeDataloaderSpec,
    TokenizeFilterSpec,
    TokenizeOutputSpec,
    TokenizerSpec,
    _INPUT_SPEC_BY_FAMILY,
)


_PROFILES = Path(__file__).resolve().parents[1] / "configs" / "pipeline"


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((_PROFILES / name).read_text())


def _schema_defaults(model_cls) -> dict[str, Any]:
    """Return the optional-field defaults of *model_cls* (required fields skipped)."""
    defaults: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        if info.is_required():
            continue
        defaults[name] = info.default_factory() if info.default_factory is not None else info.default
    return defaults


def _assert_subset(
    yaml_block: Mapping[str, Any],
    expected: Mapping[str, Any],
    where: str,
    site_overrides: tuple[str, ...] = (),
):
    """Each key in ``expected`` must appear in *yaml_block* with the same value.

    *site_overrides* names keys whose mixin value is a deliberate
    cluster/site-specific override of the schema default (e.g.
    ``text_tokenizer`` pointing at the Alps wheelhouse path). We only check
    that they're present, not that they equal the schema default.
    """
    for key, want in expected.items():
        assert key in yaml_block, f"{where}: mixin missing key {key!r}"
        if key in site_overrides:
            continue
        got = yaml_block[key]
        assert got == want, f"{where}: mixin[{key!r}]={got!r} != schema default {want!r}"


_TOKENIZE_MIXINS = [
    "tokenize/audio_only.yaml",
    "tokenize/audio_text_direct.yaml",
    "tokenize/audio_text_interleaved.yaml",
]


# Single source of truth: schema dictates the family list. Any new convert
# family added to ``_INPUT_SPEC_BY_FAMILY`` is automatically picked up here,
# so the parity check covers it on day one.
_CONVERT_FAMILY_INPUT_SPECS = sorted(
    (f"convert/{family}.yaml", spec_cls)
    for family, spec_cls in _INPUT_SPEC_BY_FAMILY.items()
)


def test_convert_common_mirrors_output_and_metadata_defaults():
    """convert/_common.yaml is the single source of schema-mirror truth.

    Family YAMLs (parquet/hf/wds/audio_dir/lhotse_recipe) compose this base via
    Hydra ``defaults`` and add only family-specific overrides. The YAML stores
    keys at top level (Hydra group convention); downstream consumers see the
    composed result wrapped under the ``convert`` package.
    """
    common = _load("convert/_common.yaml")
    _assert_subset(
        common["output"], _schema_defaults(PrepareOutputSpec),
        "convert/_common.yaml output",
    )
    _assert_subset(
        common["metadata"], _schema_defaults(PrepareMetadataSpec),
        "convert/_common.yaml metadata",
    )


def test_materialize_interleave_preset_mirrors_schema_defaults():
    materialize = _load("materialize/interleave.yaml")
    expected = _schema_defaults(InterleaveProductSpec)
    expected["enabled"] = True
    _assert_subset(
        materialize["materialize"]["interleave"], expected,
        "materialize/interleave.yaml materialize.interleave",
    )


@pytest.mark.parametrize(
    "family",
    [Path(name).stem for name, _ in _CONVERT_FAMILY_INPUT_SPECS],
)
def test_convert_family_compose_includes_common_defaults(family):
    """Hydra-compose each family; the merged ``convert.metadata`` and
    ``convert.output`` must carry every schema-default key from ``_common``.

    A real compose, not a YAML-string check: catches a missing
    ``defaults: [_common, _self_]`` AND any future restructuring that
    breaks the family→_common composition chain.
    """
    from hydra import compose, initialize_config_dir

    metadata_required = set(PrepareMetadataSpec.model_fields.keys())
    # shar_dir is the only required output field (no default), so it
    # legitimately doesn't appear in the YAML defaults.
    output_required = {
        name for name, info in PrepareOutputSpec.model_fields.items()
        if not info.is_required()
    }

    with initialize_config_dir(version_base=None, config_dir=str(_PROFILES.resolve())):
        cfg = compose(config_name=f"convert/{family}")

    metadata = set(cfg.convert.metadata.keys())
    output = set(cfg.convert.output.keys())

    missing_metadata = metadata_required - metadata
    missing_output = output_required - output

    assert not missing_metadata, (
        f"convert/{family}.yaml compose is missing metadata defaults from _common: "
        f"{sorted(missing_metadata)}. Either add `defaults: [_common, _self_]` to the "
        "family YAML, or extend _common.yaml to mirror the schema."
    )
    assert not missing_output, (
        f"convert/{family}.yaml compose is missing output defaults from _common: "
        f"{sorted(missing_output)}."
    )


@pytest.mark.parametrize("mixin_filename,input_cls", _CONVERT_FAMILY_INPUT_SPECS)
def test_source_mixin_mirrors_input_defaults(mixin_filename, input_cls):
    _assert_subset(
        _load(mixin_filename)["input"],
        _schema_defaults(input_cls),
        f"{mixin_filename} input",
    )


@pytest.mark.parametrize("mixin_filename", _TOKENIZE_MIXINS)
def test_tokenize_mixin_mirrors_subspec_defaults(mixin_filename):
    tk = _load(mixin_filename)["tokenize"]
    for section, cls in (
        ("tokenizer", TokenizerSpec),
        ("filter", TokenizeFilterSpec),
        ("dataloader", TokenizeDataloaderSpec),
        ("output", TokenizeOutputSpec),
    ):
        _assert_subset(tk[section], _schema_defaults(cls), f"{mixin_filename} tokenize.{section}")


def test_tokenize_mixins_agree_on_shared_defaults():
    """All three tokenize_*.yaml mixins must carry identical default blocks.

    Without this, schema-vs-YAML parity could pass for one mixin while the
    other two silently drift. Only the semantic header (mode, audio_text_*)
    is allowed to differ.
    """
    semantic_keys = {"mode", "audio_text_format", "audio_text_task"}
    bodies = {f: {k: v for k, v in _load(f)["tokenize"].items() if k not in semantic_keys}
              for f in _TOKENIZE_MIXINS}
    reference_file = _TOKENIZE_MIXINS[0]
    reference = bodies[reference_file]
    for f, body in bodies.items():
        assert body == reference, (
            f"{f} tokenize defaults disagree with {reference_file}: "
            f"diff_keys={set(body.keys()) ^ set(reference.keys()) or 'value-level'}"
        )
