"""Mixin defaults must mirror schema defaults exactly.

Each ``_common/*.yaml`` mixin lists default values explicitly so a user can
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
    PrepareAudioDirInputSpec,
    PrepareHfInputSpec,
    PrepareLhotseRecipeInputSpec,
    PrepareMetadataSpec,
    PrepareOutputSpec,
    PrepareParquetInputSpec,
    PrepareWdsInputSpec,
    TokenizeDataloaderSpec,
    TokenizeFilterSpec,
    TokenizeOutputSpec,
    TokenizerSpec,
)


_COMMON = Path(__file__).resolve().parents[1] / "configs" / "pipeline" / "dataset" / "_common"


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((_COMMON / name).read_text())


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
    "tokenize_audio_only.yaml",
    "tokenize_audio_text_direct.yaml",
    "tokenize_audio_text_interleaved.yaml",
]


def test_pipeline_mixin_mirrors_convert_output_and_metadata_defaults():
    pipeline = _load("pipeline.yaml")
    _assert_subset(
        pipeline["convert"]["output"], _schema_defaults(PrepareOutputSpec),
        "pipeline.yaml convert.output",
        site_overrides=("text_tokenizer",),
    )
    _assert_subset(pipeline["convert"]["metadata"], _schema_defaults(PrepareMetadataSpec),
                   "pipeline.yaml convert.metadata")
    _assert_subset(pipeline["materialize"]["interleave"], _schema_defaults(InterleaveProductSpec),
                   "pipeline.yaml materialize.interleave")


@pytest.mark.parametrize(
    "mixin_filename,input_cls",
    [
        ("source_parquet.yaml", PrepareParquetInputSpec),
        ("source_hf.yaml", PrepareHfInputSpec),
        ("source_wds.yaml", PrepareWdsInputSpec),
        ("source_audio_dir.yaml", PrepareAudioDirInputSpec),
        ("source_lhotse_recipe.yaml", PrepareLhotseRecipeInputSpec),
    ],
)
def test_source_mixin_mirrors_input_defaults(mixin_filename, input_cls):
    _assert_subset(
        _load(mixin_filename)["convert"]["input"],
        _schema_defaults(input_cls),
        f"{mixin_filename} convert.input",
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
