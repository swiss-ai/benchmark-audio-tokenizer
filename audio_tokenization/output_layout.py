"""Output path layout contracts shared by stage adapters and observability."""

from __future__ import annotations

from pathlib import Path

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.contracts.artifacts import INTERLEAVE_CACHE_OUTPUT_STEM


def resolve_tokenize_output_name(spec: TokenizeSpec, *, dataset_name: str) -> str:
    """Return the stable dataset name used inside tokenize output paths."""
    output_name = spec.output.output_name or dataset_name
    if not output_name:
        raise ValueError("tokenize.output.output_name or dataset name is required.")
    return output_name


def build_tokenize_output_subdir(spec: TokenizeSpec, *, dataset_name: str) -> Path:
    """Build the mode-specific subdirectory below ``tokenize.output.output_dir``."""
    output_name = resolve_tokenize_output_name(spec, dataset_name=dataset_name)

    if spec.mode == "audio_text":
        if spec.audio_text_format == "interleaved":
            return Path(INTERLEAVE_CACHE_OUTPUT_STEM) / output_name
        return Path(spec.audio_text_task) / output_name

    return Path("audio_only") / output_name


def resolve_tokenize_output_dir(spec: TokenizeSpec, *, dataset_name: str) -> Path:
    """Return the final tokenize output directory for a resolved dataset."""
    return Path(spec.output.output_dir) / build_tokenize_output_subdir(
        spec,
        dataset_name=dataset_name,
    )
