"""Config-driven prepare stage adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_tokenization.config.schema import DatasetSpec
from audio_tokenization.utils.prepare_data import prepare_parquet_to_shar
from audio_tokenization.utils.prepare_data.constants import PREPARE_STATE_FILE


def _validate_existing_prepare_state(shar_dir: Path) -> None:
    state_path = shar_dir / PREPARE_STATE_FILE
    if not state_path.is_file():
        return

    payload = json.loads(state_path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid prepare state format: {state_path}")

    version = payload.get("version")
    if version != 1:
        raise RuntimeError(
            f"Unsupported legacy prepare state at {state_path}. "
            "Remove the output directory or migrate the state file before using the "
            "config-driven prepare entrypoint."
        )


def build_prepare_namespace(spec: DatasetSpec) -> argparse.Namespace:
    prepare = spec.prepare
    if prepare.family != "parquet":
        raise NotImplementedError(
            f"Config-driven prepare currently supports family='parquet' only; got {prepare.family!r}"
        )

    return argparse.Namespace(
        parquet_dir=Path(prepare.input.parquet_dir),
        parquet_glob=prepare.input.parquet_glob,
        shar_dir=Path(prepare.output.shar_dir),
        shard_size=prepare.output.shard_size,
        shar_format=prepare.output.shar_format,
        target_sr=prepare.output.target_sr,
        text_tokenizer=prepare.output.text_tokenizer,
        num_workers=prepare.output.num_workers,
        resampling_backend=prepare.output.resampling_backend,
        mp_start_method=prepare.output.mp_start_method,
        read_batch_size=prepare.output.read_batch_size,
        id_column=[prepare.metadata.id_column] if prepare.metadata.id_column else None,
        audio_column=prepare.metadata.audio_column,
        text_column=prepare.metadata.text_column,
        duration_column=prepare.metadata.duration_column,
        language_column=prepare.metadata.language_column,
        language=prepare.metadata.language,
        custom_columns=list(prepare.metadata.custom_columns) or None,
        text_tokenize_custom_columns=list(prepare.metadata.text_tokenize_custom_columns) or None,
        external_metadata=prepare.metadata.external_metadata,
        custom_fields=list(prepare.metadata.custom_fields) or None,
        id_field=prepare.metadata.id_field,
        text_field=prepare.metadata.text_field,
        input_clip_id_parser=prepare.metadata.input_clip_id_parser,
    )


def run_prepare(spec: DatasetSpec):
    if not spec.prepare.enabled:
        return {"skipped": True, "reason": "prepare.disabled"}

    args = build_prepare_namespace(spec)
    _validate_existing_prepare_state(args.shar_dir)

    if spec.prepare.family == "parquet":
        return prepare_parquet_to_shar.run(args)

    raise NotImplementedError(
        f"Unsupported config-driven prepare family {spec.prepare.family!r}"
    )
