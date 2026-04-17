"""Canonical dataset-spec schema for config-driven audio pipeline entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from audio_tokenization.prepare.metadata import normalize_optional_path


SUPPORTED_PREPARE_FAMILIES = {
    "audio_dir",
    "hf",
    "lhotse_recipe",
    "parquet",
    "wds",
}


def _as_plain_mapping(cfg: DictConfig | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(cfg, DictConfig):
        plain = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(plain, dict):
            raise TypeError("Dataset config must resolve to a mapping")
        return plain
    return cfg


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Dataset spec requires mapping field {key!r}")
    return value


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"Dataset spec requires non-empty field {key!r}")
    if not isinstance(value, str):
        raise TypeError(f"Dataset spec field {key!r} must be a string")
    return value


def _optional_str(payload: Mapping[str, Any], key: str, default: str | None = None) -> str | None:
    value = payload.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Dataset spec field {key!r} must be a string or null")
    return value


def _optional_str_with_unset_default(
    payload: Mapping[str, Any], key: str, unset_default: str | None
) -> str | None:
    """Distinguish "omitted" (→ unset_default) from "explicit null" (→ None).

    Behaviour:
      - key not in payload          → unset_default
      - payload[key] is None        → None  (caller explicitly disabled the feature)
      - payload[key] is str         → str
      - else                        → TypeError

    Use this for CLI-default parity where omitting the field must yield the
    legacy argparse default (e.g. ``text_column`` → ``"text"``) while still
    supporting ``text_column: null`` to disable transcripts for truly
    unsupervised datasets.
    """
    if key not in payload:
        return unset_default
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Dataset spec field {key!r} must be a string or null")
    return value


def _coerce_int(value: Any, *, field_name: str, allow_none: bool = False) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Dataset spec field {field_name!r} is required")
    if isinstance(value, bool):
        raise TypeError(f"Dataset spec field {field_name!r} must be an int, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as e:
            raise TypeError(
                f"Dataset spec field {field_name!r} must be an int-compatible string"
            ) from e
    raise TypeError(f"Dataset spec field {field_name!r} must be an int")


def _coerce_list_of_str(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"Dataset spec field {field_name!r} must be a list of strings")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(
                f"Dataset spec field {field_name!r}[{idx}] must be a string"
            )
        out.append(item)
    return out


def _coerce_id_column(
    payload: Mapping[str, Any], key: str = "id_column"
) -> str | list[str] | None:
    """Accept id_column as str, list[str], or null.

    Returns the original str / list / None. The legacy CLI accepts
    ``--id-column`` with ``nargs="*"`` and ``extract_row_metadata`` joins
    multi-column values with ``_``; this helper preserves both shapes so
    composite-ID YAML (``id_column: [session, seg]``) does not regress.
    """
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return None
        out: list[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, str):
                raise TypeError(
                    f"Dataset spec field {key!r}[{idx}] must be a string"
                )
            out.append(item)
        return out
    raise TypeError(
        f"Dataset spec field {key!r} must be a string, a list of strings, or null"
    )


@dataclass(frozen=True)
class PrepareParquetInputSpec:
    parquet_dir: str
    parquet_glob: str = "*.parquet"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PrepareParquetInputSpec":
        return cls(
            parquet_dir=_require_str(payload, "parquet_dir"),
            parquet_glob=_optional_str(payload, "parquet_glob", "*.parquet") or "*.parquet",
        )


@dataclass(frozen=True)
class PrepareOutputSpec:
    shar_dir: str
    shard_size: int = 5000
    shar_format: str = "flac"
    target_sr: int = 24000
    text_tokenizer: str | None = None
    num_workers: int = 20
    resampling_backend: str | None = None
    mp_start_method: str = "forkserver"
    read_batch_size: int = 256

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PrepareOutputSpec":
        return cls(
            shar_dir=_require_str(payload, "shar_dir"),
            shard_size=_coerce_int(payload.get("shard_size", 5000), field_name="prepare.output.shard_size") or 5000,
            shar_format=_optional_str(payload, "shar_format", "flac") or "flac",
            target_sr=_coerce_int(payload.get("target_sr", 24000), field_name="prepare.output.target_sr") or 24000,
            text_tokenizer=_optional_str(payload, "text_tokenizer"),
            num_workers=_coerce_int(payload.get("num_workers", 20), field_name="prepare.output.num_workers") or 20,
            resampling_backend=_optional_str(payload, "resampling_backend"),
            mp_start_method=_optional_str(payload, "mp_start_method", "forkserver") or "forkserver",
            read_batch_size=_coerce_int(payload.get("read_batch_size", 256), field_name="prepare.output.read_batch_size") or 256,
        )


@dataclass(frozen=True)
class PrepareMetadataSpec:
    """Per-row metadata extraction knobs.

    Defaults mirror the legacy ``prepare_parquet_to_shar.py`` argparse defaults
    so YAML and CLI cannot drift.

    ``id_column`` may be a single column name or a list of column names; when
    a list is supplied, ``extract_row_metadata`` joins the values with ``"_"``
    to form the composite row ID.

    ``text_column`` defaults to ``"text"``; pass ``text_column: null`` to
    disable transcript extraction for unsupervised datasets.
    """

    audio_column: str = "audio"
    text_column: str | None = "text"
    duration_column: str | None = "duration"
    id_column: str | list[str] | None = None
    language_column: str | None = None
    language: str | None = None
    custom_columns: list[str] = field(default_factory=list)
    text_tokenize_custom_columns: list[str] = field(default_factory=list)
    input_clip_id_parser: str | None = None
    external_metadata: str | None = None
    custom_fields: list[str] = field(default_factory=list)
    id_field: str = "id"
    text_field: str = "text"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PrepareMetadataSpec":
        payload = payload or {}
        return cls(
            audio_column=_optional_str(payload, "audio_column", "audio") or "audio",
            text_column=_optional_str_with_unset_default(payload, "text_column", "text"),
            duration_column=_optional_str_with_unset_default(
                payload, "duration_column", "duration"
            ),
            id_column=_coerce_id_column(payload),
            language_column=_optional_str(payload, "language_column"),
            language=_optional_str(payload, "language"),
            custom_columns=_coerce_list_of_str(payload.get("custom_columns"), field_name="prepare.metadata.custom_columns"),
            text_tokenize_custom_columns=_coerce_list_of_str(
                payload.get("text_tokenize_custom_columns"),
                field_name="prepare.metadata.text_tokenize_custom_columns",
            ),
            input_clip_id_parser=_optional_str(payload, "input_clip_id_parser"),
            external_metadata=_optional_str(payload, "external_metadata"),
            custom_fields=_coerce_list_of_str(payload.get("custom_fields"), field_name="prepare.metadata.custom_fields"),
            id_field=_optional_str(payload, "id_field", "id") or "id",
            text_field=_optional_str(payload, "text_field", "text") or "text",
        )


@dataclass(frozen=True)
class PrepareSpec:
    enabled: bool
    family: str
    input: PrepareParquetInputSpec
    output: PrepareOutputSpec
    metadata: PrepareMetadataSpec

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PrepareSpec":
        family = _require_str(payload, "family")
        if family not in SUPPORTED_PREPARE_FAMILIES:
            raise ValueError(
                f"Unsupported prepare family {family!r}; supported: {sorted(SUPPORTED_PREPARE_FAMILIES)}"
            )
        if family != "parquet":
            raise NotImplementedError(
                f"Config-driven prepare currently supports family='parquet' only; got {family!r}"
            )
        return cls(
            enabled=bool(payload.get("enabled", True)),
            family=family,
            input=PrepareParquetInputSpec.from_mapping(_require_mapping(payload, "input")),
            output=PrepareOutputSpec.from_mapping(_require_mapping(payload, "output")),
            metadata=PrepareMetadataSpec.from_mapping(payload.get("metadata")),
        )

    def fingerprint_payload(self) -> dict[str, Any]:
        """Canonical fingerprint dict for resume-invariant checks."""
        if self.family != "parquet":
            raise NotImplementedError(
                f"fingerprint_payload not yet defined for family={self.family!r}"
            )
        metadata = self.metadata
        output = self.output
        return {
            "parquet_dir": normalize_optional_path(self.input.parquet_dir),
            "parquet_glob": self.input.parquet_glob,
            "shar_format": output.shar_format,
            "target_sr": output.target_sr,
            "text_tokenizer": normalize_optional_path(output.text_tokenizer),
            "resampling_backend": output.resampling_backend,
            "audio_column": metadata.audio_column,
            "text_column": metadata.text_column,
            "duration_column": metadata.duration_column,
            "id_column": _normalize_id_column_arg(metadata.id_column),
            "language_column": metadata.language_column,
            "language": metadata.language,
            "custom_columns": _normalize_list_arg(metadata.custom_columns),
            "text_tokenize_custom_columns": _normalize_list_arg(metadata.text_tokenize_custom_columns),
            "input_clip_id_parser": metadata.input_clip_id_parser,
            "external_metadata": normalize_optional_path(metadata.external_metadata),
            "custom_fields": _normalize_list_arg(metadata.custom_fields),
            "id_field": metadata.id_field,
            "text_field": metadata.text_field,
        }


@dataclass(frozen=True)
class DirectProductSpec:
    enabled: bool = False
    output_dir: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DirectProductSpec":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            output_dir=_optional_str(payload, "output_dir"),
        )


@dataclass(frozen=True)
class InterleaveProductSpec:
    enabled: bool = False
    cache_dir: str | None = None
    output_dir: str | None = None
    strategy: str = "shift_by_one"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "InterleaveProductSpec":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            cache_dir=_optional_str(payload, "cache_dir"),
            output_dir=_optional_str(payload, "output_dir"),
            strategy=_optional_str(payload, "strategy", "shift_by_one") or "shift_by_one",
        )


@dataclass(frozen=True)
class ProductMatrixSpec:
    asr_direct: DirectProductSpec = field(default_factory=DirectProductSpec)
    interleave: InterleaveProductSpec = field(default_factory=InterleaveProductSpec)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ProductMatrixSpec":
        payload = payload or {}
        return cls(
            asr_direct=DirectProductSpec.from_mapping(payload.get("asr_direct")),
            interleave=InterleaveProductSpec.from_mapping(payload.get("interleave")),
        )


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    prepare: PrepareSpec
    products: ProductMatrixSpec = field(default_factory=ProductMatrixSpec)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetSpec":
        return cls(
            name=_require_str(payload, "name"),
            prepare=PrepareSpec.from_mapping(_require_mapping(payload, "prepare")),
            products=ProductMatrixSpec.from_mapping(payload.get("products")),
        )


def load_dataset_spec(cfg: DictConfig | Mapping[str, Any]) -> DatasetSpec:
    """Load and validate a canonical dataset spec from Hydra/OmegaConf config."""
    plain = _as_plain_mapping(cfg)
    return DatasetSpec.from_mapping(plain)


def _normalize_list_arg(value: Any) -> list[str] | None:
    """Argparse list/None → sorted list / None (empty collapses to None)."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        items = sorted(value)
        return items or None
    raise TypeError(f"Unsupported list-shaped argparse value: {type(value).__name__}")


def _normalize_id_column_arg(value: Any) -> str | list[str] | None:
    """id_column is shape-preserving: single str stays str, list preserves order.

    Why: ``extract_row_metadata`` joins list values with ``'_'`` to build the
    composite ID — sorting would corrupt the ID.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) == 1:
            return value[0]
        return list(value)
    raise TypeError(f"Unsupported id_column shape: {type(value).__name__}")


def build_parquet_prepare_fingerprint(args: Any) -> dict[str, Any]:
    """Canonical fingerprint dict from an argparse Namespace."""
    return {
        "parquet_dir": normalize_optional_path(args.parquet_dir),
        "parquet_glob": args.parquet_glob,
        "shar_format": args.shar_format,
        "target_sr": args.target_sr,
        "text_tokenizer": normalize_optional_path(args.text_tokenizer),
        "resampling_backend": args.resampling_backend,
        "audio_column": args.audio_column,
        "text_column": args.text_column,
        "duration_column": args.duration_column,
        "id_column": _normalize_id_column_arg(args.id_column),
        "language_column": args.language_column,
        "language": args.language,
        "custom_columns": _normalize_list_arg(args.custom_columns),
        "text_tokenize_custom_columns": _normalize_list_arg(args.text_tokenize_custom_columns),
        "input_clip_id_parser": args.input_clip_id_parser,
        "external_metadata": normalize_optional_path(args.external_metadata),
        "custom_fields": _normalize_list_arg(args.custom_fields),
        "id_field": args.id_field,
        "text_field": args.text_field,
    }


