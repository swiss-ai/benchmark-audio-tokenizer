"""Canonical dataset-spec schema for the config-driven audio pipeline.

Side-effect free: loading a spec must not import heavy prepare modules or
touch global state. Schema is authoritative for config semantics; standalone
prepare-script argparse defaults are an isolated property of those CLIs and
are not consulted here.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping

from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    model_validator,
)

from audio_tokenization.config.authoring import resolve_authoring_config
from audio_tokenization.contracts.artifacts import SHAR_INDEX_FILENAME
from audio_tokenization.prepare.metadata import normalize_optional_path


PrepareFamily = Literal["parquet", "hf", "wds", "audio_dir", "lhotse_recipe"]
TokenizeMode = Literal["audio_only", "audio_text"]
AudioTextFormat = Literal["direct", "interleaved"]
AudioTextTask = Literal["transcribe", "translate", "annotate"]


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


def _coerce_nonempty_str(value: Any) -> str:
    if value is None or value == "":
        raise ValueError("must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError("must be a string")
    return value


def _coerce_int_like(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("must be an int, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as e:
            raise TypeError("must be an int-compatible string") from e
    raise TypeError("must be an int")


def _coerce_float_like(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("must be a float, not bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as e:
            raise TypeError("must be a float-compatible string") from e
    raise TypeError("must be a float")


def _coerce_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise TypeError("must be a list of strings")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"[{idx}] must be a string")
        out.append(item)
    return out


def _coerce_optional_list_of_str(value: Any) -> list[str] | None:
    if value is None:
        return None
    out = _coerce_list_of_str(value)
    return out or None


def _coerce_id_column_value(value: Any) -> str | list[str] | None:
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
                raise TypeError(f"id_column[{idx}] must be a string")
            out.append(item)
        return out
    raise TypeError("id_column must be a string, a list of strings, or null")


NonEmptyStr = Annotated[str, BeforeValidator(_coerce_nonempty_str)]
IntLike = Annotated[int, BeforeValidator(_coerce_int_like)]
OptIntLike = Annotated[int | None, BeforeValidator(_coerce_int_like)]
FloatLike = Annotated[float, BeforeValidator(_coerce_float_like)]
OptFloatLike = Annotated[float | None, BeforeValidator(_coerce_float_like)]
StrList = Annotated[list[str], BeforeValidator(_coerce_list_of_str)]
OptStrList = Annotated[list[str] | None, BeforeValidator(_coerce_optional_list_of_str)]
IdColumn = Annotated[str | list[str] | None, BeforeValidator(_coerce_id_column_value)]


class SchemaModel(BaseModel):
    # extra='forbid': canonical configs must reject typo'd keys (e.g.
    # ``parquet-dir`` vs ``parquet_dir``) loudly at load time. Silent
    # default-substitution is the worst failure mode for a config layer.
    model_config = ConfigDict(extra="forbid", frozen=True)


class PrepareParquetInputSpec(SchemaModel):
    parquet_dir: NonEmptyStr
    parquet_glob: NonEmptyStr = "*.parquet"

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "parquet_dir": normalize_optional_path(self.parquet_dir),
            "parquet_glob": self.parquet_glob,
        }


class PrepareHfInputSpec(SchemaModel):
    """HuggingFace arrow-shard input. Either ``arrow_files`` or ``arrow_dir``."""

    arrow_dir: str | None = None
    arrow_glob: NonEmptyStr = "*.arrow"
    arrow_files: OptStrList = None

    @model_validator(mode="after")
    def _require_arrow_source(self):
        if not self.arrow_dir and not self.arrow_files:
            raise ValueError("convert.input requires arrow_dir or arrow_files")
        return self

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "arrow_dir": normalize_optional_path(self.arrow_dir),
            "arrow_glob": self.arrow_glob,
            "arrow_files": _normalize_list_arg(self.arrow_files),
        }


class PrepareWdsInputSpec(SchemaModel):
    """Tar-archive input for WebDataset / external-metadata mode."""

    wds_shards: StrList
    min_sr: OptIntLike = None
    no_mono_downmix: StrictBool = False
    vad_segmentation: StrictBool = False
    vad_per_shard_dir: str | None = None
    vad_max_chunk_sec: FloatLike = 200.0
    vad_min_chunk_sec: FloatLike = 10.0
    vad_sample_rate: IntLike = 16000
    vad_max_merge_gap_sec: FloatLike = 0.5
    vad_max_duration_sec: OptFloatLike = None
    vad_min_rms_db: OptFloatLike = None

    @model_validator(mode="after")
    def _require_wds_shards(self):
        if not self.wds_shards:
            raise ValueError("convert.input.wds_shards must be a non-empty list")
        return self

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "wds_shards": _normalize_list_arg(self.wds_shards),
            "min_sr": self.min_sr,
            "no_mono_downmix": self.no_mono_downmix,
            "vad_segmentation": self.vad_segmentation,
            "vad_per_shard_dir": normalize_optional_path(self.vad_per_shard_dir),
            "vad_max_chunk_sec": self.vad_max_chunk_sec,
            "vad_min_chunk_sec": self.vad_min_chunk_sec,
            "vad_sample_rate": self.vad_sample_rate,
            "vad_max_merge_gap_sec": self.vad_max_merge_gap_sec,
            "vad_max_duration_sec": self.vad_max_duration_sec,
            "vad_min_rms_db": self.vad_min_rms_db,
        }


class PrepareAudioDirInputSpec(SchemaModel):
    """Audio-files-on-disk + per-language VAD JSONL input."""

    audio_root: NonEmptyStr
    jsonl_files: StrList
    audio_ext: NonEmptyStr = ".ogg"
    min_sr: OptIntLike = None
    no_mono_downmix: StrictBool = False
    vad_max_chunk_sec: FloatLike = 200.0
    vad_min_chunk_sec: FloatLike = 5.0
    vad_sample_rate: IntLike = 16000
    vad_max_merge_gap_sec: FloatLike = 1.0
    vad_max_duration_sec: OptFloatLike = None
    vad_min_rms_db: OptFloatLike = None

    @model_validator(mode="after")
    def _require_jsonls(self):
        if not self.jsonl_files:
            raise ValueError("convert.input.jsonl_files must be a non-empty list")
        return self

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "audio_root": normalize_optional_path(self.audio_root),
            "jsonl_files": _normalize_list_arg(self.jsonl_files),
            "audio_ext": self.audio_ext,
            "min_sr": self.min_sr,
            "no_mono_downmix": self.no_mono_downmix,
            "vad_max_chunk_sec": self.vad_max_chunk_sec,
            "vad_min_chunk_sec": self.vad_min_chunk_sec,
            "vad_sample_rate": self.vad_sample_rate,
            "vad_max_merge_gap_sec": self.vad_max_merge_gap_sec,
            "vad_max_duration_sec": self.vad_max_duration_sec,
            "vad_min_rms_db": self.vad_min_rms_db,
        }


class PrepareLhotseRecipeInputSpec(SchemaModel):
    """Lhotse built-in recipe input (commonvoice, librispeech, voxpopuli, ...)."""

    recipe: NonEmptyStr
    corpus_dir: NonEmptyStr
    split: NonEmptyStr
    recipe_kwargs: NonEmptyStr = "{}"
    min_sample_rate: OptIntLike = None
    trim_to_supervisions: StrictBool = False
    shar_index_filename: NonEmptyStr = SHAR_INDEX_FILENAME

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "corpus_dir": normalize_optional_path(self.corpus_dir),
            "split": self.split,
            "recipe_kwargs": self.recipe_kwargs,
            "min_sample_rate": self.min_sample_rate,
            "trim_to_supervisions": self.trim_to_supervisions,
            "shar_index_filename": self.shar_index_filename,
        }


_INPUT_SPEC_BY_FAMILY: dict[str, type[SchemaModel]] = {
    "parquet": PrepareParquetInputSpec,
    "hf": PrepareHfInputSpec,
    "wds": PrepareWdsInputSpec,
    "audio_dir": PrepareAudioDirInputSpec,
    "lhotse_recipe": PrepareLhotseRecipeInputSpec,
}


class PrepareOutputSpec(SchemaModel):
    """Cross-cutting convert-output knobs."""

    shar_dir: NonEmptyStr
    shard_size: IntLike
    shar_format: NonEmptyStr = "flac"
    target_sr: OptIntLike = 24000
    text_tokenizer: str | None = None
    num_workers: OptIntLike = None
    resampling_backend: str | None = "soxr"
    mp_start_method: NonEmptyStr = "forkserver"
    read_batch_size: IntLike = 256


class PrepareMetadataSpec(SchemaModel):
    """Per-row metadata extraction knobs."""

    audio_column: NonEmptyStr = "audio"
    text_column: str | None = None
    duration_column: str | None = None
    source_id_column: str | None = None
    clip_num_column: str | None = None
    clip_start_column: str | None = None
    clip_end_column: str | None = None
    clip_duration_column: str | None = None
    id_column: IdColumn = None
    id_prefix: str | None = None
    language_column: str | None = None
    language: str | None = None
    custom_columns: StrList = Field(default_factory=list)
    constant_custom: dict[str, Any] = Field(default_factory=dict)
    derived_custom: dict[str, str] = Field(default_factory=dict)
    text_tokenize_custom_columns: StrList = Field(default_factory=list)
    input_clip_id_parser: str | None = None
    external_metadata: str | None = None
    custom_fields: StrList = Field(default_factory=list)
    id_field: NonEmptyStr = "id"
    text_field: NonEmptyStr = "text"

    @model_validator(mode="after")
    def _validate_clip_timestamps(self):
        if self.clip_num_column and not self.source_id_column:
            raise ValueError(
                "clip_num_column requires source_id_column"
            )
        if self.input_clip_id_parser and (
            self.source_id_column or self.clip_num_column
        ):
            raise ValueError(
                "set either source_id_column / clip_num_column or input_clip_id_parser, not both"
            )
        if (self.clip_end_column or self.clip_duration_column) and not self.clip_start_column:
            raise ValueError(
                "clip_end_column / clip_duration_column require clip_start_column"
            )
        if self.clip_end_column and self.clip_duration_column:
            raise ValueError(
                "set exactly one of clip_end_column or clip_duration_column, not both"
            )
        if (
            self.source_id_column
            and not self.clip_num_column
            and not self.clip_start_column
        ):
            raise ValueError(
                "source_id_column without clip_num_column requires clip_start_column "
                "so prepare can derive a stable timestamp tie-breaker"
            )
        return self


class PrepareSpec(SchemaModel):
    enabled: StrictBool = True
    family: PrepareFamily
    input: Any
    output: PrepareOutputSpec
    metadata: PrepareMetadataSpec

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PrepareSpec":
        family = payload.get("family")
        if family is None or family == "":
            raise ValueError("Dataset spec requires non-empty field 'family'")
        if not isinstance(family, str):
            raise TypeError("Dataset spec field 'family' must be a string")
        input_cls = _INPUT_SPEC_BY_FAMILY.get(family)
        if input_cls is None:
            raise ValueError(
                f"Unsupported convert family {family!r}; supported: {sorted(_INPUT_SPEC_BY_FAMILY)}"
            )
        data = dict(payload)
        data["input"] = input_cls.model_validate(_require_mapping(payload, "input"))
        data["output"] = PrepareOutputSpec.model_validate(_require_mapping(payload, "output"))
        data["metadata"] = PrepareMetadataSpec.model_validate(payload.get("metadata") or {})
        return cls.model_validate(data)

    def fingerprint_payload(self) -> dict[str, Any]:
        """Canonical fingerprint dict for resume-invariant checks."""
        return {
            "family": self.family,
            **{
                f"input.{k}": v
                for k, v in self.input.fingerprint_payload().items()
            },
            **_prepare_output_fingerprint(self.output, family=self.family),
            **_prepare_metadata_fingerprint(self.metadata, family=self.family),
        }


class TokenizerSpec(SchemaModel):
    """Audio/text tokenizer model config (path + sampling + compile knobs)."""

    path: NonEmptyStr
    sampling_rate: OptIntLike = 24000
    torch_compile: StrictBool = False
    trim_last_tokens: IntLike = 5


class TokenizeFilterSpec(SchemaModel):
    """Per-cut filtering + audio-level quality knobs applied inside the Lhotse pipeline."""

    min_duration: FloatLike = 1.0
    max_duration: FloatLike = 200.0
    min_sample_rate: IntLike = 16000
    min_rms_db: IntLike = -50
    normalize_peak_db: IntLike = -3


class TokenizeDataloaderSpec(SchemaModel):
    """Sampler + DataLoader knobs."""

    num_workers: IntLike = 32
    prefetch_factor: IntLike = 4
    max_batch_duration: FloatLike = 2000.0
    checkpoint_interval_batches: IntLike = 1000
    max_batch_cuts: OptIntLike = None
    num_buckets: IntLike = 20
    bucket_buffer_size: IntLike = 20000
    sampler_shuffle: StrictBool = True
    sampler_seed: IntLike = 42
    quadratic_duration: OptFloatLike = None


class TokenizeOutputSpec(SchemaModel):
    """Where tokenize writes."""

    output_dir: NonEmptyStr
    output_name: str | None = None
    shar_index_filename: NonEmptyStr = SHAR_INDEX_FILENAME


def _normalize_interleave_partitioning(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {"type": "hash", "field": "source_id", "num_buckets": 16}

    ptype = str(value.get("type", "hash"))
    if ptype == "hash":
        field = str(value.get("field", "source_id"))
        num_buckets = _coerce_int_like(value.get("num_buckets", 16))
        if num_buckets is None or num_buckets <= 0:
            raise ValueError("tokenize.partitioning.num_buckets must be > 0")
        return {"type": "hash", "field": field, "num_buckets": num_buckets}

    if ptype == "field":
        field = value.get("field")
        if not isinstance(field, str) or not field:
            raise ValueError("tokenize.partitioning.field is required for field partitioning")
        return {"type": "field", "field": field}

    raise ValueError(f"Unsupported tokenize.partitioning.type: {ptype!r}")


def _normalize_wandb_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {"enabled": False}
    enabled = bool(value.get("enabled", False))
    tags = value.get("tags", [])
    if tags is None:
        tags = []
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise TypeError("tokenize.wandb.tags must be a list of strings")
    log_interval = _coerce_float_like(value.get("log_interval_seconds", 10.0))
    if log_interval is None or log_interval <= 0:
        raise ValueError("tokenize.wandb.log_interval_seconds must be > 0")
    out = {
        "enabled": enabled,
        "project": str(value.get("project", "audio-tokenization")),
        "entity": value.get("entity"),
        "name": value.get("name"),
        "tags": list(tags),
        "log_interval_seconds": float(log_interval),
    }
    for key in ("entity", "name"):
        if out[key] is not None and not isinstance(out[key], str):
            raise TypeError(f"tokenize.wandb.{key} must be a string or null")
    return out


class TokenizeSpec(SchemaModel):
    """Canonical typed config for the tokenize stage."""

    tokenizer: TokenizerSpec
    output: TokenizeOutputSpec
    mode: TokenizeMode = "audio_only"
    audio_text_format: AudioTextFormat = "direct"
    audio_text_task: AudioTextTask = "transcribe"
    input_shar_dir: OptStrList = None
    partitioning: dict[str, Any] | None = None
    filter: TokenizeFilterSpec = Field(default_factory=TokenizeFilterSpec)
    dataloader: TokenizeDataloaderSpec = Field(default_factory=TokenizeDataloaderSpec)
    wandb: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_wandb(cls, value: Any):
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if data.get("wandb") is None:
            data["wandb"] = {"enabled": False}
        elif "wandb" in data and not isinstance(data["wandb"], Mapping):
            raise TypeError("tokenize.wandb must be a mapping or null")
        return data

    @model_validator(mode="after")
    def _normalize_runtime_dicts(self):
        update: dict[str, Any] = {"wandb": _normalize_wandb_config(self.wandb)}
        is_interleaved = self.mode == "audio_text" and self.audio_text_format == "interleaved"
        if is_interleaved or self.partitioning is not None:
            update["partitioning"] = _normalize_interleave_partitioning(self.partitioning)
        return self.model_copy(update=update)

    def fingerprint_payload(self) -> dict[str, Any]:
        """Output-shaping subset of the spec, for resume-safety checks."""
        f = self.filter
        d = self.dataloader
        return {
            "tokenizer_path": normalize_optional_path(self.tokenizer.path),
            "tokenizer_sampling_rate": self.tokenizer.sampling_rate,
            "trim_last_tokens": self.tokenizer.trim_last_tokens,
            "mode": self.mode,
            "audio_text_format": self.audio_text_format,
            "audio_text_task": self.audio_text_task,
            "partitioning": (
                self.partitioning
                if self.mode == "audio_text" and self.audio_text_format == "interleaved"
                else None
            ),
            "input_shar_dir": sorted(self.input_shar_dir) if self.input_shar_dir else None,
            "output_name": self.output.output_name,
            "shar_index_filename": self.output.shar_index_filename,
            "filter_min_duration": f.min_duration,
            "filter_max_duration": f.max_duration,
            "filter_min_sample_rate": f.min_sample_rate,
            "filter_min_rms_db": f.min_rms_db,
            "filter_normalize_peak_db": f.normalize_peak_db,
            "num_buckets": d.num_buckets,
            "bucket_buffer_size": d.bucket_buffer_size,
            "sampler_seed": d.sampler_seed,
            "sampler_shuffle": d.sampler_shuffle,
            "quadratic_duration": d.quadratic_duration,
        }


class InterleaveProductSpec(SchemaModel):
    """Post-tokenize interleave product."""

    enabled: StrictBool = False
    cache_dir: str | None = None
    output_dir: str | None = None
    strategy: NonEmptyStr = "shift_by_one"
    tokenizer_path: str | None = None
    max_seq_len: IntLike = 262144
    max_gap_sec: OptFloatLike = None
    seq_threshold: OptIntLike = None
    transcribe_ratio: OptFloatLike = None
    num_workers: OptIntLike = None
    tmp_dir: str | None = None

    def fingerprint_payload(self) -> dict[str, Any]:
        """Output-shaping subset: excludes num_workers / tmp_dir (operational)."""
        return {
            "enabled": self.enabled,
            "cache_dir": normalize_optional_path(self.cache_dir),
            "output_dir": normalize_optional_path(self.output_dir),
            "strategy": self.strategy,
            "tokenizer_path": normalize_optional_path(self.tokenizer_path),
            "max_seq_len": self.max_seq_len,
            "max_gap_sec": self.max_gap_sec,
            "seq_threshold": self.seq_threshold,
            "transcribe_ratio": self.transcribe_ratio,
        }


class SftProductSpec(SchemaModel):
    """Post-tokenize SFT sequence product.

    SFT assembly is a materialization product because it combines model-specific
    text template rendering with already-tokenized audio components. The actual
    assembler is intentionally wired separately from the schema so partial
    configs can be validated before the implementation lands.
    """

    enabled: StrictBool = False
    doc_manifest: str | None = None
    audio_token_cache_dir: str | None = None
    audio_token_cache_manifest: str | None = None
    output_dir: str | None = None
    tokenizer_path: str | None = None
    chat_template: str | None = None
    max_seq_len: IntLike = 262144
    num_workers: OptIntLike = None
    tmp_dir: str | None = None

    def fingerprint_payload(self) -> dict[str, Any]:
        """Output-shaping subset: excludes num_workers / tmp_dir."""
        return {
            "enabled": self.enabled,
            "doc_manifest": normalize_optional_path(self.doc_manifest),
            "audio_token_cache_dir": normalize_optional_path(self.audio_token_cache_dir),
            "audio_token_cache_manifest": normalize_optional_path(self.audio_token_cache_manifest),
            "output_dir": normalize_optional_path(self.output_dir),
            "tokenizer_path": normalize_optional_path(self.tokenizer_path),
            "chat_template": self.chat_template,
            "max_seq_len": self.max_seq_len,
        }


class ProductMatrixSpec(SchemaModel):
    """Post-tokenize materializations."""

    interleave: InterleaveProductSpec = Field(default_factory=InterleaveProductSpec)
    sft: SftProductSpec = Field(default_factory=SftProductSpec)


class DatasetSpec(SchemaModel):
    name: NonEmptyStr
    convert: PrepareSpec | None = None
    tokenize: TokenizeSpec | None = None
    materialize: ProductMatrixSpec = Field(default_factory=ProductMatrixSpec)

    @model_validator(mode="after")
    def _validate_cross_section_invariants(self):
        interleave = self.materialize.interleave
        if interleave.enabled and interleave.cache_dir is None:
            if self.tokenize is None:
                raise ValueError(
                    "materialize.interleave.enabled=true with no explicit cache_dir "
                    "requires a tokenize section so the cache path can be derived. "
                    "Either set materialize.interleave.cache_dir explicitly, or add a "
                    "tokenize section with mode='audio_text' + "
                    "audio_text_format='interleaved'."
                )
            if self.tokenize.mode != "audio_text" or self.tokenize.audio_text_format != "interleaved":
                raise ValueError(
                    "materialize.interleave.enabled=true with no explicit cache_dir "
                    "requires tokenize.mode='audio_text' and "
                    "audio_text_format='interleaved' to derive the cache path; got "
                    f"mode={self.tokenize.mode!r}, "
                    f"audio_text_format={self.tokenize.audio_text_format!r}. "
                    "Set materialize.interleave.cache_dir explicitly to consume a "
                    "cache built by a different pipeline."
                )
        return self

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetSpec":
        data = dict(payload)
        convert_payload = payload.get("convert")
        data["convert"] = (
            PrepareSpec.from_mapping(convert_payload)
            if convert_payload is not None
            else None
        )
        tokenize_payload = payload.get("tokenize")
        data["tokenize"] = (
            TokenizeSpec.model_validate(tokenize_payload)
            if tokenize_payload is not None
            else None
        )
        data["materialize"] = ProductMatrixSpec.model_validate(payload.get("materialize") or {})
        return cls.model_validate(data)


def load_dataset_spec(cfg: DictConfig | Mapping[str, Any]) -> DatasetSpec:
    """Load and validate a canonical dataset spec from Hydra/OmegaConf config."""
    plain = resolve_authoring_config(_as_plain_mapping(cfg))
    try:
        return DatasetSpec.from_mapping(plain)
    except ValidationError as e:
        raise ValueError(str(e)) from None


def _prepare_output_fingerprint(
    output: PrepareOutputSpec, *, family: PrepareFamily
) -> dict[str, Any]:
    payload = {
        "output.shard_size": output.shard_size,
        "output.shar_format": output.shar_format,
        "output.target_sr": output.target_sr,
    }
    if family in {"parquet", "hf", "wds", "lhotse_recipe"}:
        payload["output.text_tokenizer"] = normalize_optional_path(output.text_tokenizer)
    if family in {"parquet", "hf", "wds", "audio_dir"}:
        payload["output.resampling_backend"] = output.resampling_backend
    if family == "lhotse_recipe":
        payload["output.num_workers"] = output.num_workers
    return payload


def _prepare_metadata_fingerprint(
    metadata: PrepareMetadataSpec, *, family: PrepareFamily
) -> dict[str, Any]:
    if family == "parquet":
        return {
            "metadata.audio_column": metadata.audio_column,
            "metadata.text_column": metadata.text_column,
            "metadata.duration_column": metadata.duration_column,
            "metadata.source_id_column": metadata.source_id_column,
            "metadata.clip_num_column": metadata.clip_num_column,
            "metadata.clip_start_column": metadata.clip_start_column,
            "metadata.clip_end_column": metadata.clip_end_column,
            "metadata.clip_duration_column": metadata.clip_duration_column,
            "metadata.id_column": _normalize_id_column_arg(metadata.id_column),
            "metadata.id_prefix": metadata.id_prefix,
            "metadata.language_column": metadata.language_column,
            "metadata.language": metadata.language,
            "metadata.custom_columns": _normalize_list_arg(metadata.custom_columns),
            "metadata.constant_custom": dict(sorted(metadata.constant_custom.items())),
            "metadata.derived_custom": dict(sorted(metadata.derived_custom.items())),
            "metadata.text_tokenize_custom_columns": _normalize_list_arg(metadata.text_tokenize_custom_columns),
            "metadata.input_clip_id_parser": metadata.input_clip_id_parser,
            "metadata.external_metadata": normalize_optional_path(metadata.external_metadata),
            "metadata.custom_fields": _normalize_list_arg(metadata.custom_fields),
            "metadata.id_field": metadata.id_field,
            "metadata.text_field": metadata.text_field,
        }
    if family == "hf":
        return {
            "metadata.audio_column": metadata.audio_column,
            "metadata.text_column": metadata.text_column,
            "metadata.source_id_column": metadata.source_id_column,
            "metadata.clip_num_column": metadata.clip_num_column,
            "metadata.clip_start_column": metadata.clip_start_column,
            "metadata.clip_end_column": metadata.clip_end_column,
            "metadata.clip_duration_column": metadata.clip_duration_column,
            "metadata.id_column": _normalize_id_column_arg(metadata.id_column),
            "metadata.id_prefix": metadata.id_prefix,
            "metadata.language_column": metadata.language_column,
            "metadata.language": metadata.language,
            "metadata.custom_columns": _normalize_list_arg(metadata.custom_columns),
            "metadata.constant_custom": dict(sorted(metadata.constant_custom.items())),
            "metadata.derived_custom": dict(sorted(metadata.derived_custom.items())),
            "metadata.text_tokenize_custom_columns": _normalize_list_arg(metadata.text_tokenize_custom_columns),
            "metadata.input_clip_id_parser": metadata.input_clip_id_parser,
            "metadata.external_metadata": normalize_optional_path(metadata.external_metadata),
            "metadata.custom_fields": _normalize_list_arg(metadata.custom_fields),
            "metadata.id_field": metadata.id_field,
            "metadata.text_field": metadata.text_field,
        }
    if family == "wds":
        return {
            "metadata.language": metadata.language,
            "metadata.input_clip_id_parser": metadata.input_clip_id_parser,
            "metadata.external_metadata": normalize_optional_path(metadata.external_metadata),
            "metadata.custom_fields": _normalize_list_arg(metadata.custom_fields),
            "metadata.id_field": metadata.id_field,
            "metadata.text_field": metadata.text_field,
        }
    if family == "lhotse_recipe":
        return {
            "metadata.language": metadata.language,
        }
    if family == "audio_dir":
        return {}
    raise ValueError(f"Unsupported convert family {family!r}")


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
