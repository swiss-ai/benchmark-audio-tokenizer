"""Semantic validation for tokenized audio canaries.

This module intentionally lives outside the runtime pipeline. It samples
already-tokenized outputs, reconstructs audio with the production vokenizer,
transcribes the reconstruction with a local ASR model, and compares that ASR
text to the text tokens stored by the tokenization stage.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from audio_tokenization.interleave.common import (
    list_interleave_cache_partitions,
    load_interleave_cache,
)
from audio_tokenization.utils.indexed_dataset.constants import CUT_ID_SIDECAR_SUFFIX
from audio_tokenization.utils.indexed_dataset.cut_id_sidecar import (
    MegatronChunkReader,
    discover_cut_id_prefixes,
    read_cut_id_sidecar,
)
from audio_tokenization.utils.io import atomic_streaming_write, atomic_write_json, open_compressed
from audio_tokenization.utils.token_mapping import load_audio_token_mapping
from audio_tokenization.vokenizers.wavtokenizer.audio_only import WavTokenizerAudioOnly

DEFAULT_TOKENIZER_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok"
DEFAULT_ASR_MODEL = (
    "/capstor/store/cscs/swissai/infra01/MLLM/audio_asr/"
    "parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo"
)
DEFAULT_WHISPER_MODEL = "/capstor/store/cscs/swissai/infra01/MLLM/audio_asr/whisper-large-v3"


@dataclass(frozen=True)
class SemanticSample:
    sample_id: str
    mode: str
    audio_tokens: list[int]
    expected_text: str
    decoded_text_tokens: str
    metadata: dict[str, object]
    context_before: list[str]
    context_after: list[str]


def normalize_text(text: str) -> str:
    """Normalize text for coarse ASR-vs-reference comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def compare_texts(expected: str, observed: str) -> dict[str, float | int]:
    """Return lightweight, language-agnostic similarity metrics."""
    exp = normalize_text(expected)
    obs = normalize_text(observed)
    exp_words = exp.split()
    obs_words = obs.split()
    if not exp and not obs:
        return {"char_ratio": 1.0, "word_recall": 1.0, "expected_words": 0, "observed_words": 0}
    if not exp or not obs:
        return {
            "char_ratio": 0.0,
            "word_recall": 0.0,
            "expected_words": len(exp_words),
            "observed_words": len(obs_words),
        }
    obs_word_set = set(obs_words)
    covered = sum(1 for word in exp_words if word in obs_word_set)
    return {
        "char_ratio": SequenceMatcher(None, exp, obs).ratio(),
        "word_recall": covered / max(1, len(exp_words)),
        "expected_words": len(exp_words),
        "observed_words": len(obs_words),
    }


def load_reference_texts_from_shar(shar_dirs: Iterable[Path]) -> dict[str, str]:
    """Build a ``cut_id -> source text`` lookup from SHAR cut manifests."""
    references: dict[str, str] = {}
    for shar_dir in shar_dirs:
        root = Path(shar_dir).expanduser().resolve()
        for cut_path in sorted(root.rglob("cuts.*.jsonl*")):
            if not cut_path.is_file() or cut_path.name.endswith(".tmp"):
                continue
            with open_compressed(cut_path, "rt") as f:
                for line in f:
                    if not line.strip():
                        continue
                    cut = json.loads(line)
                    cut_id = cut.get("id")
                    if not cut_id:
                        continue
                    supervisions = cut.get("supervisions") or []
                    texts = [
                        str(sup.get("text") or "")
                        for sup in supervisions
                        if isinstance(sup, dict) and sup.get("text")
                    ]
                    if texts:
                        references[str(cut_id)] = " ".join(texts)
                        continue
                    custom = cut.get("custom") or {}
                    if isinstance(custom, dict) and custom.get("text"):
                        references[str(cut_id)] = str(custom["text"])
    return references


class TokenDecoder:
    """Decode audio/text token payloads using the production tokenizer."""

    def __init__(
        self,
        tokenizer_path: str,
        *,
        device: str,
    ) -> None:
        self.vokenizer = WavTokenizerAudioOnly(
            omni_tokenizer_path=tokenizer_path,
            device=device,
            torch_compile=False,
        )
        self.device = torch.device(device)
        mapping = load_audio_token_mapping(tokenizer_path)
        self.structure_tokens = mapping["structure_tokens"]
        self.audio_token_offset = int(mapping["audio_token_offset"])
        self.special_ids = set(int(v) for v in self.structure_tokens.values())
        self.special_ids.add(int(self.vokenizer.bos_id))
        self.special_ids.add(int(self.vokenizer.eos_id))

    @property
    def sample_rate(self) -> int:
        return 24000

    def decode_text(self, tokens: Iterable[int]) -> str:
        ids = [int(tok) for tok in tokens if int(tok) not in self.special_ids]
        if not ids:
            return ""
        return self.vokenizer.omni_tokenizer.decode(ids)

    def split_megatron_sequence(self, tokens: Iterable[int]) -> tuple[list[int], list[int]]:
        """Extract the first audio segment and all text tokens from a sequence."""
        seq = [int(tok) for tok in tokens]
        audio_start_id = int(self.vokenizer.audio_start_id)
        audio_end_id = int(self.vokenizer.audio_end_id)

        audio_tokens: list[int] = []
        text_tokens: list[int] = []
        i = 0
        while i < len(seq):
            tok = seq[i]
            if tok == audio_start_id:
                j = i + 1
                while j < len(seq) and seq[j] != audio_end_id:
                    j += 1
                if j >= len(seq):
                    raise RuntimeError("Megatron sequence has audio_start without matching audio_end")
                if not audio_tokens:
                    audio_tokens = seq[i : j + 1]
                i = j + 1
                continue
            if tok < self.audio_token_offset and tok not in self.special_ids:
                text_tokens.append(tok)
            i += 1
        if not audio_tokens:
            raise RuntimeError("Megatron sequence has no audio segment")
        return audio_tokens, text_tokens

    def decode_audio_to_file(self, audio_tokens: Iterable[int], path: Path) -> float:
        import soundfile as sf

        token_tensor = torch.tensor(list(audio_tokens), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            _codes, audio, info = self.vokenizer.detokenize(token_tensor)
        audio_np = audio.detach().float().cpu().numpy()
        sample_rate = int(info.get("output_sample_rate", self.sample_rate))
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio_np, sample_rate)
        return float(len(audio_np) / sample_rate)


def sample_megatron_outputs(
    root: Path,
    *,
    decoder: TokenDecoder,
    num_samples: int,
    seed: int,
    recursive: bool,
    max_audio_seconds: float | None,
    reference_texts: Mapping[str, str] | None,
) -> list[SemanticSample]:
    prefixes = discover_cut_id_prefixes(root, recursive=recursive)
    candidates: list[tuple[Path, int, str]] = []
    for prefix in prefixes:
        with MegatronChunkReader(prefix) as reader:
            sidecar = read_cut_id_sidecar(Path(str(prefix) + CUT_ID_SIDECAR_SUFFIX))
            if len(sidecar) != reader.document_count:
                raise RuntimeError(
                    f"Sidecar/document count mismatch for {prefix}: "
                    f"sidecar={len(sidecar)}, indexed_docs={reader.document_count}"
                )
        candidates.extend((prefix, doc_idx, cut_id) for doc_idx, cut_id in enumerate(sidecar))

    rng = random.Random(seed)
    rng.shuffle(candidates)

    samples: list[SemanticSample] = []
    readers: dict[Path, MegatronChunkReader] = {}
    try:
        for prefix, doc_idx, cut_id in candidates:
            reader = readers.get(prefix)
            if reader is None:
                reader = MegatronChunkReader(prefix)
                readers[prefix] = reader
            seq = reader.read_document(doc_idx)
            audio_tokens, text_tokens = decoder.split_megatron_sequence(seq)
            duration_sec = _audio_duration_from_tokens(len(audio_tokens))
            if max_audio_seconds is not None and duration_sec > max_audio_seconds:
                continue
            decoded_text = decoder.decode_text(text_tokens)
            expected_text = ""
            reference_missing = True
            if reference_texts is not None and cut_id in reference_texts:
                expected_text = reference_texts[cut_id]
                reference_missing = False
            samples.append(
                SemanticSample(
                    sample_id=f"{prefix.name}:{doc_idx}",
                    mode="megatron",
                    audio_tokens=audio_tokens,
                    expected_text=expected_text,
                    decoded_text_tokens=decoded_text,
                    metadata={
                        "cut_id": cut_id,
                        "prefix": str(prefix),
                        "doc_index": doc_idx,
                        "audio_token_count": len(audio_tokens),
                        "text_token_count": len(text_tokens),
                        "reference_text_missing": reference_missing,
                        "decoded_text_matches_source": (
                            bool(expected_text)
                            and normalize_text(decoded_text) == normalize_text(expected_text)
                        ),
                    },
                    context_before=[],
                    context_after=[],
                )
            )
            if len(samples) >= num_samples:
                break
    finally:
        for reader in readers.values():
            reader.close()
    return samples


def sample_interleave_cache(
    root: Path,
    *,
    decoder: TokenDecoder,
    num_samples: int,
    seed: int,
    max_audio_seconds: float | None,
    context_window: int,
) -> list[SemanticSample]:
    import polars as pl

    partition_dirs = list_interleave_cache_partitions(root)
    rng = random.Random(seed)
    samples: list[SemanticSample] = []

    # Keep this validation path simple and explicit: it is an operator canary,
    # not the high-throughput materializer.
    loaded = []
    total_rows = 0
    for partition_dir in partition_dirs:
        df, reader = load_interleave_cache(partition_dir, include_text=True)
        if df.is_empty():
            continue
        sort_cols = ["source_id", "clip_num"]
        if "clip_start" in df.columns:
            sort_cols = ["source_id", "clip_start", "clip_num"]
        sorted_df = df.sort(sort_cols)
        loaded.append((partition_dir, sorted_df, reader))
        total_rows += len(sorted_df)

    if total_rows == 0:
        return []

    candidates = []
    for part_idx, (_partition_dir, df, _reader) in enumerate(loaded):
        candidates.extend((part_idx, row_idx) for row_idx in range(len(df)))
    rng.shuffle(candidates)

    for part_idx, row_idx in candidates:
        partition_dir, df, reader = loaded[part_idx]
        row = df.row(row_idx, named=True)
        audio_len = int(row["audio_token_length"])
        duration_sec = _audio_duration_from_tokens(audio_len)
        if max_audio_seconds is not None and duration_sec > max_audio_seconds:
            continue

        source_id = row["source_id"]
        start = max(0, row_idx - context_window)
        stop = min(len(df), row_idx + context_window + 1)
        window = df.slice(start, stop - start)
        # Do not cross source boundaries when building human-readable context.
        window = window.filter(pl.col("source_id") == source_id)
        cache = reader.prepare(window)
        rows = window.to_dicts()
        center = next(
            idx
            for idx, candidate in enumerate(rows)
            if candidate["source_id"] == row["source_id"]
            and int(candidate["clip_num"]) == int(row["clip_num"])
            and int(candidate["audio_token_offset"]) == int(row["audio_token_offset"])
        )
        decoded_texts = [decoder.decode_text(cache.text.get(idx)) for idx in range(len(rows))]
        source_texts = [
            str(row_data.get("text") or decoded_texts[idx])
            for idx, row_data in enumerate(rows)
        ]
        audio_tokens = cache.audio.get(center)
        expected_text = str(row.get("text") or decoded_texts[center])

        samples.append(
            SemanticSample(
                sample_id=f"{partition_dir.name}:{row['source_id']}:{row['clip_num']}",
                mode="interleave",
                audio_tokens=audio_tokens,
                expected_text=expected_text,
                decoded_text_tokens=decoded_texts[center],
                metadata={
                    "partition": str(partition_dir),
                    "clip_id": row.get("clip_id"),
                    "source_id": row["source_id"],
                    "clip_num": int(row["clip_num"]),
                    "clip_start": row.get("clip_start"),
                    "clip_duration": row.get("clip_duration"),
                    "speaker": row.get("speaker"),
                    "duration": row.get("duration"),
                    "dataset": row.get("dataset"),
                    "audio_token_count": len(audio_tokens),
                    "text_token_count": int(row["text_token_length"]),
                    "decoded_text_matches_source": (
                        normalize_text(decoded_texts[center]) == normalize_text(expected_text)
                    ),
                },
                context_before=source_texts[:center],
                context_after=source_texts[center + 1 :],
            )
        )
        if len(samples) >= num_samples:
            break
    return samples


def transcribe_with_parakeet(
    wav_paths: list[Path],
    *,
    model_path: str,
    device: str,
    batch_size: int,
) -> list[str]:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=device)
    model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    outputs = model.transcribe([str(path) for path in wav_paths], batch_size=batch_size)
    return [_hypothesis_text(output) for output in outputs]


def transcribe_with_whisper_hf(
    wav_paths: list[Path],
    *,
    model_path: str,
    device: str,
    batch_size: int,
    language: str | None,
) -> list[str]:
    """Transcribe with a local HuggingFace Whisper checkpoint."""
    import soundfile as sf
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    if device == "cuda":
        model = model.to("cuda")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if device == "cuda" else -1,
        batch_size=batch_size,
    )
    inputs = []
    for path in wav_paths:
        audio, sample_rate = sf.read(path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        inputs.append({"array": audio, "sampling_rate": sample_rate})
    kwargs: dict[str, object] = {"return_timestamps": True}
    if language:
        kwargs["generate_kwargs"] = {"language": language, "task": "transcribe"}
    outputs = pipe(inputs, **kwargs)
    return [str(output.get("text", "")) if isinstance(output, dict) else str(output) for output in outputs]


def run_validation(args: argparse.Namespace) -> dict[str, object]:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    decoder = TokenDecoder(args.tokenizer_path, device=device)
    if args.format == "megatron":
        reference_texts = None
        if args.reference_shar_dir:
            reference_texts = load_reference_texts_from_shar(
                Path(path) for path in args.reference_shar_dir
            )
        samples = sample_megatron_outputs(
            input_dir,
            decoder=decoder,
            num_samples=args.num_samples,
            seed=args.seed,
            recursive=args.recursive,
            max_audio_seconds=args.max_audio_seconds,
            reference_texts=reference_texts,
        )
    else:
        samples = sample_interleave_cache(
            input_dir,
            decoder=decoder,
            num_samples=args.num_samples,
            seed=args.seed,
            max_audio_seconds=args.max_audio_seconds,
            context_window=args.context_window,
        )

    wav_paths: list[Path] = []
    records: list[dict[str, object]] = []
    for idx, sample in enumerate(samples):
        wav_path = output_dir / f"sample_{idx:04d}.wav"
        duration_sec = decoder.decode_audio_to_file(sample.audio_tokens, wav_path)
        record = asdict(sample)
        record.pop("audio_tokens")
        record["wav_path"] = str(wav_path)
        record["decoded_duration_sec"] = duration_sec
        records.append(record)
        wav_paths.append(wav_path)

    if not args.no_asr and wav_paths:
        if args.asr_backend == "parakeet":
            transcripts = transcribe_with_parakeet(
                wav_paths,
                model_path=args.asr_model,
                device=device,
                batch_size=args.asr_batch_size,
            )
        elif args.asr_backend == "whisper-hf":
            transcripts = transcribe_with_whisper_hf(
                wav_paths,
                model_path=args.whisper_model,
                device=device,
                batch_size=args.asr_batch_size,
                language=args.asr_language,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported ASR backend: {args.asr_backend!r}")
        for record, transcript in zip(records, transcripts):
            record["asr_text"] = transcript
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            if metadata.get("reference_text_missing"):
                record["similarity"] = None
            else:
                record["similarity"] = compare_texts(str(record.get("expected_text", "")), transcript)
    else:
        for record in records:
            record["asr_text"] = None
            record["similarity"] = None

    report_path = output_dir / "samples.jsonl"
    with atomic_streaming_write(report_path, mode="w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
            f.write("\n")

    scored = [r["similarity"] for r in records if isinstance(r.get("similarity"), dict)]
    summary = {
        "input_dir": str(input_dir),
        "format": args.format,
        "num_samples": len(records),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "asr_backend": None if args.no_asr else args.asr_backend,
        "asr_model": (
            None
            if args.no_asr
            else args.asr_model if args.asr_backend == "parakeet" else args.whisper_model
        ),
        "mean_char_ratio": _mean(float(s["char_ratio"]) for s in scored),
        "mean_word_recall": _mean(float(s["word_recall"]) for s in scored),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _audio_duration_from_tokens(num_audio_tokens: int) -> float:
    # audio_tokens includes audio_start/audio_end. WavTokenizer-40 is 40 tokens/s.
    return max(0, num_audio_tokens - 2) / 40.0


def _hypothesis_text(output) -> str:
    if isinstance(output, str):
        return output
    if hasattr(output, "text"):
        return str(output.text)
    if isinstance(output, (list, tuple)) and output:
        return _hypothesis_text(output[0])
    return str(output)


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample tokenized audio outputs, reconstruct audio, transcribe with "
            "local Parakeet ASR, and compare ASR text to stored text tokens."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Megatron output dir or interleave cache dir")
    parser.add_argument("--format", choices=["megatron", "interleave"], required=True)
    parser.add_argument("--output-dir", required=True, help="Directory for WAVs and JSONL report")
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--asr-backend", choices=["parakeet", "whisper-hf"], default="parakeet")
    parser.add_argument(
        "--asr-language",
        default=None,
        help="Optional Whisper language hint, e.g. german, icelandic, english.",
    )
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--asr-batch-size", type=int, default=8)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan Megatron chunks")
    parser.add_argument(
        "--reference-shar-dir",
        action="append",
        default=[],
        help=(
            "SHAR directory containing source cut manifests for Megatron reference text. "
            "May be supplied multiple times."
        ),
    )
    parser.add_argument("--no-asr", action="store_true", help="Only reconstruct WAVs and decode expected text")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_validation(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
