#!/usr/bin/env python3
"""Run inference on a Stage 2 audio-text checkpoint with audio prompts.

Supports two input formats:
  - HuggingFace dataset (Arrow, via --audio-dir with load_from_disk)
  - Parquet files with FLAC audio bytes (via --parquet-dir)

Usage
-----
    # Arrow dataset (e.g. eurospeech)
    python scripts/audio_inference.py \
        --model-path /capstor/.../audio-weight-1-phase-transition \
        --audio-dir /capstor/.../eurospeech_cache/uk \
        --task transcribe --num-samples 5

    # Parquet with FLAC bytes (e.g. spc-r-segmented)
    python scripts/audio_inference.py \
        --model-path /capstor/.../audio-weight-1-phase-transition \
        --parquet-dir /capstor/.../spc-r-segmented/test \
        --task transcribe --num-samples 5
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from glob import glob

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# WavTokenizer import (from local repo)
# ---------------------------------------------------------------------------
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

from src.audio_tokenizers.implementations.wavtokenizer import WavTokenizer40


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AUDIO_TOKEN_OFFSET = 262344
AUDIO_VOCAB_SIZE = 4096


def _downmix_audio_array(audio_array: np.ndarray, *, channel_layout: str | None) -> np.ndarray:
    """Return mono audio while preserving the time axis for known layouts.

    HuggingFace ``datasets.Audio(mono=False)`` decodes stereo as ``(channels, time)``.
    ``soundfile.read()`` returns multi-channel audio as ``(time, channels)``.
    """
    if audio_array.ndim != 2:
        return audio_array
    if channel_layout == "channels_first":
        return audio_array.mean(axis=0)
    if channel_layout == "channels_last":
        return audio_array.mean(axis=1)

    # Conservative fallback for unknown 2D layouts: small leading dimension is
    # usually channel count, otherwise assume soundfile's (time, channels).
    if audio_array.shape[0] <= 8 and audio_array.shape[1] > audio_array.shape[0]:
        return audio_array.mean(axis=0)
    return audio_array.mean(axis=1)


def load_special_token_ids(tokenizer, tokenizer_path: str) -> dict[str, int]:
    """Load audio structure token IDs from the mapping file (single source of truth)."""
    from audio_tokenization.utils.token_mapping import get_structure_tokens

    required = ["audio_start", "audio_end", "stt_transcribe", "stt_continue", "tts_continue", "stt_translate"]
    st = get_structure_tokens(tokenizer_path, required=required)
    return {key: st[key] for key in required}


def build_prompt(
    audio_codes: torch.Tensor,
    task: str,
    bos_id: int,
    special_ids: dict[str, int],
) -> list[int]:
    """Build a prompt token list from raw WavTokenizer codes."""
    shifted = (audio_codes + AUDIO_TOKEN_OFFSET).tolist()

    prompt = [bos_id]
    prompt.append(special_ids["audio_start"])
    prompt.extend(shifted)
    prompt.append(special_ids["audio_end"])

    if task == "transcribe":
        prompt.append(special_ids["stt_transcribe"])
    elif task == "continue":
        prompt.append(special_ids["stt_continue"])
    elif task == "translate":
        prompt.append(special_ids["stt_translate"])

    return prompt


def decode_output(
    generated_ids: list[int],
    tokenizer,
    prompt_len: int,
) -> str:
    """Decode generated token IDs to text, skipping audio-range tokens and structure tokens."""
    new_ids = generated_ids[prompt_len:]
    text_ids = [
        tid for tid in new_ids
        if not (AUDIO_TOKEN_OFFSET <= tid < AUDIO_TOKEN_OFFSET + AUDIO_VOCAB_SIZE)
    ]
    return tokenizer.decode(text_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_arrow_dataset(audio_dir: str, audio_column: str, num_samples: int):
    """Load samples from a HuggingFace Arrow dataset."""
    import datasets as hf_datasets

    ds = hf_datasets.load_from_disk(audio_dir)
    if isinstance(ds, hf_datasets.DatasetDict):
        split_name = list(ds.keys())[0]
        print(f"  DatasetDict detected, using split '{split_name}'")
        ds = ds[split_name]

    n_total = len(ds)
    n = min(num_samples, n_total)
    print(f"  {n_total} samples available, will process {n}")

    samples = []
    for i in range(n):
        row = ds[i]
        audio_field = row[audio_column]
        if isinstance(audio_field, dict):
            audio_array = np.array(audio_field["array"], dtype=np.float32)
            sr = audio_field["sampling_rate"]
            channel_layout = "channels_first"
        elif isinstance(audio_field, (np.ndarray, list)):
            audio_array = np.array(audio_field, dtype=np.float32)
            sr = 16000
            channel_layout = None
        else:
            continue

        text = row.get("text", "")
        sample_id = row.get("sample_id", row.get("id", i))
        samples.append({
            "audio_array": audio_array,
            "sr": sr,
            "text": text or "",
            "id": sample_id,
            "channel_layout": channel_layout,
        })
    return samples


def load_parquet_dataset(parquet_dir: str, audio_column: str, num_samples: int):
    """Load samples from parquet files with FLAC audio bytes."""
    import polars as pl

    files = sorted(glob(os.path.join(parquet_dir, "shard_*.parquet")))
    if not files:
        files = sorted(glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"  Found {len(files)} parquet files")

    samples = []
    for f in files:
        if len(samples) >= num_samples:
            break
        remaining = num_samples - len(samples)
        df = pl.read_parquet(f, n_rows=remaining)

        for row_idx in range(len(df)):
            audio_struct = df[audio_column][row_idx]
            audio_bytes = audio_struct["bytes"]
            sr_parquet = audio_struct["sampling_rate"]

            # Decode FLAC bytes to numpy array
            audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if sr != sr_parquet:
                sr = sr_parquet  # trust parquet metadata

            text = df["text"][row_idx] if "text" in df.columns else ""
            sample_id = df["id"][row_idx] if "id" in df.columns else row_idx

            samples.append({
                "audio_array": audio_array,
                "sr": sr,
                "text": text or "",
                "id": sample_id,
                "channel_layout": "channels_last",
            })

    print(f"  Loaded {len(samples)} samples")
    return samples


def load_wav_dir_dataset(wav_dir: str, num_samples: int):
    """Load audio files from a directory of wav/mp3/flac files.

    Optionally reads metadata.tsv (tab-separated: filename<TAB>text) for
    ground-truth transcriptions.
    """
    exts = ("*.wav", "*.mp3", "*.flac")
    audio_files = []
    for ext in exts:
        audio_files.extend(sorted(glob(os.path.join(wav_dir, ext))))
    audio_files.sort()
    print(f"  Found {len(audio_files)} audio files")

    # Optional metadata for ground-truth text (and optional dataset column)
    meta_path = os.path.join(wav_dir, "metadata.tsv")
    text_map: dict[str, str] = {}
    dataset_map: dict[str, str] = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            header = f.readline().strip().split("\t")
            has_dataset_col = len(header) >= 3 and header[2] == "dataset"
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    text_map[parts[0]] = parts[1]
                if has_dataset_col and len(parts) >= 3:
                    dataset_map[parts[0]] = parts[2]
        print(f"  Loaded metadata for {len(text_map)} files")

    n = min(num_samples, len(audio_files))
    samples = []
    for path in audio_files[:n]:
        audio_array, sr = sf.read(path, dtype="float32")
        fname = os.path.basename(path)
        sample_id = os.path.splitext(fname)[0]
        samples.append({
            "audio_array": audio_array,
            "sr": sr,
            "text": text_map.get(fname, ""),
            "id": sample_id,
            "dataset": dataset_map.get(fname, ""),
            "channel_layout": "channels_last",
        })

    print(f"  Loaded {len(samples)} samples")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio inference with a Stage 2 audio-text checkpoint.",
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the HF-format model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default=None,
        help="Path to the tokenizer (defaults to --model-path).",
    )
    # Input: one of these three
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio-dir", type=str,
        help="Path to a HuggingFace dataset saved with save_to_disk (Arrow).",
    )
    input_group.add_argument(
        "--parquet-dir", type=str,
        help="Path to directory with parquet files (FLAC audio bytes).",
    )
    input_group.add_argument(
        "--wav-dir", type=str,
        help="Path to directory with audio files (wav/mp3/flac). "
             "Optionally include metadata.tsv (filename<TAB>text) for ground truth.",
    )
    parser.add_argument(
        "--task", type=str, choices=["transcribe", "continue", "translate"],
        default="transcribe",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. 0 = greedy.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--backend", type=str, choices=["transformers", "vllm"], default="transformers",
        help="Inference backend to use.",
    )
    parser.add_argument("--audio-column", type=str, default="audio",
                        help="Column name containing audio data.")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Name for the dataset (used in output dir). "
                             "Auto-derived from --audio-dir/--parquet-dir basename if not specified.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save results as JSON. Auto-generated if not specified.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip peak normalization (-3 dBFS).")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or args.model_path

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from {tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    special_ids = load_special_token_ids(tokenizer, tokenizer_path)
    print(f"  bos={bos_id}  eos={eos_id}")
    print(f"  audio_start={special_ids['audio_start']}  audio_end={special_ids['audio_end']}")
    print(f"  stt_transcribe={special_ids['stt_transcribe']}  stt_continue={special_ids['stt_continue']}  tts_continue={special_ids['tts_continue']}  stt_translate={special_ids['stt_translate']}")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print(f"\nLoading model from {args.model_path} with backend={args.backend} ...")
    t0 = time.time()
    if args.backend == "transformers":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    else:
        try:
            os.environ["VLLM_USE_V1"] = "0"
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM backend requested but vllm is not installed. "
                "Install it or run with --backend transformers."
            ) from e
        model = LLM(
            model=args.model_path,
            tokenizer=tokenizer_path,
            trust_remote_code=True,
            dtype="bfloat16",
        )

    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Load WavTokenizer (for encoding audio → codes)
    # ------------------------------------------------------------------
    print("\nLoading WavTokenizer (40 tokens/s) ...")
    wav_tokenizer = WavTokenizer40(device="cuda", torch_compile=False)
    print("  WavTokenizer ready")

    # ------------------------------------------------------------------
    # 4. Load audio samples
    # ------------------------------------------------------------------
    data_source = args.audio_dir or args.parquet_dir or args.wav_dir
    print(f"\nLoading audio from {data_source} ...")
    if args.audio_dir:
        samples = load_arrow_dataset(args.audio_dir, args.audio_column, args.num_samples)
    elif args.parquet_dir:
        samples = load_parquet_dataset(args.parquet_dir, args.audio_column, args.num_samples)
    else:
        samples = load_wav_dir_dataset(args.wav_dir, args.num_samples)

    # ------------------------------------------------------------------
    # 5. Determine output directory and file path
    # ------------------------------------------------------------------
    dataset_name = args.dataset_name or os.path.basename(data_source.rstrip("/"))
    model_name = os.path.basename(args.model_path.rstrip("/"))
    output_dir = os.path.join("results/inference", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = os.path.join(output_dir, f"{model_name}_{args.task}.json")

    # ------------------------------------------------------------------
    # 6. Stop token IDs — stop at EOS or audio_start (to avoid generating
    #    useless audio tokens after a switch/continue response)
    # ------------------------------------------------------------------
    stop_token_ids = [eos_id, special_ids["audio_start"]]
    vllm_sampling_params = None
    if args.backend == "vllm":
        vllm_top_p = args.top_p if args.temperature > 0 else 1.0
        vllm_sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=vllm_top_p,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
        )

    # ------------------------------------------------------------------
    # 7. Inference loop
    # ------------------------------------------------------------------
    print(f"\nTask: {args.task}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Output dir: {output_dir}")
    print(f"Output file: {output_file}")
    print("=" * 70)

    results = []
    _resamplers: dict[int, torchaudio.transforms.Resample] = {}

    for i, sample in enumerate(samples):
        audio_array = sample["audio_array"]
        sr = sample["sr"]
        ground_truth = sample["text"]
        sample_id = sample["id"]
        sample_dataset = sample.get("dataset", "")
        channel_layout = sample.get("channel_layout")

        audio_array = _downmix_audio_array(
            audio_array,
            channel_layout=channel_layout,
        )
        audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)  # (1, T)

        duration_s = audio_tensor.shape[-1] / sr

        # Export wav file into the output directory (skip if already exists
        # or if loading from wav-dir where files are already on disk)
        if not args.wav_dir:
            wav_name = f"sample_{i}.wav"
            if isinstance(sample_id, str) and sample_id != str(i):
                wav_name = f"{sample_id}.wav"
            wav_path = os.path.join(output_dir, wav_name)
            if not os.path.exists(wav_path):
                sf.write(wav_path, audio_array, sr)

        # Resample to 24 kHz for WavTokenizer (cache resampler per sample rate)
        if sr != 24000:
            if sr not in _resamplers:
                _resamplers[sr] = torchaudio.transforms.Resample(sr, 24000)
            audio_24k = _resamplers[sr](audio_tensor)
        else:
            audio_24k = audio_tensor

        # Peak normalize to -3 dBFS (matches tokenization pipeline)
        if not args.no_normalize:
            peak = audio_24k.abs().max().clamp(min=1e-10)
            target_peak = 10 ** (-3.0 / 20.0)
            audio_24k = audio_24k * (target_peak / peak)

        # Encode with WavTokenizer
        with torch.no_grad():
            codes = wav_tokenizer.encode_audio(audio_24k)  # (1, N)
        codes = codes.squeeze(0).cpu()  # (N,)

        # Build prompt
        prompt_ids = build_prompt(codes, args.task, bos_id, special_ids)

        # Generate
        t_gen = time.time()
        if args.backend == "transformers":
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                eos_token_id=stop_token_ids,
            )
            if args.temperature == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                output = model.generate(prompt_tensor, **gen_kwargs)
            generated_ids = output[0].tolist()
        else:
            request_outputs = model.generate(
                [{"prompt_token_ids": prompt_ids}],
                sampling_params=vllm_sampling_params,
                use_tqdm=False,
            )
            if not request_outputs or not request_outputs[0].outputs:
                generated_ids = prompt_ids
            else:
                completion_ids = list(request_outputs[0].outputs[0].token_ids)
                generated_ids = prompt_ids + completion_ids
        gen_time = time.time() - t_gen

        n_new = len(generated_ids) - len(prompt_ids)

        # Decode text portion of output
        text_output = decode_output(generated_ids, tokenizer, len(prompt_ids))

        # Count audio tokens in output
        new_ids = generated_ids[len(prompt_ids):]
        n_audio_out = sum(
            1 for tid in new_ids
            if AUDIO_TOKEN_OFFSET <= tid < AUDIO_TOKEN_OFFSET + AUDIO_VOCAB_SIZE
        )
        n_text_out = n_new - n_audio_out

        # Name the model output based on the task
        output_label = {"transcribe": "transcription", "continue": "continuation", "translate": "translation"}[args.task]

        result = {
            "sample_idx": i,
            "sample_id": str(sample_id),
            "duration_s": round(duration_s, 2),
            "audio_codes": len(codes),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": n_new,
            "text_tokens": n_text_out,
            "audio_tokens_out": n_audio_out,
            "gen_time_s": round(gen_time, 2),
            output_label: text_output,
            "ground_truth": ground_truth,
        }
        if sample_dataset:
            result["dataset"] = sample_dataset
        results.append(result)

        print(f"\n--- Sample {i} (id={sample_id}) ---")
        print(f"  Duration: {duration_s:.2f}s | Audio codes: {len(codes)} | Prompt tokens: {len(prompt_ids)}")
        print(f"  Generated: {n_new} tokens ({n_text_out} text, {n_audio_out} audio) in {gen_time:.2f}s")
        if ground_truth:
            print(f"  Ground Truth: {ground_truth[:200]}")
        print(f"  {output_label.title()}: {text_output}")
        print()

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    output_data = {
        "model_path": args.model_path,
        "data_source": data_source,
        "dataset_name": dataset_name,
        "task": args.task,
        "backend": args.backend,
        "num_samples": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "results": results,
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"Results saved to {output_file}")
    print("Done.")


if __name__ == "__main__":
    main()
