"""Reconstruct audio+text from Megatron indexed datasets to verify correctness.

Usage:
    python scripts/reconstruct_from_indexed.py \
        --prefix /path/to/dataset_name \
        --seq-idx 42 \
        --output-dir /tmp/reconstruct
"""

import argparse
import struct
import numpy as np
import os
import sys
import torch
import soundfile as sf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_MAP = {
    1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
    5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16,
}
_TOKENIZER_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok"


def read_idx(idx_path):
    with open(idx_path, "rb") as f:
        assert f.read(9) == _INDEX_HEADER
        f.read(8)  # version
        dtype = _DTYPE_MAP[struct.unpack("<B", f.read(1))[0]]
        seq_count = struct.unpack("<Q", f.read(8))[0]
        doc_count = struct.unpack("<Q", f.read(8))[0]
        seq_lengths = np.frombuffer(f.read(seq_count * 4), dtype=np.int32)
        pointers = np.frombuffer(f.read(seq_count * 8), dtype=np.int64)
    return dtype, seq_lengths, pointers


def read_sequence(bin_path, pointer, length, dtype):
    itemsize = np.dtype(dtype).itemsize
    with open(bin_path, "rb") as f:
        f.seek(pointer)
        return np.frombuffer(f.read(length * itemsize), dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Dataset prefix (without .bin/.idx)")
    parser.add_argument("--seq-idx", type=int, default=None, help="Sequence index (default: random)")
    parser.add_argument("--output-dir", default="/tmp/reconstruct")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read index
    dtype, seq_lengths, pointers = read_idx(args.prefix + ".idx")
    print(f"Dataset: {os.path.basename(args.prefix)}")
    print(f"  {len(seq_lengths)} sequences, dtype={dtype.__name__}")

    # Pick sequence
    if args.seq_idx is None:
        args.seq_idx = np.random.randint(0, len(seq_lengths))
    seq_len = seq_lengths[args.seq_idx]
    print(f"\nSequence {args.seq_idx}: {seq_len} tokens")

    # Read tokens
    tokens = read_sequence(args.prefix + ".bin", pointers[args.seq_idx], seq_len, dtype)
    tokens_tensor = torch.from_numpy(tokens.astype(np.int64)).long()

    # Load vokenizer (has detokenize + text tokenizer)
    from audio_tokenization.vokenizers.wavtokenizer.audio_only import WavTokenizerAudioOnly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vok = WavTokenizerAudioOnly(omni_tokenizer_path=_TOKENIZER_PATH, device=device, torch_compile=False)

    # Get special token IDs
    omni = vok.omni_tokenizer
    stt_continue_id = omni.convert_tokens_to_ids("<|stt_continue|>")
    stt_transcribe_id = omni.convert_tokens_to_ids("<|stt_transcribe|>")
    tts_continue_id = omni.convert_tokens_to_ids("<|tts_continue|>")
    print(f"  bos={vok.bos_id} eos={vok.eos_id} audio_start={vok.audio_start_id} "
          f"audio_end={vok.audio_end_id} offset={vok.audio_token_offset}")
    print(f"  stt_continue={stt_continue_id} stt_transcribe={stt_transcribe_id} "
          f"tts_continue={tts_continue_id}")

    # Parse structure and print
    tokens_list = tokens.tolist()
    print(f"\n  First 10: {tokens_list[:10]}")
    print(f"  Last 10:  {tokens_list[-10:]}")

    # Find audio/text segments
    audio_seg_count = 0
    text_seg_count = 0
    i = 0
    while i < len(tokens_list):
        t = tokens_list[i]
        if t == vok.audio_start_id:
            j = i + 1
            while j < len(tokens_list) and tokens_list[j] != vok.audio_end_id:
                j += 1
            n = j - i - 1
            print(f"  [AUDIO seg {audio_seg_count}] {n} tokens ({n/40:.2f}s) at pos {i+1}:{j}")
            audio_seg_count += 1
            i = j + 1
        elif t in (vok.bos_id, vok.eos_id, stt_continue_id, stt_transcribe_id, tts_continue_id):
            name = {vok.bos_id: "BOS", vok.eos_id: "EOS",
                    stt_continue_id: "STT_CONTINUE", stt_transcribe_id: "STT_TRANSCRIBE",
                    tts_continue_id: "TTS_CONTINUE"}.get(t)
            print(f"  [{name}] at pos {i}")
            i += 1
        elif t < vok.audio_token_offset:
            j = i
            while j < len(tokens_list) and tokens_list[j] < vok.audio_token_offset and tokens_list[j] not in (
                    vok.bos_id, vok.eos_id, vok.audio_start_id, vok.audio_end_id,
                    stt_continue_id, stt_transcribe_id, tts_continue_id):
                j += 1
            text = omni.decode(tokens_list[i:j])
            print(f"  [TEXT seg {text_seg_count}] {j-i} tokens at pos {i}:{j}")
            print(f"    \"{text[:300]}{'...' if len(text) > 300 else ''}\"")
            text_seg_count += 1
            i = j
        else:
            i += 1

    # Decode all audio segments using detokenize
    # detokenize expects the full sequence with audio_start/audio_end markers
    # For interleaved with multiple audio segments, decode each one separately
    print(f"\n--- Decoding {audio_seg_count} audio segment(s) ---")
    i = 0
    seg_idx = 0
    while i < len(tokens_list):
        if tokens_list[i] == vok.audio_start_id:
            j = i + 1
            while j < len(tokens_list) and tokens_list[j] != vok.audio_end_id:
                j += 1
            # Build a minimal sequence: [audio_start, audio_tokens..., audio_end]
            seg_tokens = torch.tensor(tokens_list[i:j+1], dtype=torch.long)
            raw_codes = seg_tokens[1:-1] - vok.audio_token_offset
            codes_tensor = raw_codes.unsqueeze(0).to(device)
            audio_out = vok.wavtokenizer.decode_tokens(codes_tensor)
            audio_np = audio_out.squeeze().cpu().numpy()

            name = os.path.basename(args.prefix)
            out_path = os.path.join(args.output_dir, f"{name}_seq{args.seq_idx}_audio{seg_idx}.wav")
            sf.write(out_path, audio_np, 24000)
            print(f"  Saved: {out_path} ({len(audio_np)/24000:.2f}s)")
            seg_idx += 1
            i = j + 1
        else:
            i += 1

    # Save text
    if text_seg_count > 0:
        out_path = os.path.join(args.output_dir, f"{name}_seq{args.seq_idx}_text.txt")
        # Re-extract text segments
        texts = []
        i = 0
        t_idx = 0
        while i < len(tokens_list):
            t = tokens_list[i]
            if t == vok.audio_start_id:
                j = i + 1
                while j < len(tokens_list) and tokens_list[j] != vok.audio_end_id:
                    j += 1
                i = j + 1
            elif t < vok.audio_token_offset and t not in (
                    vok.bos_id, vok.eos_id, stt_continue_id, stt_transcribe_id, tts_continue_id):
                j = i
                while j < len(tokens_list) and tokens_list[j] < vok.audio_token_offset and tokens_list[j] not in (
                        vok.bos_id, vok.eos_id, vok.audio_start_id, vok.audio_end_id,
                        stt_continue_id, stt_transcribe_id, tts_continue_id):
                    j += 1
                texts.append(f"--- Segment {t_idx} ---\n{omni.decode(tokens_list[i:j])}")
                t_idx += 1
                i = j
            else:
                i += 1
        with open(out_path, "w") as f:
            f.write("\n\n".join(texts))
        print(f"  Saved: {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
