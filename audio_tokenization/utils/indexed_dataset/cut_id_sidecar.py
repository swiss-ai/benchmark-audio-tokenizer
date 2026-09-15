"""Cut-ID sidecar readers and cross-run comparison helpers.

Megatron ``.bin/.idx`` chunks are positional. The companion
``.cut_ids.jsonl.zst`` sidecar makes the production contract explicit:
line ``N`` identifies document ``N`` in the matching chunk.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from audio_tokenization.utils.indexed_dataset.constants import (
    CUT_ID_SIDECAR_SUFFIX,
    MEGATRON_INDEX_HEADER,
)
from audio_tokenization.utils.indexed_dataset.dtypes import DType
from audio_tokenization.utils.io import open_compressed


TokenTuple = tuple[int, ...]


def discover_cut_id_prefixes(root: str | Path, *, recursive: bool = False) -> list[Path]:
    """Find complete Megatron chunks that also have cut-ID sidecars."""
    root_path = Path(root).expanduser().resolve()
    pattern = "**/*.idx" if recursive else "*.idx"
    prefixes: list[Path] = []
    missing: list[Path] = []

    for idx_path in sorted(root_path.glob(pattern)):
        prefix = idx_path.with_suffix("")
        bin_path = prefix.with_suffix(".bin")
        sidecar_path = Path(str(prefix) + CUT_ID_SIDECAR_SUFFIX)
        if bin_path.is_file() and sidecar_path.is_file():
            prefixes.append(prefix)
        else:
            missing.append(prefix)

    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise RuntimeError(
            f"Found {len(missing)} indexed chunks missing .bin or cut-ID sidecar. "
            f"First entries:\n{preview}"
        )
    if not prefixes:
        raise RuntimeError(f"No indexed chunks with cut-ID sidecars found under {root_path}")
    return prefixes


def read_cut_id_sidecar(path: str | Path) -> list[str]:
    """Read a ``.cut_ids.jsonl.zst`` file as an ordered list of cut IDs."""
    with open_compressed(path, "rt") as f:
        return [json.loads(line) for line in f]


class MegatronChunkReader:
    """Minimal Megatron chunk reader for canary comparison."""

    def __init__(self, prefix: str | Path):
        self.prefix = Path(prefix)
        self.idx_path = self.prefix.with_suffix(".idx")
        self.bin_path = self.prefix.with_suffix(".bin")

        with open(self.idx_path, "rb") as f:
            header = f.read(len(MEGATRON_INDEX_HEADER))
            if header != MEGATRON_INDEX_HEADER:
                raise RuntimeError(f"Bad Megatron index header in {self.idx_path}")
            (version,) = struct.unpack("<Q", f.read(8))
            if version != 1:
                raise RuntimeError(f"Unsupported Megatron index version {version} in {self.idx_path}")
            (dtype_code,) = struct.unpack("<B", f.read(1))
            self.dtype = DType.dtype_from_code(dtype_code)
            self.itemsize = np.dtype(self.dtype).itemsize
            (self.sequence_count,) = struct.unpack("<Q", f.read(8))
            (document_index_count,) = struct.unpack("<Q", f.read(8))
            self.lengths = np.frombuffer(
                f.read(self.sequence_count * np.dtype(np.int32).itemsize),
                dtype=np.int32,
            ).copy()
            self.pointers = np.frombuffer(
                f.read(self.sequence_count * np.dtype(np.int64).itemsize),
                dtype=np.int64,
            ).copy()
            self.document_indices = np.frombuffer(
                f.read(document_index_count * np.dtype(np.int64).itemsize),
                dtype=np.int64,
            ).copy()
        self._data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")

    def close(self) -> None:
        """Close the underlying memmap file handle."""
        data = getattr(self, "_data", None)
        if data is None:
            return
        mmap_obj = getattr(data, "_mmap", None)
        self._data = None
        if mmap_obj is not None:
            mmap_obj.close()

    def __enter__(self) -> "MegatronChunkReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @property
    def document_count(self) -> int:
        return max(0, len(self.document_indices) - 1)

    def read_document(self, document_index: int) -> tuple[int, ...]:
        """Return one document as a token tuple.

        The current tokenizer writes one sequence per document. This reader also
        handles the general Megatron case by concatenating all sequences in a
        document, so the sidecar contract remains tied to documents, not files.
        """
        start = int(self.document_indices[document_index])
        end = int(self.document_indices[document_index + 1])
        parts = []
        for seq_idx in range(start, end):
            offset = int(self.pointers[seq_idx]) // self.itemsize
            length = int(self.lengths[seq_idx])
            parts.append(self._data[offset: offset + length].astype(np.int64, copy=False))
        if not parts:
            return ()
        if len(parts) == 1:
            return tuple(int(x) for x in parts[0])
        return tuple(int(x) for x in np.concatenate(parts))


def collect_cutid_token_pairs(
    root: str | Path,
    *,
    recursive: bool = False,
) -> dict[str, TokenTuple]:
    """Collect a run output directory as ``cut_id -> token tuple``."""
    result: dict[str, TokenTuple] = {}
    for prefix in discover_cut_id_prefixes(root, recursive=recursive):
        with MegatronChunkReader(prefix) as reader:
            sidecar = read_cut_id_sidecar(Path(str(prefix) + CUT_ID_SIDECAR_SUFFIX))
            if len(sidecar) != reader.document_count:
                raise RuntimeError(
                    f"Sidecar/document count mismatch for {prefix}: "
                    f"sidecar={len(sidecar)}, indexed_docs={reader.document_count}"
                )
            for doc_idx, cut_id in enumerate(sidecar):
                if cut_id in result:
                    raise RuntimeError(f"Duplicate cut_id: {cut_id!r}")
                tokens = reader.read_document(doc_idx)
                result[cut_id] = tokens
    return result


def _format_preview(values: list[str], *, limit: int = 5) -> str:
    if not values:
        return "[]"
    preview = ", ".join(repr(value) for value in values[:limit])
    suffix = ", ..." if len(values) > limit else ""
    return f"[{preview}{suffix}]"


def _split_single_audio_span(
    tokens: TokenTuple,
    *,
    cut_id: str,
    audio_start_id: int,
    audio_end_id: int,
) -> tuple[TokenTuple, TokenTuple]:
    """Split marker-wrapped Megatron documents into structure/text and audio.

    Megatron direct/audio-only outputs contain exactly one audio span. The
    comparison keeps tokens outside that span strict while allowing bounded
    drift only inside the audio payload.
    """
    try:
        start = tokens.index(audio_start_id)
    except ValueError as exc:
        raise RuntimeError(
            f"{cut_id!r}: missing audio_start token {audio_start_id}; "
            "cannot apply marker-aware trim tolerance"
        ) from exc
    try:
        end = tokens.index(audio_end_id, start + 1)
    except ValueError as exc:
        raise RuntimeError(
            f"{cut_id!r}: missing audio_end token {audio_end_id}; "
            "cannot apply marker-aware trim tolerance"
        ) from exc
    if audio_start_id in tokens[start + 1:] or audio_end_id in tokens[end + 1:]:
        raise RuntimeError(
            f"{cut_id!r}: multiple audio marker spans found; "
            "marker-aware comparison expects one audio span"
        )
    structure = tokens[: start + 1] + tokens[end:]
    audio = tokens[start + 1: end]
    return structure, audio


def _bounded_prefix_match(
    left: TokenTuple,
    right: TokenTuple,
    *,
    trim_tolerance: int,
) -> tuple[bool, int]:
    """Return whether token content matches under padding-trim tolerance."""
    drift = abs(len(left) - len(right))
    if drift > trim_tolerance:
        return False, drift
    if trim_tolerance == 0:
        return left == right, drift
    safe_prefix = min(len(left), len(right)) - trim_tolerance
    if safe_prefix > 0 and left[:safe_prefix] != right[:safe_prefix]:
        return False, drift
    return True, drift


def _compare_tokens(
    cut_id: str,
    left: TokenTuple,
    right: TokenTuple,
    *,
    trim_tolerance: int,
    audio_start_id: int | None,
    audio_end_id: int | None,
) -> tuple[bool, int]:
    if trim_tolerance < 0:
        raise ValueError("trim_tolerance must be non-negative")
    if (audio_start_id is None) != (audio_end_id is None):
        raise ValueError("audio_start_id and audio_end_id must be provided together")
    if trim_tolerance == 0:
        return left == right, 0
    if audio_start_id is None or audio_end_id is None:
        return _bounded_prefix_match(left, right, trim_tolerance=trim_tolerance)

    left_structure, left_audio = _split_single_audio_span(
        left,
        cut_id=cut_id,
        audio_start_id=audio_start_id,
        audio_end_id=audio_end_id,
    )
    right_structure, right_audio = _split_single_audio_span(
        right,
        cut_id=cut_id,
        audio_start_id=audio_start_id,
        audio_end_id=audio_end_id,
    )
    if left_structure != right_structure:
        return False, abs(len(left_audio) - len(right_audio))
    return _bounded_prefix_match(left_audio, right_audio, trim_tolerance=trim_tolerance)


def compare_cutid_token_sets(
    left_root: str | Path,
    right_root: str | Path,
    *,
    recursive: bool = False,
    trim_tolerance: int = 0,
    audio_start_id: int | None = None,
    audio_end_id: int | None = None,
) -> dict[str, int]:
    """Compare two Megatron output directories by cut identity and tokens.

    By default this is exact. With ``trim_tolerance > 0`` it accepts bounded
    tail drift caused by padding-only token trimming while still requiring
    strict cut identity and strict token-prefix equality. When audio marker IDs
    are provided, tolerance applies only inside the marker-wrapped audio span;
    text and structure tokens remain exact.
    """
    left = collect_cutid_token_pairs(left_root, recursive=recursive)
    right = collect_cutid_token_pairs(right_root, recursive=recursive)

    left_ids = set(left)
    right_ids = set(right)
    missing_left = sorted(right_ids - left_ids)
    missing_right = sorted(left_ids - right_ids)
    changed: list[str] = []
    max_token_drift = 0
    for cut_id in sorted(left_ids & right_ids):
        ok, drift = _compare_tokens(
            cut_id,
            left[cut_id],
            right[cut_id],
            trim_tolerance=trim_tolerance,
            audio_start_id=audio_start_id,
            audio_end_id=audio_end_id,
        )
        max_token_drift = max(max_token_drift, drift)
        if not ok:
            changed.append(cut_id)
    if missing_left or missing_right or changed:
        raise RuntimeError(
            "Cut-id token sets differ: "
            f"missing_from_left={len(missing_left)}, "
            f"missing_from_right={len(missing_right)}, "
            f"token_mismatches={len(changed)}, "
            f"missing_from_left_preview={_format_preview(missing_left)}, "
            f"missing_from_right_preview={_format_preview(missing_right)}, "
            f"token_mismatch_preview={_format_preview(changed)}"
        )
    left_tokens = sum(len(tokens) for tokens in left.values())
    right_tokens = sum(len(tokens) for tokens in right.values())
    if trim_tolerance == 0:
        return {"cuts": len(left), "tokens": left_tokens}
    return {
        "cuts": len(left),
        "tokens_left": left_tokens,
        "tokens_right": right_tokens,
        "max_token_drift": max_token_drift,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two Megatron tokenization outputs by cut_id -> tokens."
    )
    parser.add_argument("left", help="First tokenized output directory")
    parser.add_argument("right", help="Second tokenized output directory")
    parser.add_argument("--recursive", action="store_true", help="Scan chunks recursively")
    parser.add_argument(
        "--trim-tolerance",
        type=int,
        default=0,
        help="Allow bounded per-cut token drift from padding trim",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Tokenizer directory containing audio_token_mapping.json; used to load audio markers",
    )
    parser.add_argument("--audio-start-id", type=int, help="Explicit audio_start token ID")
    parser.add_argument("--audio-end-id", type=int, help="Explicit audio_end token ID")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    audio_start_id = args.audio_start_id
    audio_end_id = args.audio_end_id
    if args.tokenizer_path:
        from audio_tokenization.utils.token_mapping import get_structure_tokens

        structure_tokens = get_structure_tokens(
            args.tokenizer_path,
            required=["audio_start", "audio_end"],
        )
        if audio_start_id is None:
            audio_start_id = int(structure_tokens["audio_start"])
        if audio_end_id is None:
            audio_end_id = int(structure_tokens["audio_end"])
    summary = compare_cutid_token_sets(
        args.left,
        args.right,
        recursive=args.recursive,
        trim_tolerance=args.trim_tolerance,
        audio_start_id=audio_start_id,
        audio_end_id=audio_end_id,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
