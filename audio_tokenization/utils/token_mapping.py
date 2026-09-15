"""Shared loader for audio_token_mapping.json (single source of truth)."""

from __future__ import annotations

import json
import os
from typing import Any


def load_audio_token_mapping(tokenizer_path: str) -> dict[str, Any]:
    """Load and validate audio_token_mapping.json from a tokenizer directory.

    Returns the full mapping dict (with ``structure_tokens``,
    ``audio_token_offset``, etc.).

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError``
    if ``structure_tokens`` is absent.
    """
    mapping_path = os.path.join(tokenizer_path, "audio_token_mapping.json")
    with open(mapping_path) as f:
        mapping = json.load(f)

    if "structure_tokens" not in mapping:
        raise ValueError(
            f"No 'structure_tokens' in {mapping_path}. "
            "Rebuild the tokenizer with the latest omnitok."
        )
    return mapping


def get_structure_tokens(
    tokenizer_path: str,
    required: list[str] | None = None,
) -> dict[str, int]:
    """Return the ``structure_tokens`` sub-dict, validating *required* keys.

    Convenience wrapper around :func:`load_audio_token_mapping`.
    """
    mapping = load_audio_token_mapping(tokenizer_path)
    st = mapping["structure_tokens"]
    for key in required or ():
        if key not in st:
            raise ValueError(
                f"'{key}' missing from structure_tokens in "
                f"{os.path.join(tokenizer_path, 'audio_token_mapping.json')}"
            )
    return st
