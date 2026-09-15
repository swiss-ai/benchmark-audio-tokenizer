"""Validation shared by flat int32 token-cache readers."""

from __future__ import annotations

import numpy as np


def validate_int32_token_spans(
    offsets,
    counts,
    payload_bytes,
    *,
    allow_empty: bool,
    context: str,
) -> None:
    """Validate byte offsets and token counts before normalizing or reading them.

    ``payload_bytes`` is either one file size or one size per span. Checking
    remaining capacity instead of adding span ends avoids integer overflow.
    """
    offsets = np.asarray(offsets)
    counts = np.asarray(counts)
    payload_bytes = np.asarray(payload_bytes)
    if offsets.ndim != 1 or counts.shape != offsets.shape:
        raise ValueError(f"{context}: token offsets and counts must be matching 1-D arrays")
    if offsets.dtype.kind != "i" or counts.dtype.kind != "i":
        raise ValueError(f"{context}: token offsets and counts must be non-null signed integers")
    if payload_bytes.dtype.kind not in "iu" or (
        payload_bytes.ndim != 0 and payload_bytes.shape != offsets.shape
    ):
        raise ValueError(f"{context}: payload sizes must be integers matching the token spans")
    if np.any(payload_bytes < 0) or np.any(payload_bytes % 4 != 0):
        raise ValueError(f"{context}: payload size must be nonnegative and aligned to int32 bytes")
    if np.any(offsets < 0) or np.any(offsets % 4 != 0):
        raise ValueError(f"{context}: token offsets must be nonnegative and aligned to int32 bytes")
    if np.any(counts < (0 if allow_empty else 1)):
        requirement = "nonnegative" if allow_empty else "positive"
        raise ValueError(f"{context}: token counts must be {requirement}")
    if np.any(offsets > payload_bytes) or np.any(counts > (payload_bytes - offsets) // 4):
        raise ValueError(f"{context}: token span exceeds payload bounds")
