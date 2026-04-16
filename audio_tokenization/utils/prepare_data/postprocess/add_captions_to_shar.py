#!/usr/bin/env python3
"""DEPRECATED — manifest-only SHAR patching is unsound; do not use.

This script previously merged captions into existing SHAR cuts and rewrote
``cut.id`` to a "universal" format while leaving ``recording.*.tar``
untouched. Lhotse's SHAR reader asserts that every tar member's stem equals
the corresponding ``cut.id`` and iterates cuts and tar entries in lockstep,
so rewriting IDs without rebuilding the tars trips the stem-mismatch
assertion at ``lhotse/shar/readers/lazy.py:289`` on the first cut. The
output SHAR is unreadable.

The script's previous implementation has been removed.

The supported path is to attach captions and set canonical IDs at *prepare*
time. For new datasets, build the cut + caption pipeline so the captions
land as supervisions during ``prepare_*_to_shar`` and the cut IDs are
canonical from the start. To verify an existing SHAR is consumable:

    python -m audio_tokenization.utils.prepare_data.validate_shar \\
        --shar-dir <path>
"""

from __future__ import annotations

import sys


_DEPRECATION_MESSAGE = (
    "add_captions_to_shar.py is deprecated and removed: rewriting cut IDs without "
    "rebuilding recording tars produces unreadable SHARs (Lhotse stem-mismatch "
    "assertion). Attach captions and set canonical IDs at prepare time, then "
    "verify with: python -m audio_tokenization.utils.prepare_data.validate_shar "
    "--shar-dir <path>. See module docstring for details."
)


def main() -> None:
    print(_DEPRECATION_MESSAGE, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
