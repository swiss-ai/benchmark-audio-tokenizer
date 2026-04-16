#!/usr/bin/env python3
"""DEPRECATED — manifest-only SHAR patching is unsound; do not use.

This script previously rewrote ``cut.id`` and ``recording.id`` in
``cuts.*.jsonl.gz`` while symlinking the original ``recording.*.tar`` files.
Lhotse's SHAR reader asserts that every tar member's stem equals the
corresponding ``cut.id`` and iterates cuts and tar entries in lockstep, so:

  * rewriting IDs without rebuilding the tars trips the stem-mismatch
    assertion at ``lhotse/shar/readers/lazy.py:289`` on the first cut, and
  * dropping cuts (quiet/no-text) without rewriting the tars desynchronises
    the lockstep ``zip`` and silently corrupts every subsequent cut↔audio pair.

Both failure modes make the output SHAR unreadable. The script's previous
implementation has been removed.

The supported path is to set canonical IDs at *prepare* time, not by
post-processing. For the parquet family, use ``--id-column`` (or a list of
columns for composite IDs) and ``--input-clip-id-parser`` on
``prepare_parquet_to_shar``. Then rebuild affected outputs from the raw
source. Migrated examples for reference:

  * ``infore2_audiobooks_train_fixed_ids_v4`` — parquet rebuild with
    ``--id-column audio.path --input-clip-id-parser trailing_number_basename``
  * ``aozora_hurigana_train`` — identity preserved upstream in the parquet
    conversion step (``convert_to_parquet.py``).

To confirm an existing SHAR is consumable, run:

    python -m audio_tokenization.prepare.validate_shar \\
        --shar-dir <path>
"""

from __future__ import annotations

import sys


_DEPRECATION_MESSAGE = (
    "patch_universal_ids.py is deprecated and removed: rewriting cut IDs without "
    "rebuilding recording tars produces unreadable SHARs (Lhotse stem-mismatch "
    "assertion). Set canonical IDs at prepare time via --id-column / "
    "--input-clip-id-parser and rebuild the SHAR from raw, then verify with: "
    "python -m audio_tokenization.prepare.validate_shar "
    "--shar-dir <path>. See module docstring for details."
)


def main() -> None:
    print(_DEPRECATION_MESSAGE, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
