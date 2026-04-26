#!/usr/bin/env python3
"""Compare two Megatron tokenization outputs by cut_id -> token sequence.

Use ``--trim-tolerance`` with audio markers for 1-GPU vs multi-GPU canaries:
the cut set and safe audio prefix must match, while padding-trim tail drift is
bounded.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio_tokenization.utils.indexed_dataset.cut_id_sidecar import main


if __name__ == "__main__":
    raise SystemExit(main())
