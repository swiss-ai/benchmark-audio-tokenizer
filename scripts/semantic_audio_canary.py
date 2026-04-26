#!/usr/bin/env python3
"""Run semantic ASR validation for tokenized audio canaries."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from audio_tokenization.validation.semantic_audio import main


if __name__ == "__main__":
    raise SystemExit(main())
