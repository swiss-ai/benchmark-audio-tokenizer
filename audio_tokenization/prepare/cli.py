"""Shared helpers for prepare runners.

After the prepare CLIs were retired in favor of Hydra (``python -m audio_tokenization run``),
this module's only public surface is ``expand_path_patterns``, used by the
``resolve()`` step of each family runner to turn user-authored file/glob lists
into a concrete file list.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable


def expand_path_patterns(patterns: Iterable[str]) -> list[str]:
    """Expand user-provided path patterns to a deduplicated, sorted list.

    Convert input file lists (``arrow_files``, ``wds_shards``, ``jsonl_files``)
    are user-authored; literal paths and glob patterns must both resolve here.

    Each pattern must resolve to at least one file: a typo'd or stale glob
    raises ``FileNotFoundError`` rather than silently producing a partial
    dataset. ``~`` is expanded so user-home paths in YAML resolve.
    """
    out: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(os.path.expanduser(pattern))
        if not matches:
            raise FileNotFoundError(f"No files match pattern: {pattern!r}")
        out.update(matches)
    return sorted(out)
