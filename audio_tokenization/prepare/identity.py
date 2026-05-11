"""Identity and clip numbering helpers for prepare scripts."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

INTERLEAVE_CUSTOM_KEY = "interleave"


def resolve_input_source_and_clip_num(
    raw_id: object,
    *,
    chunk_idx: int = 0,
    input_clip_id_parser: Callable[[str], tuple[str, int]] | None = None,
) -> tuple[str, int]:
    """Resolve the output ``(source_id, clip_num)`` for a prepared cut."""
    source_id = str(raw_id)
    if input_clip_id_parser is None:
        return source_id, int(chunk_idx)
    if chunk_idx != 0:
        raise ValueError(
            "input_clip_id_parser cannot be combined with chunked outputs; "
            "the input ID already encodes clip numbering."
        )
    return input_clip_id_parser(source_id)


def set_interleave_metadata(
    cut,
    source_id: str,
    clip_num: int,
    *,
    clip_start: float | None = None,
    clip_duration: float | None = None,
):
    """Store canonical interleaving metadata without mutating cut identity."""
    if clip_num < 0:
        raise ValueError(f"clip_num must be >= 0, got {clip_num}")
    if clip_duration is None and clip_start is not None:
        clip_duration = float(cut.duration)
    if clip_duration is not None and clip_duration <= 0:
        raise ValueError(f"clip_duration must be > 0 when set, got {clip_duration}")
    cut.custom = dict(cut.custom or {})
    cut.custom[INTERLEAVE_CUSTOM_KEY] = {
        "source_id": str(source_id),
        "clip_num": int(clip_num),
        "clip_start": float(clip_start) if clip_start is not None else None,
        "clip_duration": float(clip_duration) if clip_duration is not None else None,
    }
    return cut


def assign_interleave_metadata(
    cuts: list,
    store_clip_start: bool = True,
    max_gap_sec: float | None = None,
) -> list:
    """Assign canonical interleave metadata without rewriting cut IDs."""
    groups = defaultdict(list)
    for cut in cuts:
        groups[cut.recording_id].append(cut)

    result = []
    for rec_id, group in groups.items():
        group.sort(key=lambda c: (c.start, c.id))

        run_idx = 0
        clip_num = 0
        prev_end = None

        for cut in group:
            if max_gap_sec is not None and prev_end is not None:
                gap = cut.start - prev_end
                if gap > max_gap_sec:
                    run_idx += 1
                    clip_num = 0

            source_id = f"{rec_id}_R{run_idx}" if run_idx > 0 else rec_id
            set_interleave_metadata(
                cut,
                source_id,
                clip_num,
                # Timestamp metadata is canonicalized as start + duration so
                # materialize-time gap detection works across all prepare
                # families without storing redundant clip_end.
                clip_start=cut.start if store_clip_start else None,
                clip_duration=cut.duration if store_clip_start else None,
            )

            prev_end = cut.start + cut.duration
            clip_num += 1
            result.append(cut)
    return result
