"""Identity and clip numbering helpers for prepare_data."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from audio_tokenization.utils.clip_id_parsers import available_clip_id_parsers


def add_input_clip_id_parser_arg(parser) -> None:
    """Add a CLI option for parsing legacy input clip IDs at prepare time."""
    parser.add_argument(
        "--input-clip-id-parser",
        type=str,
        choices=available_clip_id_parsers(),
        default=None,
        help=(
            "Parse incoming sample IDs into (source_id, clip_num) before writing "
            "universal IDs. If unset, direct inputs become {raw_id}@000000 and "
            "chunked inputs use dense chunk numbering."
        ),
    )


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


def set_universal_cut_id(
    cut,
    source_id: str,
    clip_num: int,
    *,
    clip_start: float | None = None,
):
    """Rewrite ``cut.id`` and store canonical interleaving metadata."""
    if clip_num < 0:
        raise ValueError(f"clip_num must be >= 0, got {clip_num}")
    legacy_cut_id = getattr(cut, "id", None)
    cut.id = f"{source_id}@{clip_num:06d}"
    for supervision in getattr(cut, "supervisions", ()) or ():
        supervision.id = cut.id
    cut.custom = dict(cut.custom or {})
    cut.custom["source_id"] = source_id
    cut.custom["clip_num"] = int(clip_num)
    if clip_start is not None:
        cut.custom["clip_start"] = float(clip_start)
    if legacy_cut_id is not None and legacy_cut_id != cut.id:
        cut.custom["legacy_cut_id"] = legacy_cut_id
    return cut


def assign_universal_ids(
    cuts: list,
    store_clip_start: bool = True,
    max_gap_sec: float | None = None,
) -> list:
    """Rewrite cut IDs to the universal format: ``{recording_id}@{clip_num:06d}``."""
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
            set_universal_cut_id(
                cut,
                source_id,
                clip_num,
                clip_start=cut.start if store_clip_start else None,
            )

            prev_end = cut.start + cut.duration
            clip_num += 1
            result.append(cut)
    return result

