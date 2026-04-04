"""Registry of per-dataset clip ID parsers for extracting (source_id, clip_num).

Used by the audio_text_interleaving pipeline to group clips from the same
source recording and sort them in order.

Examples:
    Emilia:  ``EN_tKvmUvxYZXI_W000006`` -> ``("EN_tKvmUvxYZXI", 6)``
    People's Speech: ``...forum_SLASH_..._00002.flac`` -> ``("..._DOT_mp3", 2)``
"""

import re
from typing import Tuple


def parse_emilia_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse Emilia-style clip IDs.

    Format: ``{lang}_{youtube_id}_W{clip_num:06d}``
    e.g. ``EN_tKvmUvxYZXI_W000006`` -> ``("EN_tKvmUvxYZXI", 6)``
    """
    match = re.match(r"^(.+)_W(\d+)$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse Emilia clip ID: {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_trailing_number_clip_id(clip_id: str) -> Tuple[str, int]:
    """Generic parser: split on the last ``_DIGITS`` with optional file extension.

    Works for any ID format ending in ``_{number}`` or ``_{number}.ext``:
        ``forum_SLASH_foo_DOT_mp3_00002.flac`` -> ``("forum_SLASH_foo_DOT_mp3", 2)``
        ``187_003_0011``                       -> ``("187_003", 11)``
        ``rIa-Qb8EYsA_123-0``                 -> ``("rIa-Qb8EYsA", 123)``
        ``conv_07f9708fc0b8_00005``            -> ``("conv_07f9708fc0b8", 5)``
    """
    match = re.match(r"^(.+?)_(\d+)(?:-\d+)?(?:\.\w+)?$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse clip ID (expected trailing _NUMBER): {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_wenetspeech_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse WenetSpeech clip IDs.

    Format: ``{split}_{recording_id}_S{clip_num:05d}``
    e.g. ``L_T0000005699_S00003`` -> ``("L_T0000005699", 3)``
         ``DEV_T0000005699_S00000`` -> ``("DEV_T0000005699", 0)``
    """
    match = re.match(r"^(.+)_S(\d+)$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse WenetSpeech clip ID: {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_spc_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse SPC (Speech Parliament Corpus) segmented clip IDs.

    Format: ``row{NNNNN}_seg{NNN}``
    e.g. ``row00000_seg003`` -> ``("row00000", 3)``
    """
    match = re.match(r"^(.+)_seg(\d+)$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse SPC clip ID: {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_aishell_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse AISHELL-1 clip IDs.

    Format: ``{prefix}{speaker_id}W{utterance_num}``
    e.g. ``BAC009S0002W0122`` -> ``("BAC009S0002", 122)``
    """
    match = re.match(r"^(.+)W(\d+)$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse AISHELL clip ID: {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_libriheavy_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse LibriHeavy clip IDs.

    Format: ``large/{speaker_id}/{book_chapter_mp3}/{chapter}_{segment_index}``
    e.g. ``large/10018/conquestofcanaan_1710_librivox_64kb_mp3/conquestofcanaan_01_tarkington_64kb_5``
      -> ``("large/10018/conquestofcanaan_1710_librivox_64kb_mp3/conquestofcanaan_01_tarkington_64kb", 5)``
    """
    match = re.match(r"^(.+)_(\d+)$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse LibriHeavy clip ID: {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_parlaspeech_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse ParlaSpeech clip IDs.

    Format: ``{session}.{utterance_id}_{start}-{end}``
    e.g. ``ParlaMint-RS_2013-07-09-0.u20685_112-143``
      -> ``("ParlaMint-RS_2013-07-09-0.u20685", 112)``
    """
    match = re.match(r"^(.+)_(\d+)-\d+$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse ParlaSpeech clip ID: {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_voxpopuli_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse VoxPopuli clip IDs.

    Format: ``{session}_{lang}_{clip_num}``
    e.g. ``20160118-0900-PLENARY-12_fi_3`` -> ``("20160118-0900-PLENARY-12_fi", 3)``
    """
    last_underscore = clip_id.rfind("_")
    if last_underscore < 0:
        raise ValueError(f"Cannot parse VoxPopuli clip ID: {clip_id!r}")
    source_id = clip_id[:last_underscore]
    clip_num = int(clip_id[last_underscore + 1:])
    return source_id, clip_num


def parse_ytc_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse YTC (YouTube Captions) clip IDs.

    Format: ``{video_id}-{start_ms}-{duration_ms}``
    e.g. ``WydnCJflnNU-189904-96`` -> ``("WydnCJflnNU", 189904)``

    Uses start_ms as clip_num for temporal ordering.
    """
    parts = clip_id.rsplit("-", 2)
    if len(parts) < 3:
        raise ValueError(f"Cannot parse YTC clip ID: {clip_id!r}")
    video_id = parts[0]
    start_ms = int(parts[1])
    return video_id, start_ms


def parse_universal_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse the universal clip ID format: ``{source_id}@{clip_num:06d}``.

    This is the standard format for all datasets. The ``@`` separator is
    chosen because it does not appear in any known source ID.

    e.g. ``EN_tKvmUvxYZXI@000006`` -> ``("EN_tKvmUvxYZXI", 6)``
         ``DIY_-3ywrgCA-1I@000042`` -> ``("DIY_-3ywrgCA-1I", 42)``
    """
    at_pos = clip_id.rfind("@")
    if at_pos < 0:
        raise ValueError(
            f"Universal clip ID missing '@' separator: {clip_id!r}. "
            "Expected format: {{source_id}}@{{clip_num:06d}}"
        )
    return clip_id[:at_pos], int(clip_id[at_pos + 1:])


def parse_generic_clip_id(clip_id: str) -> Tuple[str, int]:
    """Fallback parser: treats entire clip ID as source, clip_num=0."""
    return clip_id, 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PARSERS = {
    "universal": parse_universal_clip_id,
    "trailing_number": parse_trailing_number_clip_id,
    "voxpopuli": parse_voxpopuli_clip_id,
    "ytc": parse_ytc_clip_id,
    "emilia": parse_emilia_clip_id,
    "wenetspeech": parse_wenetspeech_clip_id,
    "spc": parse_spc_clip_id,
    "aishell": parse_aishell_clip_id,
    "libriheavy": parse_libriheavy_clip_id,
    "parlaspeech": parse_parlaspeech_clip_id,
    "generic": parse_generic_clip_id,
}


def available_clip_id_parsers() -> Tuple[str, ...]:
    """Return the registered clip-ID parser names in stable order."""
    return tuple(sorted(_PARSERS))


def get_clip_id_parser(name: str):
    """Look up a clip ID parser by name.

    Args:
        name: Parser name. ``"universal"`` is the standard format for all
            new datasets. Prepare-time input parsers are kept only for
            datasets whose raw IDs still need normalization.

    Returns:
        Callable[[str], Tuple[str, int]]
    """
    if name not in _PARSERS:
        raise ValueError(
            f"Unknown clip_id_parser: {name!r}. "
            f"Available: {list(available_clip_id_parsers())}"
        )
    return _PARSERS[name]
