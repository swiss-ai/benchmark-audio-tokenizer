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


def parse_peoples_speech_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse People's Speech clip IDs.

    Format: ``{source_path_with_SLASH_DOT}_NNNNN.flac``
    e.g. ``forum_SLASH_foo_DOT_mp3_00002.flac`` -> ``("forum_SLASH_foo_DOT_mp3", 2)``
    """
    match = re.match(r"^(.+?)_(\d+)(?:\.\w+)?$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse People's Speech clip ID: {clip_id!r}")
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


def parse_legco_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse LegCo speech clip IDs.

    Format: ``{recording_id}_{clip_num}`` with optional dedup suffix.
    e.g. ``rIa-Qb8EYsA_123`` -> ``("rIa-Qb8EYsA", 123)``
         ``rIa-Qb8EYsA_123-0`` -> ``("rIa-Qb8EYsA", 123)``
    """
    match = re.match(r"^(.+?)_(\d+)(?:-\d+)?$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse LegCo clip ID: {clip_id!r}")
    source_id = match.group(1)
    clip_num = int(match.group(2))
    return source_id, clip_num


def parse_coral_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse CoRal conversation clip IDs.

    Format: ``{source_id}_{clip_num}`` with optional dedup suffix.
    e.g. ``conv_07f9708fc0b8316a9dea85d473db112b_00005``
      -> ``("conv_07f9708fc0b8316a9dea85d473db112b", 5)``
    """
    match = re.match(r"^(.+?)_(\d+)(?:-\d+)?$", clip_id)
    if match is None:
        raise ValueError(f"Cannot parse CoRal clip ID: {clip_id!r}")
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


def parse_generic_clip_id(clip_id: str) -> Tuple[str, int]:
    """Fallback parser: treats entire clip ID as source, clip_num=0."""
    return clip_id, 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PARSERS = {
    "emilia": parse_emilia_clip_id,
    "peoples_speech": parse_peoples_speech_clip_id,
    "wenetspeech": parse_wenetspeech_clip_id,
    "spc": parse_spc_clip_id,
    "aishell": parse_aishell_clip_id,
    "legco": parse_legco_clip_id,
    "coral": parse_coral_clip_id,
    "libriheavy": parse_libriheavy_clip_id,
    "parlaspeech": parse_parlaspeech_clip_id,
    "generic": parse_generic_clip_id,
}


def get_clip_id_parser(name: str):
    """Look up a clip ID parser by name.

    Args:
        name: One of ``"emilia"``, ``"peoples_speech"``, ``"wenetspeech"``,
            ``"spc"``, ``"aishell"``, ``"legco"``, ``"coral"``, ``"generic"``.

    Returns:
        Callable[[str], Tuple[str, int]]
    """
    if name not in _PARSERS:
        raise ValueError(
            f"Unknown clip_id_parser: {name!r}. "
            f"Available: {sorted(_PARSERS.keys())}"
        )
    return _PARSERS[name]
