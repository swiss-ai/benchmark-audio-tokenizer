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


def parse_trailing_number_basename_clip_id(clip_id: str) -> Tuple[str, int]:
    """Strip any leading directory prefix, then parse a trailing-number clip ID."""
    basename = clip_id.rsplit("/", 1)[-1]
    return parse_trailing_number_clip_id(basename)


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


def parse_eurospeech_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse EuroSpeech clip IDs.

    Format: ``{country}_{country}_{speaker}_{date}_{start_ms}_{end_ms}``
    e.g. ``bulgaria_bulgaria_0_01022012_50288_69767``
      -> ``("bulgaria_bulgaria_0_01022012", 50288)``

    Uses start_ms as clip_num for temporal ordering.
    """
    parts = clip_id.rsplit("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Cannot parse EuroSpeech clip ID: {clip_id!r}")
    source_id = parts[0]
    start_ms = int(parts[1])
    return source_id, start_ms


def parse_hui_audio_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse HUI Audio Corpus clip IDs.

    Format: ``{book_chapter}_f{NNNNNN}-{idx}``
    e.g. ``jane_eyre_die_waise_von_lowood_37_f000008-0``
      -> ``("jane_eyre_die_waise_von_lowood_37", 8)``
    """
    m = re.match(r"^(.+)_f(\d+)-\d+$", clip_id)
    if m is None:
        raise ValueError(f"Cannot parse HUI Audio clip ID: {clip_id!r}")
    return m.group(1), int(m.group(2))


def parse_trailing_dash_number_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse clip IDs with trailing dash-separated number.

    Format: ``{source_id}-{clip_num}``
    e.g. ``BAC009S0195W0359-0``     -> ``("BAC009S0195W0359", 0)``
         ``SSB00050001-0``          -> ``("SSB00050001", 0)``
         ``session-SPK0001-129``    -> ``("session-SPK0001", 129)``
    """
    last_dash = clip_id.rfind("-")
    if last_dash < 0:
        raise ValueError(f"Cannot parse clip ID (expected trailing -NUMBER): {clip_id!r}")
    source_id = clip_id[:last_dash]
    clip_num = int(clip_id[last_dash + 1:])
    return source_id, clip_num


def parse_f1_radio_clip_id(clip_id: str) -> Tuple[str, int]:
    """Parse F1 team radio clip IDs.

    Format: ``{grand_prix}_{driver_id}_{racing_num}_{YYYYMMDD}_{HHMMSS}``
    e.g. ``2018_Australian_Grand_Prix_BREHAR01_28_20180325_161424``
      -> ``("2018_Australian_Grand_Prix_BREHAR01", 20180325161424)``

    Uses datetime as clip_num for temporal ordering.
    """
    # Split from the right: last two parts are YYYYMMDD and HHMMSS
    parts = clip_id.rsplit("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Cannot parse F1 radio clip ID: {clip_id!r}")
    # parts[0] = everything before racing_num, parts[1] = YYYYMMDD, parts[2] = HHMMSS
    # But racing_num is also before YYYYMMDD. Split more carefully.
    # ID: 2018_Australian_Grand_Prix_BREHAR01_28_20180325_161424
    # Last 2 underscored parts: 20180325_161424
    timestamp = int(parts[1] + parts[2])  # 20180325161424
    # source_id: everything before racing_num_YYYYMMDD_HHMMSS
    # The racing_num is the part just before YYYYMMDD
    prefix_parts = parts[0].rsplit("_", 1)
    source_id = prefix_parts[0]  # grand_prix_driver_id
    return source_id, timestamp


# parse_vimedcss_clip_id is identical to parse_trailing_dash_number_clip_id
parse_vimedcss_clip_id = parse_trailing_dash_number_clip_id


def parse_generic_clip_id(clip_id: str) -> Tuple[str, int]:
    """Fallback parser: treats entire clip ID as source, clip_num=0."""
    return clip_id, 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PARSERS = {
    "universal": parse_universal_clip_id,
    "trailing_number": parse_trailing_number_clip_id,
    "trailing_number_basename": parse_trailing_number_basename_clip_id,
    "voxpopuli": parse_voxpopuli_clip_id,
    "ytc": parse_ytc_clip_id,
    "emilia": parse_emilia_clip_id,
    "wenetspeech": parse_wenetspeech_clip_id,
    "spc": parse_spc_clip_id,
    "aishell": parse_aishell_clip_id,
    "libriheavy": parse_libriheavy_clip_id,
    "parlaspeech": parse_parlaspeech_clip_id,
    "eurospeech": parse_eurospeech_clip_id,
    "hui_audio": parse_hui_audio_clip_id,
    "trailing_dash_number": parse_trailing_dash_number_clip_id,
    "f1_radio": parse_f1_radio_clip_id,
    "vimedcss": parse_vimedcss_clip_id,
    "generic": parse_generic_clip_id,
}


# Clip-num to seconds conversion for parsers that use non-second units.
# Parsers not listed here have clip_num in arbitrary sequential units (no time meaning).
_CLIP_NUM_TO_SEC = {
    "ytc": 0.001,          # clip_num is start_ms
    "eurospeech": 0.001,   # clip_num is start_ms
    "parlaspeech": 0.1,    # clip_num is start_deciseconds
}


def available_clip_id_parsers() -> Tuple[str, ...]:
    """Return the registered clip-ID parser names in stable order."""
    return tuple(sorted(_PARSERS))


def get_clip_num_to_sec(name: str) -> float | None:
    """Return the scale factor to convert clip_num to seconds, or None if not applicable."""
    return _CLIP_NUM_TO_SEC.get(name)


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
