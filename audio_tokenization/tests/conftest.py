import sys
from pathlib import Path

import pytest

# scripts/ holds CLI entrypoints that aren't a package; make them importable
# by tests without per-file sys.path hacks.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@pytest.fixture
def token_ids():
    return dict(bos_id=1, eos_id=2, stt_continue_id=99, stt_transcribe_id=98, tts_continue_id=97)


# Shared by test_build_interleaved* tests that exercise the v2 structured-cache
# format. The 11-key row dict is the StructuredCacheChunkWriter input contract;
# centralizing it here keeps writer-side schema additions from rotting tests.
_V2_CACHE_ROW_TEMPLATE = {
    "speaker": "",
    "duration": 1.0,
    "clip_duration": None,
    "dataset": "ds",
}


def make_v2_cache_row(
    *,
    clip_num: int,
    audio_tokens: list[int],
    text_tokens: list[int],
    source_id: str = "s1",
    text: str | None = None,
    clip_start: float | None = None,
    **overrides,
) -> dict:
    """Build one StructuredCacheChunkWriter row with sensible defaults.

    ``text`` defaults to ``chr(ord('a') + clip_num)`` so each row is
    distinguishable when grepping test failures.
    ``clip_start`` defaults to ``float(clip_num)`` so monotonically-increasing
    timestamps fall out of monotonically-increasing clip_num.
    """
    if text is None:
        text = chr(ord("a") + clip_num) if 0 <= clip_num < 26 else f"c{clip_num}"
    if clip_start is None:
        clip_start = float(clip_num)
    row = {
        **_V2_CACHE_ROW_TEMPLATE,
        "clip_id": f"{source_id}@{clip_num:06d}",
        "source_id": source_id,
        "clip_num": clip_num,
        "clip_start": clip_start,
        "text": text,
        "text_tokens": list(text_tokens),
        "audio_tokens": list(audio_tokens),
    }
    row.update(overrides)
    return row


def make_v2_cache_rows(
    n: int,
    *,
    source_id: str = "s1",
    audio_token_base: int = 10,
    text_token_base: int = 20,
) -> list[dict]:
    """Build *n* sequential v2-cache rows with single-token audio/text payloads.

    The default token IDs are ``[10..10+n)`` for audio and ``[20..20+n)`` for
    text — distinct ranges so test assertions can verify cross-modality
    routing without confusion.
    """
    return [
        make_v2_cache_row(
            clip_num=i,
            audio_tokens=[audio_token_base + i],
            text_tokens=[text_token_base + i],
            source_id=source_id,
        )
        for i in range(n)
    ]


@pytest.fixture
def reset_interleave_globals():
    """Yield a callable that, when called with an interleave module, zeros its
    ``_shared_*`` module-level globals on test teardown.

    Both ``audio_tokenization.interleave.shift_by_one`` and
    ``audio_tokenization.interleave.greedy`` use a fork-COW shared-state
    pattern (``_shared_cache``, ``_shared_run_starts``, ``_shared_run_lengths``,
    ``_shared_transcribe_only_runs``). Forgetting to reset these between tests
    leaks state across runs and produces hard-to-debug failures.
    """
    registered: list = []

    def _register(module) -> None:
        registered.append(module)

    yield _register

    for module in registered:
        module._shared_cache = None
        module._shared_run_starts = None
        module._shared_run_lengths = None
        module._shared_transcribe_only_runs = set()
