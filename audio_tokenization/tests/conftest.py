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
