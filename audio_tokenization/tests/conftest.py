import pytest


@pytest.fixture
def token_ids():
    return dict(bos_id=1, eos_id=2, stt_continue_id=99, stt_transcribe_id=98, tts_continue_id=97)
