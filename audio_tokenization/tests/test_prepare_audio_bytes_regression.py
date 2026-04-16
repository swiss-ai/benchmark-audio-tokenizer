from audio_tokenization.prepare import common
from audio_tokenization.prepare import prepare_hf_to_shar
from audio_tokenization.prepare import prepare_parquet_to_shar
from audio_tokenization.prepare import prepare_wds_to_shar


def test_byte_based_prepare_modules_share_common_recording_builder():
    assert (
        prepare_hf_to_shar.build_recording_from_audio_bytes
        is common.build_recording_from_audio_bytes
    )
    assert (
        prepare_parquet_to_shar.build_recording_from_audio_bytes
        is common.build_recording_from_audio_bytes
    )
    assert (
        prepare_wds_to_shar.build_recording_from_audio_bytes
        is common.build_recording_from_audio_bytes
    )
