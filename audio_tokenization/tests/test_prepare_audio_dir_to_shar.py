import gzip
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_tokenization.prepare import prepare_audio_dir_to_shar


def _write_wav(path: Path, samples: np.ndarray, sampling_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), sampling_rate)


def _write_vad_jsonl(
    path: Path,
    *,
    key: str = "clip",
    duration_sec: float = 12.0,
    sample_rate: int = 16000,
    lang: str = "en",
) -> None:
    path.write_text(
        json.dumps(
            {
                key: {
                    "timestamps": [
                        {
                            "start": 0,
                            "end": int(duration_sec * sample_rate),
                        }
                    ],
                    "duration_sec": duration_sec,
                    "sample_rate": sample_rate,
                    "lang": lang,
                }
            }
        )
        + "\n"
    )


def _worker_args(tmp_path: Path, jsonl_path: Path, audio_path: Path):
    return (
        0,
        [str(jsonl_path)],
        {"clip": audio_path},
        str(tmp_path / "shar"),
        None,
        100,
        "wav",
        16000,
        True,
        200.0,
        5.0,
        16000,
        1.0,
        None,
        ".wav",
        "default",
    )


def _read_first_written_cut(tmp_path: Path) -> dict:
    cuts_path = next((tmp_path / "shar" / "worker_00").glob("cuts.*.jsonl.gz"))
    with gzip.open(cuts_path, "rt") as f:
        return json.loads(next(f))


def test_audio_dir_worker_records_rms_db_via_shared_audio_pipeline(tmp_path):
    sample_rate = 16000
    duration_sec = 12.0
    audio_path = tmp_path / "clip.wav"
    jsonl_path = tmp_path / "vad.jsonl"
    samples = np.full(int(sample_rate * duration_sec), 0.1, dtype=np.float32)
    _write_wav(audio_path, samples, sample_rate)
    _write_vad_jsonl(jsonl_path, duration_sec=duration_sec, sample_rate=sample_rate)

    result = prepare_audio_dir_to_shar._convert_worker(
        _worker_args(tmp_path, jsonl_path, audio_path),
    )

    cut = _read_first_written_cut(tmp_path)
    custom = cut["custom"]
    assert result["written"] == 1
    assert result["skipped"] == 0
    assert result["worker_stats"]["runtime_counts"]["reused_decoded_audio_for_shar_write"] == 1
    assert -21.0 < custom["rms_db"] < -19.0
    assert custom["global_offset_sec"] == 0.0
    assert custom["lang"] == "en"
    assert custom["interleave"] == {
        "source_id": "clip",
        "clip_num": 0,
        "clip_start": 0.0,
        "clip_duration": 12.0,
    }


def test_audio_dir_worker_skips_quiet_chunks_before_write(tmp_path):
    sample_rate = 16000
    duration_sec = 12.0
    audio_path = tmp_path / "clip.wav"
    jsonl_path = tmp_path / "vad.jsonl"
    samples = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
    _write_wav(audio_path, samples, sample_rate)
    _write_vad_jsonl(jsonl_path, duration_sec=duration_sec, sample_rate=sample_rate)

    result = prepare_audio_dir_to_shar._convert_worker(
        _worker_args(tmp_path, jsonl_path, audio_path),
    )

    worker_dir = tmp_path / "shar" / "worker_00"
    assert result["written"] == 0
    assert result["skipped"] == 1
    assert result["worker_stats"]["runtime_counts"]["skipped_quiet_audio"] == 1
    assert not list(worker_dir.glob("cuts.*.jsonl.gz"))
