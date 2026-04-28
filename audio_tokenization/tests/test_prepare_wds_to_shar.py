import io
import sys
import tarfile
import types
from collections import Counter

import audio_tokenization.prepare.prepare_wds_to_shar as prepare_wds_to_shar
from audio_tokenization.config.schema import PrepareSpec
from audio_tokenization.prepare.prepare_wds_to_shar import (
    SidecarMetadataProvider,
    TarScanResult,
    _convert_worker,
    iter_tar_cuts,
)


class _FakeMember:
    def __init__(self, name: str):
        self.name = name

    def isfile(self) -> bool:
        return True


class _FakeSupervisionSegment:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeCut:
    def __init__(self, cut_id: str, recording_id: str, duration: float = 1.25):
        self.id = cut_id
        self.recording_id = recording_id
        self.duration = duration
        self.custom = None
        self.supervisions = []


class _FakeRecording:
    def __init__(self, cut: _FakeCut):
        self._cut = cut

    def to_cut(self):
        return self._cut


class _FakeSharWriter:
    def __init__(self, *, sink, **kwargs):
        self._sink = sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def write(self, cut):
        self._sink.append(cut)


class _TruncatedTar:
    def __iter__(self):
        yield _FakeMember("clip.wav")
        raise tarfile.ReadError("truncated tar")

    def extractfile(self, member):
        return io.BytesIO(b"payload")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ReadableTar:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads

    def extractfile(self, member):
        return io.BytesIO(self.payloads[member.name])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_sidecar_scan_marks_partial_tar_incomplete():
    provider = SidecarMetadataProvider()

    scan = provider.scan_tar(_TruncatedTar(), stats=Counter())

    assert not scan.scan_complete
    assert scan.audio_members == []
    assert scan.scan_error == "truncated tar"


def test_iter_tar_cuts_skips_incomplete_scan(monkeypatch):
    fake_lhotse = types.ModuleType("lhotse")

    class _Recording:
        @staticmethod
        def from_bytes(*args, **kwargs):
            raise AssertionError("corrupt tar should be skipped before audio decode")

    class _SupervisionSegment:
        def __init__(self, *args, **kwargs):
            pass

    fake_lhotse.Recording = _Recording
    fake_lhotse.SupervisionSegment = _SupervisionSegment
    monkeypatch.setitem(sys.modules, "lhotse", fake_lhotse)

    class _Provider:
        def scan_tar(self, tf, stats=None):
            return TarScanResult(
                audio_members=[_FakeMember("clip.wav")],
                scan_complete=False,
                scan_error="truncated tar",
            )

        def lookup(self, stem, scan, stats=None):
            return None, {}

    monkeypatch.setattr(tarfile, "open", lambda _: _TruncatedTar())

    stats = Counter()
    cuts = list(iter_tar_cuts(["broken.tar"], provider=_Provider(), stats=stats))

    assert cuts == []
    assert stats["skipped_corrupt_tar"] == 1


def test_iter_tar_cuts_builds_recording_via_shared_helper(monkeypatch):
    fake_lhotse = types.ModuleType("lhotse")
    fake_lhotse.SupervisionSegment = _FakeSupervisionSegment
    monkeypatch.setitem(sys.modules, "lhotse", fake_lhotse)

    calls = {}

    def fake_build_recording(audio_bytes, recording_id, *, runtime_counts=None):
        calls["audio_bytes"] = audio_bytes
        calls["recording_id"] = recording_id
        if runtime_counts is not None:
            runtime_counts["recording_from_bytes"] += 1
        return _FakeRecording(_FakeCut(cut_id=recording_id, recording_id=recording_id))

    class _Provider:
        def scan_tar(self, tf, stats=None):
            return TarScanResult(audio_members=[_FakeMember("nested/clip.wav")])

        def lookup(self, stem, scan, stats=None):
            return "hello world", {"speaker": "narrator"}

    monkeypatch.setattr(
        tarfile,
        "open",
        lambda _: _ReadableTar({"nested/clip.wav": b"encoded-audio"}),
    )
    monkeypatch.setattr(
        prepare_wds_to_shar,
        "build_recording_from_audio_bytes",
        fake_build_recording,
    )

    stats = Counter()
    cuts = list(iter_tar_cuts(["ok.tar"], provider=_Provider(), stats=stats))

    assert len(cuts) == 1
    assert calls == {
        "audio_bytes": b"encoded-audio",
        "recording_id": "nested/clip",
    }
    assert cuts[0].supervisions[0].text == "hello world"
    assert cuts[0].custom == {"speaker": "narrator"}
    assert stats["recording_from_bytes"] == 1
    assert stats["cuts_yielded"] == 1


def test_iter_tar_cuts_prefilters_keep_ids_before_decode(monkeypatch):
    fake_lhotse = types.ModuleType("lhotse")
    fake_lhotse.SupervisionSegment = _FakeSupervisionSegment
    monkeypatch.setitem(sys.modules, "lhotse", fake_lhotse)

    class _Provider:
        def scan_tar(self, tf, stats=None):
            return TarScanResult(audio_members=[_FakeMember("nested/clip.wav")])

        def lookup(self, stem, scan, stats=None):
            return None, {}

    monkeypatch.setattr(
        tarfile,
        "open",
        lambda _: _ReadableTar({"nested/clip.wav": b"encoded-audio"}),
    )
    monkeypatch.setattr(
        prepare_wds_to_shar,
        "canonical_sample_key",
        lambda stem: "skip-me",
    )
    monkeypatch.setattr(
        prepare_wds_to_shar,
        "build_recording_from_audio_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("keep_ids prefilter should skip decode")
        ),
    )

    stats = Counter()
    cuts = list(
        iter_tar_cuts(
            ["ok.tar"],
            provider=_Provider(),
            stats=stats,
            keep_ids={"wanted"},
        )
    )

    assert cuts == []
    assert stats["skipped_no_match"] == 1


def test_convert_worker_preserves_presegmented_clip_number_with_parser(monkeypatch, tmp_path):
    written_cuts = []
    fake_lhotse_shar = types.ModuleType("lhotse.shar")
    fake_lhotse_shar.SharWriter = lambda **kwargs: _FakeSharWriter(
        sink=written_cuts,
        **kwargs,
    )
    monkeypatch.setitem(sys.modules, "lhotse.shar", fake_lhotse_shar)
    monkeypatch.setattr(prepare_wds_to_shar, "check_worker_reuse", lambda *args, **kwargs: None)
    monkeypatch.setattr(prepare_wds_to_shar, "init_worker_process", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        prepare_wds_to_shar,
        "apply_audio_pipeline",
        lambda cut, **kwargs: (cut, False, None),
    )
    monkeypatch.setattr(
        prepare_wds_to_shar,
        "write_worker_result",
        lambda **kwargs: {
            "written": kwargs["written"],
            "skipped": kwargs["skipped"],
            "errors": kwargs["errors"],
            "worker_stats": {"runtime_counts": dict(kwargs["runtime_counts"])},
        },
    )

    cut = _FakeCut(
        cut_id="conv_07f9708fc0b8316a9dea85d473db112b_00005",
        recording_id="conv_07f9708fc0b8316a9dea85d473db112b_00005",
    )
    cut.sampling_rate = 16000
    cut.supervisions = [
        _FakeSupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.0,
            duration=cut.duration,
            text="hi",
        )
    ]
    monkeypatch.setattr(
        prepare_wds_to_shar,
        "iter_tar_cuts",
        lambda *args, **kwargs: iter([cut]),
    )

    result = _convert_worker(
        (
            0,
            ["dataset.tar"],
            str(tmp_path / "shar"),
            None,
            100,
            "flac",
            None,
            None,
            None,
            False,
            None,
            200.0,
            10.0,
            16000,
            0.5,
            None,
            None,
            None,
            None,
            "trailing_number",
            None,
        )
    )

    assert result["written"] == 1
    assert written_cuts[0].id == "conv_07f9708fc0b8316a9dea85d473db112b_00005"
    assert written_cuts[0].supervisions[0].id == written_cuts[0].id
    assert written_cuts[0].custom == {
        "interleave": {
            "source_id": "conv_07f9708fc0b8316a9dea85d473db112b",
            "clip_num": 5,
            "clip_start": 0.0,
            "clip_duration": 1.25,
        },
    }


def test_run_uses_configured_mp_start_method_without_external_metadata(monkeypatch, tmp_path):
    captured = {}
    spec = PrepareSpec.from_mapping(
        {
            "family": "wds",
            "input": {"wds_shards": ["/data/*.tar"]},
            "output": {
                "shar_dir": str(tmp_path / "shar"),
                "shard_size": 100,
                "text_tokenizer": None,
                "mp_start_method": "fork",
            },
        }
    )

    monkeypatch.setattr(prepare_wds_to_shar, "expand_path_patterns", lambda _patterns: ["/data/a.tar"])
    monkeypatch.setattr(prepare_wds_to_shar, "validate_prepare_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(prepare_wds_to_shar, "write_prepare_state_for_spec", lambda _spec: None)
    monkeypatch.setattr(
        prepare_wds_to_shar,
        "ensure_worker_assignment",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(prepare_wds_to_shar, "load_text_tokenizer", lambda _path: None)

    def fake_run_pool(_worker, _worker_args, _shar_dir, _num_workers, *, mp_start_method):
        captured["mp_start_method"] = mp_start_method

    monkeypatch.setattr(prepare_wds_to_shar, "run_pool_and_finalize", fake_run_pool)

    prepare_wds_to_shar.run(spec)

    assert captured["mp_start_method"] == "fork"
