"""Exercise failure boundaries with real writers and CPU inference fixtures."""

import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from audio_tokenization.config.schema import TokenizeSpec
from audio_tokenization.pipelines.lhotse import core
from audio_tokenization.pipelines.lhotse._unsupervised_batch import tokenize_unsupervised_batch
from audio_tokenization.pipelines.lhotse.audio_only import AudioOnlyHandler
from audio_tokenization.pipelines.lhotse.audio_cache import AudioCacheHandler
from audio_tokenization.pipelines.lhotse.audio_text import AudioTextHandler
from audio_tokenization.pipelines.lhotse.stats_reducer import wait_for_rank_stats, write_rank_stats
from audio_tokenization.stages import tokenize as stage
from audio_tokenization.stages._stage_runner import run_stage


def _spec(mode="audio_only"):
    return TokenizeSpec.model_validate({
        "mode": mode,
        "tokenizer": {"path": "/unused/tokenizer"},
        "output": {"output_dir": "/unused/output"},
        "dataloader": {"num_workers": 0, "num_buckets": 1},
        "partitioning": {"type": "hash", "field": "source_id", "num_buckets": 1},
    })


def _batch(cut_id, token=11, *, supervised=False):
    cut = SimpleNamespace(
        id=cut_id, num_samples=1, duration=1 / 24000, supervisions=[],
        custom={"interleave": {"source_id": "source", "clip_num": token}, "text_tokens": [1]},
    )
    audio = torch.tensor([[token]], dtype=torch.float32)
    if supervised:
        return {"inputs": audio, "supervisions": {"cut": [cut]}}
    return {"audio": audio, "audio_lens": torch.tensor([1]), "cuts": [cut]}


class _CpuAudioOnly(AudioOnlyHandler):
    def create_dataset(self):
        return object()

    def process_batch(self, batch, tokenizer, stats, sr, device):
        return super().process_batch(batch, tokenizer, stats, sr, "cpu")


class _CpuAudioText(AudioTextHandler):
    def create_dataset(self):
        return object()

    def process_batch(self, batch, tokenizer, stats, sr, device):
        return super().process_batch(batch, tokenizer, stats, sr, "cpu")


def _run(monkeypatch, output, handler, batches, tokenizer, spec):
    module = ModuleType("audio_tokenization.vokenizers")
    module.create_tokenizer = lambda **kw: tokenizer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(core, "build_cutset", lambda *a, **kw: list(range(len(batches))))
    monkeypatch.setattr(core, "_create_rank_sampler", lambda *a, **kw: (object(), {}))
    monkeypatch.setattr(core.torch.utils.data, "DataLoader", lambda *a, **kw: batches)
    monkeypatch.setattr(core, "_normalize_batch", lambda batch, *a: batch)
    return run_stage(
        stage="tokenize", output_dir=output, fingerprint={}, overwrite=False,
        logger=logging.getLogger(__name__),
        work=lambda: core.tokenize_loop(
            spec, dataset_name="test", input_shar_dirs=[], planned_shar_fields={},
            rank=0, world_size=1, local_rank=0, final_output_dir=output,
            handler=handler, assigned_cut_count=len(batches),
        ),
    )


def test_oom_fails_completeness_and_records_cut_ids(monkeypatch, tmp_path):
    def encode(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("injected OOM")

    tokenizer = SimpleNamespace(omni_tokenizer=range(1000), tokenize_batch=encode)
    with pytest.raises(RuntimeError):
        _run(monkeypatch, tmp_path / "out", _CpuAudioOnly(_spec()),
             [_batch("failed"), _batch("later")], tokenizer, _spec())
    out = tmp_path / "out"
    assert not (out / "_SUCCESS").exists()
    assert not list(out.glob("*.idx"))
    stats = json.loads((out / "rank_0000_stats.json").read_text())
    assert stats["success"] is False
    assert stats["failed_cut_ids"] == ["failed"]


def test_mid_row_write_failure_aborts_structured_chunk(monkeypatch, tmp_path):
    class FailSecondWrite:
        def __init__(self, fh):
            self.fh, self.calls = fh, 0

        def write(self, data):
            self.calls += 1
            if self.calls == 2:
                raise OSError("injected text write failure")
            return self.fh.write(data)

        def __getattr__(self, name):
            return getattr(self.fh, name)

    class Handler(_CpuAudioText):
        def setup_writer(self, *args):
            super().setup_writer(*args)
            partition = self._writer._get_partition_writer("bucket_0000")
            partition._open_chunk()
            partition._text_fh = FailSecondWrite(partition._text_fh)

    spec = _spec("audio_text").model_copy(update={"audio_text_format": "interleaved"})
    tokenizer = SimpleNamespace(tokenize_batch_raw=lambda audio, *a, **kw: [[int(audio[0, 0])]])
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        _run(monkeypatch, out, Handler(spec, dataset_name="test"),
             [_batch("first", 11, supervised=True), _batch("failed", 22, supervised=True),
              _batch("last", 33, supervised=True)], tokenizer, spec)
    assert not (out / "_SUCCESS").exists()
    assert not list(out.rglob("clips.*.parquet"))
    assert not list(out.rglob("*.tmp"))
    stats = json.loads((out / "rank_0000_stats.json").read_text())
    assert stats["success"] is False
    assert stats["failed_cut_ids"] == ["failed"]
    assert stats["affected_cut_ids"] == ["first", "failed"]


def test_finalization_failure_reports_failed_rank(monkeypatch, tmp_path):
    class Handler(_CpuAudioOnly):
        def finalize_writer(self):
            raise OSError("injected finalization failure")

    tokenizer = SimpleNamespace(
        omni_tokenizer=range(1000), tokenize_batch=lambda *a, **kw: [torch.tensor([1, 11, 2])],
    )
    out = tmp_path / "out"
    with pytest.raises((RuntimeError, OSError)):
        _run(monkeypatch, out, Handler(_spec()), [_batch("first")], tokenizer, _spec())
    stats = json.loads((out / "rank_0000_stats.json").read_text())
    assert stats["success"] is False
    assert not (out / "_SUCCESS").exists()
    assert not list(out.glob("*.tmp"))
    assert stats["failed_cut_ids"] == ["first"]
    assert stats["affected_cut_ids"] == ["first"]


def test_pipeline_setup_failure_reports_failed_rank(monkeypatch, tmp_path):
    def fail(*a, **kw):
        raise ValueError("injected model initialization failure")

    monkeypatch.setattr(stage, "_invoke_pipeline", fail)
    assignment = SimpleNamespace(active=True, rank=1, fields={}, cut_count=1)
    with pytest.raises(ValueError, match="initialization"):
        stage._invoke_pipeline_for_assignment(
            _spec(), dataset_name="test", input_shar_dirs=[], final_output_dir=tmp_path,
            rank_assignment=assignment, world_size=2, local_rank=0,
        )
    assert json.loads((tmp_path / "rank_0001_stats.json").read_text())["success"] is False


def test_known_failed_rank_does_not_wait_for_missing_rank(tmp_path):
    write_rank_stats(tmp_path, {"rank": 0, "success": False, "error": "disk full"})
    with pytest.raises(RuntimeError, match=r"reported failure.*0"):
        wait_for_rank_stats(tmp_path, expected_ranks=2, timeout_sec=0)


@pytest.mark.parametrize("tokens", [[], [None], [[]]])
def test_missing_tokens_are_rejected_before_writing(tokens):
    tokenizer = SimpleNamespace(tokenize_batch=lambda *a, **kw: tokens)
    with pytest.raises(ValueError, match="token"):
        tokenize_unsupervised_batch(
            _batch("cut"), tokenizer, target_sr=24000, device="cpu", dtype=torch.int64,
        )


def test_single_bucket_uses_lhotse_explicit_duration_bins():
    kwargs = core._build_sampler_kwargs(_spec())
    assert kwargs["duration_bins"] == []


def test_unrelated_sampler_assertion_is_not_retried(monkeypatch):
    calls = []
    def fail(*a, **kw):
        calls.append(kw)
        raise AssertionError()

    monkeypatch.setattr(core, "DynamicBucketingSampler", fail)
    with pytest.raises(AssertionError):
        core._create_rank_sampler([], {}, rank=0)
    assert len(calls) == 1


@pytest.mark.parametrize("output_format", ["direct", "interleaved"])
@pytest.mark.parametrize("tokens", [[], [None], [[]]])
def test_missing_supervised_tokens_are_rejected_before_writing(output_format, tokens):
    spec = _spec("audio_text").model_copy(update={"audio_text_format": output_format})
    handler = AudioTextHandler(spec, dataset_name="test")
    tokenizer = SimpleNamespace(tokenize_batch_raw=lambda *a, **kw: tokens)
    with pytest.raises(ValueError, match="token"):
        handler.process_batch(_batch("cut", supervised=True), tokenizer,
                              SimpleNamespace(), 24000, "cpu")


def test_native_lhotse_single_bucket_respects_padded_budget():
    from lhotse import CutSet, MonoCut

    cuts = CutSet.from_cuts([
        MonoCut(id=str(i), start=0, duration=duration, channel=0)
        for i, duration in enumerate([1, 1, 1, 1, 1, 20])
    ])
    spec = _spec()
    spec = spec.model_copy(update={"dataloader": spec.dataloader.model_copy(
        update={"max_batch_duration": 50},
    )})
    sampler, _ = core._create_rank_sampler(cuts, core._build_sampler_kwargs(spec), rank=0)
    batches = list(sampler)
    assert sorted(cut.id for batch in batches for cut in batch) == [str(i) for i in range(6)]
    assert all(max(cut.duration for cut in batch) * len(batch) <= 50 for batch in batches)


@pytest.mark.parametrize("mode", ["audio_only", "audio_text", "audio_cache"])
@pytest.mark.parametrize("after_marker", [False, True])
def test_abort_cleans_unpublished_renames_and_preserves_commits(monkeypatch, tmp_path, mode, after_marker):
    from audio_tokenization.utils import io

    spec = _spec(mode).model_copy(update={"audio_text_format": "interleaved"})
    if mode == "audio_cache":
        mapping_dir = tmp_path / "tokenizer"
        mapping_dir.mkdir()
        (mapping_dir / "audio_token_mapping.json").write_text(json.dumps({
            "audio_token_offset": 100,
            "structure_tokens": {"audio_start": 1, "audio_end": 3},
        }))
        spec = spec.model_copy(update={"tokenizer": spec.tokenizer.model_copy(
            update={"path": str(mapping_dir)},
        )})
    if mode in ("audio_only", "audio_cache"):
        handler = AudioOnlyHandler(spec) if mode == "audio_only" else AudioCacheHandler(spec)
        tokenizer = SimpleNamespace(omni_tokenizer=range(1000), tokenize_batch=lambda *a, **k: [torch.tensor([1, 2, 3])])
        marker = ".idx" if mode == "audio_only" else ".parquet"
    else:
        handler = AudioTextHandler(spec, dataset_name="test")
        tokenizer = SimpleNamespace(tokenize_batch_raw=lambda *a, **k: [[1, 2, 3]])
        marker = ".parquet"
    handler.setup_writer(str(tmp_path), 0, 0, tokenizer)
    from audio_tokenization.pipelines.lhotse.checkpoint import WorkerStats
    stats = WorkerStats()
    handler.process_batch(_batch("committed", supervised=mode == "audio_text"), tokenizer, stats, 24000, "cpu")
    handler.checkpoint_writer()
    committed = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and not path.name.endswith(".tmp")}
    handler.process_batch(_batch("next", supervised=mode == "audio_text"), tokenizer, stats, 24000, "cpu")
    replace = io.os.replace
    def fail_at_marker(source, target):
        if str(target).endswith(marker):
            if after_marker:
                replace(source, target)
            raise OSError("injected marker publication failure")
        replace(source, target)
    monkeypatch.setattr(io.os, "replace", fail_at_marker)
    with pytest.raises(OSError):
        handler.finalize_writer()
    handler.abort_writer()
    handler.abort_writer()
    assert not list(tmp_path.rglob("*.tmp"))
    assert all(path.read_bytes() == payload for path, payload in committed.items())
    final_paths = {path for path in tmp_path.rglob("*") if path.is_file()}
    if after_marker:
        assert len(final_paths - committed.keys()) == (2 if mode == "audio_cache" else 3)
    else:
        assert final_paths == committed.keys()


def test_trimmed_empty_audio_cache_tokens_are_rejected():
    tokenizer = SimpleNamespace(tokenize_batch=lambda *a, **kw: [torch.tensor([1, 2])])
    with pytest.raises(ValueError, match="token"):
        tokenize_unsupervised_batch(
            _batch("cut"), tokenizer, target_sr=24000, device="cpu", dtype=torch.int32,
            trim_prefix_tokens=1, trim_suffix_tokens=1,
        )


def test_stage_preserves_original_failed_rank_error(monkeypatch, tmp_path):
    write_rank_stats(tmp_path, {"rank": 0, "success": False,
                               "error": "OutOfMemoryError: original allocation failure",
                               "failed_cut_ids": ["cut"]})
    def fail(*a, **kw):
        raise RuntimeError("Tokenization loop failed")
    monkeypatch.setattr(stage, "_invoke_pipeline", fail)
    with pytest.raises(RuntimeError):
        stage._invoke_pipeline_for_assignment(
            _spec(), dataset_name="test", input_shar_dirs=[], final_output_dir=tmp_path,
            rank_assignment=SimpleNamespace(active=True, rank=0, fields={}, cut_count=1),
            world_size=1, local_rank=0,
        )
    stats = json.loads((tmp_path / "rank_0000_stats.json").read_text())
    assert stats["error"] == "OutOfMemoryError: original allocation failure"
    assert stats["failed_cut_ids"] == ["cut"]
