"""Worker-count resolution respects SLURM / CPU caps."""
from __future__ import annotations

import os

import pytest

from audio_tokenization.prepare.runtime import resolve_num_workers


@pytest.fixture
def _clean_env(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)


def test_explicit_count_passes_through(_clean_env):
    assert resolve_num_workers(16) == 16


def test_explicit_count_clamped_by_num_inputs(_clean_env):
    assert resolve_num_workers(100, num_inputs=8) == 8


def test_none_uses_slurm_cpus_when_set(_clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    assert resolve_num_workers(None) == 64


def test_none_clamped_by_num_inputs(_clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "288")
    assert resolve_num_workers(None, num_inputs=4) == 4


def test_none_falls_back_to_cpu_count_off_slurm(_clean_env):
    expected = os.cpu_count() or 1
    assert resolve_num_workers(None) == expected


def test_result_is_at_least_one_even_with_zero_inputs(_clean_env):
    assert resolve_num_workers(8, num_inputs=0) == 1


def test_slurm_dataset_oom_regression(_clean_env, monkeypatch):
    """Regression: prior behaviour was ``min(None or len(inputs), len(inputs))``,
    which spawned 1361 workers for SRG Apertus's 1361 shards and OOM-killed the
    node. Under SLURM with 288 cores allocated, the resolver must cap at 288."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "288")
    assert resolve_num_workers(None, num_inputs=1361) == 288
