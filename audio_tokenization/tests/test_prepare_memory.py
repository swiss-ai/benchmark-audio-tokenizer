"""Conversion memory policy is applied before numerical libraries and worker startup."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process memory policy")
@pytest.mark.parametrize("command,stage,opt_out,wanted", [
    ("run", "convert", False, 1),
    ("run", "convert", True, 0),
    ("plan", "convert", False, 0),
    ("run", "tokenize", False, 0),
    ("run", "materialize", False, 0),
])
def test_cli_sets_memory_policy_before_loading_stage_code(command, stage, opt_out, wanted):
    # Stop at the stage import: running a real dataset here would write external output.
    script = r'''
import ctypes
import sys
from omegaconf import OmegaConf

libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
libc.prctl.restype = ctypes.c_int
assert libc.prctl(41, 0, 0, 0, 0) == 0

class StageBoundary(Exception):
    pass

class CheckStageImport:
    def find_spec(self, fullname, *args):
        if fullname == "audio_tokenization.stages":
            actual = libc.prctl(42, 0, 0, 0, 0)
            assert actual == int(sys.argv[3]), (actual, sys.argv[3])
            assert not {"numpy", "torch", "lhotse"}.intersection(sys.modules)
            raise StageBoundary

sys.meta_path.insert(0, CheckStageImport())
try:
    from audio_tokenization.__main__ import _execute_command
    _execute_command(sys.argv[1], OmegaConf.create({"stage": sys.argv[2]}))
except StageBoundary:
    print("stage boundary checked")
else:
    raise AssertionError("stage boundary not reached")
'''
    env = dict(os.environ)
    env.pop("AUDIO_PREPARE_DISABLE_THP", None)
    if opt_out:
        env["AUDIO_PREPARE_DISABLE_THP"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", script, command, stage, str(wanted)],
        env=env, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stage boundary checked" in result.stdout


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process memory policy")
@pytest.mark.parametrize("method", ["fork", "spawn", "forkserver"])
def test_conversion_workers_inherit_memory_policy(tmp_path, method):
    script = tmp_path / "worker_policy.py"
    script.write_text('''
import ctypes
import multiprocessing as mp
import sys
from audio_tokenization.prepare.memory import disable_transparent_huge_pages


def worker(queue):
    libc = ctypes.CDLL(None)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    queue.put(libc.prctl(42, 0, 0, 0, 0))


if __name__ == "__main__":
    disable_transparent_huge_pages()
    context = mp.get_context(sys.argv[1])
    queue = context.Queue()
    process = context.Process(target=worker, args=(queue,))
    process.start()
    assert queue.get(timeout=30) == 1
    process.join(timeout=30)
    assert process.exitcode == 0
''')
    env = dict(os.environ)
    env.pop("AUDIO_PREPARE_DISABLE_THP", None)
    result = subprocess.run(
        [sys.executable, str(script), method], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
