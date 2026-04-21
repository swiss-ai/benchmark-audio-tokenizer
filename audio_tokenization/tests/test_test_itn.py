from __future__ import annotations

from queue import Empty

import pytest

import _test_itn


class _FakeProcess:
    def __init__(self, *, pid: int = 1234, exitcode: int | None = 0, alive: bool = True):
        self.pid = pid
        self.exitcode = exitcode
        self._alive = alive
        self.joined = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        del timeout  # _FakeProcess always "joins" instantly in tests
        self.joined = True
        self._alive = False


class _FakeQueue:
    def __init__(self, responses):
        self._responses = list(responses)

    def get(self, timeout=None):
        if not self._responses:
            raise Empty
        result = self._responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_await_worker_result_returns_results_and_joins():
    proc = _FakeProcess(exitcode=0, alive=True)
    q = _FakeQueue([{1: "x"}])

    results = _test_itn._await_worker_result(proc, q, poll_timeout_s=0.01)

    assert results == {1: "x"}
    assert proc.joined


def test_await_worker_result_raises_when_worker_exits_before_results():
    proc = _FakeProcess(exitcode=9, alive=False)
    q = _FakeQueue([Empty()])

    with pytest.raises(RuntimeError, match="before writing results"):
        _test_itn._await_worker_result(proc, q, poll_timeout_s=0.01)
    assert proc.joined
