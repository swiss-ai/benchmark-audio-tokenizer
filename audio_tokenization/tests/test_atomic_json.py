import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from audio_tokenization.utils import io as durable_io


def test_atomic_json_concurrent_writers_do_not_share_temporary_files(tmp_path, monkeypatch):
    # Threads share a PID, reproducing the name collision of equal PIDs on
    # different nodes. Hold both completed writes at the publication boundary.
    target = tmp_path / "_CACHE_LAYOUT.json"
    barrier = Barrier(2)
    replace = durable_io.os.replace

    def publish_together(source, destination):
        barrier.wait(timeout=10)
        replace(source, destination)

    monkeypatch.setattr(durable_io.os, "replace", publish_together)
    payloads = [{"writer": n, "layout": "x" * 8192} for n in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(durable_io.atomic_write_json, target, payload) for payload in payloads]
        for future in futures:
            future.result()

    assert json.loads(target.read_text()) in payloads
    assert list(tmp_path.iterdir()) == [target]
