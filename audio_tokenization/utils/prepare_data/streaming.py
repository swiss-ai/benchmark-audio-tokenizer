#!/usr/bin/env python3
"""Streaming row readers for prepare-time conversion."""

from __future__ import annotations

import gc
from typing import Any, Iterator


def _yield_materialized_rows(rows, *owned_refs: Any) -> Iterator[dict]:
    """Yield row dicts, then eagerly drop temporary batch objects."""
    try:
        for row in rows:
            yield row
    finally:
        del rows
        del owned_refs
        gc.collect()


def iter_parquet_rows(
    pq_path: str,
    *,
    columns: list[str],
    batch_size: int,
) -> Iterator[dict]:
    """Yield row dicts from a parquet shard without materializing the whole file."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(pq_path)
    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=batch_size,
        use_threads=False,
    ):
        rows = batch.to_pylist()
        yield from _yield_materialized_rows(rows, batch)


def iter_arrow_rows(
    arrow_path: str,
    *,
    batch_size: int,
) -> Iterator[dict]:
    """Yield row dicts from an Arrow stream without materializing the whole file."""
    import pyarrow.ipc as ipc

    reader = ipc.open_stream(arrow_path)
    for batch in reader:
        if batch.num_rows > batch_size:
            table = batch.to_table()
            try:
                for subbatch in table.to_batches(max_chunksize=batch_size):
                    rows = subbatch.to_pylist()
                    yield from _yield_materialized_rows(rows, subbatch)
            finally:
                del table
                del batch
                gc.collect()
        else:
            rows = batch.to_pylist()
            yield from _yield_materialized_rows(rows, batch)
