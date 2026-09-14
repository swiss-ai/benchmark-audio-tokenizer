#!/usr/bin/env python3
"""Streaming row readers for prepare-time conversion."""

from __future__ import annotations

from typing import Iterator


def _iter_batch_rows(batch, *, batch_size: int) -> Iterator[dict]:
    """Own each materialized list only until its last row has been consumed.

    ``yield from`` releases the list iterator before the next slice is
    materialized. Callers must not retain a second reference to that row list.
    Arrow slices retain the original batch buffers until the batch is exhausted.
    """
    for start in range(0, batch.num_rows, batch_size):
        yield from batch.slice(start, batch_size).to_pylist()


def iter_parquet_rows(
    pq_path: str,
    *,
    columns: list[str],
    batch_size: int,
) -> Iterator[dict]:
    """Yield projected row dicts and close the file on exhaustion or cancellation."""
    import pyarrow.parquet as pq

    with pq.ParquetFile(pq_path) as parquet_file:
        for batch in parquet_file.iter_batches(
            columns=columns,
            batch_size=batch_size,
            use_threads=False,
        ):
            try:
                yield from _iter_batch_rows(batch, batch_size=batch_size)
            finally:
                del batch


def iter_arrow_rows(
    arrow_path: str,
    *,
    batch_size: int,
    columns: list[str] | None = None,
) -> Iterator[dict]:
    """Yield row dicts from an Arrow stream with optional root projection.

    Arrow IPC reads entire record batches. Selecting requested top-level roots
    before Python conversion avoids materializing unrelated columns; dotted
    selectors retain their complete struct root. Parquet additionally projects
    nested fields during its read.
    """
    import pyarrow.ipc as ipc

    with ipc.open_stream(arrow_path) as reader:
        for batch in reader:
            try:
                if columns is not None:
                    roots = dict.fromkeys(column.split(".", 1)[0] for column in columns)
                    batch = batch.select([root for root in roots if root in batch.schema.names])
                yield from _iter_batch_rows(batch, batch_size=batch_size)
            finally:
                del batch
