import struct
import tracemalloc

import numpy as np
import pytest
import torch

from audio_tokenization.utils.indexed_dataset.indexed_dataset_megatron import (
    IndexedDatasetBuilder,
)


@pytest.mark.parametrize("method", ["item", "document"])
@pytest.mark.parametrize("dtype,code,format_code", [(np.int32, 4, "i"), (np.uint16, 8, "H")])
@pytest.mark.parametrize(
    "source,values",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        (np.arange(1, 5, dtype=np.int32), [1, 2, 3, 4]),
        (np.arange(1, 9, dtype=np.int64)[::2], [1, 3, 5, 7]),
        (np.array([[1, 2], [3, 4]], order="F"), [1, 2, 3, 4]),
        (np.array([1, 2, 3, 4], dtype=">i4"), [1, 2, 3, 4]),
        (torch.tensor([1., 2., 3., 4.], requires_grad=True), [1, 2, 3, 4]),
        (np.empty((0, 2), dtype=np.int32), []),
    ],
    ids=["list", "array", "strided", "fortran", "big-endian", "tensor", "empty"],
)
def test_writer_preserves_payload_and_multimodal_index(
    tmp_path, method, dtype, code, format_code, source, values
):
    """Changing buffer handling must preserve C-order bytes and document boundaries."""
    bin_path, idx_path = tmp_path / "data.bin", tmp_path / "data.idx"
    writer = IndexedDatasetBuilder(str(bin_path), dtype=dtype, multimodal=True)
    if method == "item":
        writer.add_item(source, mode=7)
        writer.end_document()
    else:
        writer.add_document(source, lengths=[len(values)], modes=[7])
    writer.finalize(str(idx_path))

    assert bin_path.read_bytes() == struct.pack(f"<{len(values)}{format_code}", *values)
    assert idx_path.read_bytes() == (
        b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, code, 1, 2)
        + struct.pack("<iqqqb", len(values), 0, 0, 1, 7)
    )


@pytest.mark.parametrize("method", ["item", "document"])
def test_contiguous_payload_write_does_not_allocate_another_payload(tmp_path, method):
    """A large ready-to-write token array must not require another full-size buffer."""
    tokens = np.arange(1_048_576, dtype=np.int32)
    writer = IndexedDatasetBuilder(str(tmp_path / "data.bin"), dtype=np.int32)
    tracemalloc.start()
    try:
        if method == "item":
            writer.add_item(tokens)
        else:
            writer.add_document(tokens, lengths=[len(tokens)])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        writer.data_file.close()
    assert (tmp_path / "data.bin").stat().st_size == tokens.nbytes
    # Half a payload leaves ample room for fixed Python/file-buffer overhead.
    assert peak < tokens.nbytes // 2
