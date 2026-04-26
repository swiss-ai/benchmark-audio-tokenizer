"""Megatron-LM compatible IndexedDataset writer for token streams.

The audio tokenization pipeline writes Megatron ``.bin/.idx`` chunks for
audio-only and direct audio-text datasets. Each cut is stored as one Megatron
document, and a separate cut-ID sidecar records document identity.

This module intentionally implements only the small subset of the Megatron
indexed-dataset format needed by the pipeline writer and validation tools:

- ``dataset.bin`` stores contiguous token IDs.
- ``dataset.idx`` stores dtype, lengths, byte offsets, and document boundaries.
- dtype is selected from vocabulary cardinality to avoid wasting disk space.
- optional sequence modes are preserved for compatibility with Megatron readers.
"""

import struct
from typing import List, Optional, Type, Union

import numpy as np
import torch

from .constants import MEGATRON_INDEX_HEADER
from .dtypes import DType


class IndexedDatasetBuilder:
    """Builder class for the IndexedDataset class

    This is the exact implementation from the reference notebook.

    Args:
        bin_path (str): The path to the data (.bin) file
        dtype (Type[np.number], optional): The dtype of the index file. Defaults to np.int32.
        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(self, bin_path: str, dtype: Type[np.number] = np.int32, multimodal: bool = False) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype
        self.multimodal = multimodal

        self.sequence_lengths = []
        self.document_indices = [0]
        self.sequence_modes = [] if self.multimodal else None

    def add_item(self, tensor: torch.Tensor, mode: int = 0) -> None:
        """Add a single item to the dataset

        Args:
            tensor (torch.Tensor): The item to add to the data file
            mode (int, optional): The mode for the item. Defaults to 0.
        """
        if isinstance(tensor, torch.Tensor):
            np_array = np.array(tensor.cpu().detach().numpy(), dtype=self.dtype)
        else:
            np_array = np.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)
        if self.multimodal:
            self.sequence_modes.append(mode)

    def add_document(
        self, tensor: Union[torch.Tensor, List[int]], lengths: List[int], modes: Optional[List[int]] = None
    ) -> None:
        """Add an entire document to the dataset

        Args:
            tensor (torch.Tensor or List[int]): The document to add
            lengths (List[int]): The lengths of each item in the document
            modes (Optional[List[int]], optional): The modes for each item in the document. Defaults to None.
        """
        if isinstance(tensor, torch.Tensor):
            np_array = np.array(tensor.cpu().detach().numpy(), dtype=self.dtype)
        else:
            np_array = np.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))
        if self.multimodal:
            self.sequence_modes.extend(modes if modes is not None else [0] * len(lengths))

    def end_document(self) -> None:
        """Finalize the document, for use with IndexedDatasetBuilder.add_item"""
        self.document_indices.append(len(self.sequence_lengths))

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()

        with open(idx_path, "wb") as idx_writer:
            # Write header
            idx_writer.write(MEGATRON_INDEX_HEADER)
            # Write version
            idx_writer.write(struct.pack("<Q", 1))
            # Write dtype code
            idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))

            # Write counts
            # - sequence_count = N
            # - document_count (in file) = N+1 (length of document_indices array)
            # - actual documents = N
            sequence_count = len(self.sequence_lengths)
            idx_writer.write(struct.pack("<Q", sequence_count))

            # IMPORTANT: Write the length of document_indices array, not the number of documents
            # Megatron reads exactly this many elements from the array
            # Megatron then checks: assert sequence_count == document_indices[-1]
            document_count = len(self.document_indices)
            idx_writer.write(struct.pack("<Q", document_count))

            # Write document lengths (stored as sequence_lengths for compatibility)
            sequence_lengths = np.array(self.sequence_lengths, dtype=np.int32)
            idx_writer.write(sequence_lengths.tobytes(order="C"))

            # Write document pointers (byte offsets into .bin file)
            sequence_pointers = self._sequence_pointers(self.sequence_lengths)
            sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
            idx_writer.write(sequence_pointers.tobytes(order="C"))

            # Write document indices (for compatibility, [0, 1, 2, ..., #docs])
            document_indices = np.array(self.document_indices, dtype=np.int64)
            idx_writer.write(document_indices.tobytes(order="C"))

            # Write sequence modes if multimodal
            if self.sequence_modes is not None:
                sequence_modes = np.array(self.sequence_modes, dtype=np.int8)
                idx_writer.write(sequence_modes.tobytes(order="C"))

    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Build the sequence pointers per the sequence lengths and dtype size"""
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * itemsize
        return list_ptr


def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix"""
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix"""
    return path_prefix + ".bin"
