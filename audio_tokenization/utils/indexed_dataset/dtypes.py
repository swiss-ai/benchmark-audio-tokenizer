"""Lightweight dtype helpers for Megatron indexed datasets."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Type, Union

import numpy as np


class DType(Enum):
    """NumPy dtype enum used by Megatron indexed-dataset ``.idx`` files."""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[np.number]) -> int:
        """Return the Megatron dtype code for a NumPy dtype class."""
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[np.number]:
        """Return the NumPy dtype class for a Megatron dtype code."""
        return getattr(np, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[np.number]]) -> int:
        """Return the dtype size in bytes."""
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        if np.number in key.__mro__:
            return key().itemsize
        raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[np.number]:
        """Choose the compact token dtype for a vocabulary cardinality."""
        if cardinality is not None and cardinality < 65500:
            return np.uint16
        return np.int32
