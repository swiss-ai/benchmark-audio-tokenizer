"""Typed dataset-spec loading for config-driven audio pipeline entrypoints."""

from .schema import (
    DatasetSpec,
    PrepareParquetInputSpec,
    PrepareSpec,
    ProductMatrixSpec,
    load_dataset_spec,
)

__all__ = [
    "DatasetSpec",
    "PrepareParquetInputSpec",
    "PrepareSpec",
    "ProductMatrixSpec",
    "load_dataset_spec",
]
