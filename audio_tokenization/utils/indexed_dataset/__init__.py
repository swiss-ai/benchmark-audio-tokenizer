"""Megatron indexed-dataset helpers and merge utilities.

Keep heavy Megatron/torch imports lazy. Lightweight validation helpers import
submodules from this package during canaries and should not initialize CUDA.
"""

__all__ = [
    "DType",
    "IndexedDatasetBuilder",
    "discover_indexed_prefixes",
    "merge_indexed_dataset",
]


def __getattr__(name):
    if name == "DType":
        from .dtypes import DType

        return DType
    if name == "IndexedDatasetBuilder":
        from .indexed_dataset_megatron import IndexedDatasetBuilder

        return IndexedDatasetBuilder
    if name in {"discover_indexed_prefixes", "merge_indexed_dataset"}:
        from .merge_indexed_dataset import discover_indexed_prefixes, merge_indexed_dataset

        return {
            "discover_indexed_prefixes": discover_indexed_prefixes,
            "merge_indexed_dataset": merge_indexed_dataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
