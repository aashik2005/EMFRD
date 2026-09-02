"""
Data module for EMFRD
Handles dataset loading, preprocessing, and normalization
"""
from .dataset_base import BaseDataset
from .dataset_registry import DatasetRegistry, get_dataset
from .schemas import ReviewRecord, DatasetInfo

__all__ = [
    "BaseDataset",
    "DatasetRegistry",
    "get_dataset",
    "ReviewRecord",
    "DatasetInfo",
]
