"""
Dataset Registry for EMFRD
Centralized dataset factory
"""
from typing import Dict, Type, Optional
from pathlib import Path
from .dataset_base import BaseDataset
from .fake_reviews_dataset import FakeReviewsDataset

# FraudAmazon requires DGL - import conditionally
try:
    from .fraud_amazon_dataset import FraudAmazonDataset
    FRAUD_AMAZON_AVAILABLE = True
except ImportError:
    FRAUD_AMAZON_AVAILABLE = False


class DatasetRegistry:
    """Registry for all available datasets"""

    _datasets: Dict[str, Type[BaseDataset]] = {
        "fake_reviews": FakeReviewsDataset,
        # Future datasets will be added here:
        # "amazon_reviews_2023": AmazonReviews2023Dataset,
        # "modern_fake_reviews": ModernFakeReviewsDataset,
    }

    # Add FraudAmazon if DGL is available
    if FRAUD_AMAZON_AVAILABLE:
        _datasets["fraud_amazon"] = FraudAmazonDataset

    @classmethod
    def register(cls, name: str, dataset_class: Type[BaseDataset]) -> None:
        """
        Register a new dataset

        Args:
            name: Dataset identifier
            dataset_class: Dataset class
        """
        cls._datasets[name] = dataset_class

    @classmethod
    def get(cls, name: str, data_dir: Path, cache_dir: Optional[Path] = None) -> BaseDataset:
        """
        Get dataset instance by name

        Args:
            name: Dataset identifier
            data_dir: Data directory
            cache_dir: Cache directory

        Returns:
            Dataset instance
        """
        if name not in cls._datasets:
            available = list(cls._datasets.keys())
            raise ValueError(
                f"Dataset '{name}' not found. "
                f"Available datasets: {available}"
            )

        dataset_class = cls._datasets[name]
        return dataset_class(data_dir=data_dir, cache_dir=cache_dir)

    @classmethod
    def list_datasets(cls) -> list:
        """List all available datasets"""
        return list(cls._datasets.keys())


def get_dataset(
    name: str,
    data_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> BaseDataset:
    """
    Convenience function to get dataset instance

    Args:
        name: Dataset identifier
        data_dir: Data directory (default: ./data/raw/{name})
        cache_dir: Cache directory (default: ./data/cache)

    Returns:
        Dataset instance
    """
    if data_dir is None:
        data_dir = Path("./data/raw") / name

    if cache_dir is None:
        cache_dir = Path("./data/cache")

    return DatasetRegistry.get(name, data_dir, cache_dir)
