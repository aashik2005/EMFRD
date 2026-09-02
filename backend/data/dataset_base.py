"""
Base dataset class for EMFRD
All dataset adapters must inherit from this class
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from pathlib import Path
import pandas as pd
from .schemas import ReviewRecord, DatasetInfo


class BaseDataset(ABC):
    """Abstract base class for all EMFRD datasets"""

    def __init__(self, data_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize dataset

        Args:
            data_dir: Directory containing raw dataset files
            cache_dir: Directory for caching processed features
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.df: Optional[pd.DataFrame] = None
        self.info: Optional[DatasetInfo] = None

    @abstractmethod
    def download(self) -> None:
        """Download dataset if needed"""
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load raw dataset from disk

        Returns:
            DataFrame with raw data
        """
        pass

    @abstractmethod
    def normalize(self, df: pd.DataFrame) -> List[ReviewRecord]:
        """
        Normalize dataset to canonical ReviewRecord schema

        Args:
            df: Raw DataFrame

        Returns:
            List of ReviewRecord objects
        """
        pass

    def validate(self) -> DatasetInfo:
        """
        Validate dataset and generate statistics

        Returns:
            DatasetInfo with dataset statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Basic statistics
        total_reviews = len(self.df)

        # Check for label column
        if 'label' not in self.df.columns:
            raise ValueError("Dataset must have 'label' column")

        fake_count = int((self.df['label'] == 1).sum())
        genuine_count = int((self.df['label'] == 0).sum())

        # Check for optional fields
        has_user_id = 'user_id' in self.df.columns and self.df['user_id'].notna().any()
        has_product_id = 'product_id' in self.df.columns and self.df['product_id'].notna().any()
        has_timestamp = 'timestamp' in self.df.columns and self.df['timestamp'].notna().any()
        has_rating = 'rating' in self.df.columns and self.df['rating'].notna().any()

        # Missing values
        missing_values = self.df.isnull().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_values.items() if v > 0}

        # Duplicates
        duplicate_count = int(self.df.duplicated(subset=['review_text']).sum())

        self.info = DatasetInfo(
            name=self.__class__.__name__,
            total_reviews=total_reviews,
            fake_count=fake_count,
            genuine_count=genuine_count,
            has_user_id=has_user_id,
            has_product_id=has_product_id,
            has_timestamp=has_timestamp,
            has_rating=has_rating,
            train_size=0,  # Will be set after splitting
            val_size=0,
            test_size=0,
            missing_values=missing_values,
            duplicate_count=duplicate_count,
        )

        return self.info

    def prepare(self) -> Tuple[List[ReviewRecord], DatasetInfo]:
        """
        Complete pipeline: download -> load -> normalize -> validate

        Returns:
            Tuple of (records, dataset_info)
        """
        # Download if needed
        if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
            print(f"Downloading dataset to {self.data_dir}...")
            self.download()

        # Load raw data
        print("Loading dataset...")
        self.df = self.load()

        # Validate
        print("Validating dataset...")
        info = self.validate()

        # Normalize to canonical schema
        print("Normalizing to canonical schema...")
        records = self.normalize(self.df)

        print(f"Dataset prepared: {len(records)} reviews")
        print(f"  Fake: {info.fake_count} ({info.fake_count/info.total_reviews*100:.1f}%)")
        print(f"  Genuine: {info.genuine_count} ({info.genuine_count/info.total_reviews*100:.1f}%)")
        print(f"  Can build graph: {info.can_build_graph}")

        return records, info

    def get_cache_path(self, name: str) -> Optional[Path]:
        """Get path for cached file"""
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{self.__class__.__name__}_{name}"
