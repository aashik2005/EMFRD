"""
Kaggle Fake Reviews Dataset Adapter
https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
https://osf.io/tyue9/
"""
from typing import List
from pathlib import Path
import pandas as pd
from datetime import datetime
from .dataset_base import BaseDataset
from .schemas import ReviewRecord


class FakeReviewsDataset(BaseDataset):
    """
    Kaggle Fake Reviews Dataset

    Expected format:
    - CSV file with columns that may include:
      - text/review/review_text (review content)
      - label/rating/is_fake (label)
      - Other optional fields
    """

    def download(self) -> None:
        """
        Download instructions for manual download

        This dataset requires manual download from Kaggle.
        Users should download and place in data/raw/fake_reviews/
        """
        print("\n" + "="*80)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*80)
        print("\nKaggle Fake Reviews Dataset:")
        print("  1. Visit: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset")
        print("  2. Download the CSV file")
        print(f"  3. Place it in: {self.data_dir}/")
        print("\nAlternative (OSF):")
        print("  1. Visit: https://osf.io/tyue9/")
        print("  2. Download fake_reviews_dataset.csv")
        print(f"  3. Place it in: {self.data_dir}/")
        print("\nExpected file name: fake_reviews_dataset.csv or deceptive-opinion.csv")
        print("="*80 + "\n")

        raise FileNotFoundError(
            f"Dataset files not found in {self.data_dir}. "
            "Please download manually (see instructions above)."
        )

    def load(self) -> pd.DataFrame:
        """
        Load Kaggle Fake Reviews dataset

        Returns:
            Raw DataFrame
        """
        # Try common file names
        possible_files = [
            "fake_reviews_dataset.csv",
            "deceptive-opinion.csv",
            "reviews.csv",
            "fake_reviews.csv",
        ]

        dataset_file = None
        for fname in possible_files:
            fpath = self.data_dir / fname
            if fpath.exists():
                dataset_file = fpath
                break

        if dataset_file is None:
            # List available files for debugging
            available = list(self.data_dir.glob("*.csv"))
            if available:
                print(f"Found CSV files: {[f.name for f in available]}")
                # Use the first CSV file found
                dataset_file = available[0]
                print(f"Using: {dataset_file.name}")
            else:
                self.download()  # Show download instructions
                return pd.DataFrame()

        print(f"Loading from: {dataset_file}")

        # Try to load with different encodings
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(dataset_file, encoding=encoding)
                print(f"Successfully loaded with encoding: {encoding}")
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not read {dataset_file} with any encoding")

    def normalize(self, df: pd.DataFrame) -> List[ReviewRecord]:
        """
        Normalize Kaggle Fake Reviews dataset to canonical schema

        Args:
            df: Raw DataFrame

        Returns:
            List of ReviewRecord objects
        """
        records = []

        # Identify columns (handle different naming conventions)
        text_col = self._find_column(df, ['text', 'review', 'review_text', 'text_', 'reviewText'])
        label_col = self._find_column(df, ['label', 'is_fake', 'fake', 'deceptive', 'category'])
        rating_col = self._find_column(df, ['rating', 'stars', 'star_rating'])
        user_col = self._find_column(df, ['user_id', 'userId', 'reviewer_id', 'reviewerId'])
        product_col = self._find_column(df, ['product_id', 'productId', 'asin', 'item_id'])
        timestamp_col = self._find_column(df, ['timestamp', 'date', 'review_date', 'time'])

        if text_col is None:
            raise ValueError(f"Could not find text column in: {list(df.columns)}")

        if label_col is None:
            raise ValueError(f"Could not find label column in: {list(df.columns)}")

        print(f"\nColumn mapping:")
        print(f"  Text: {text_col}")
        print(f"  Label: {label_col}")
        print(f"  Rating: {rating_col}")
        print(f"  User ID: {user_col}")
        print(f"  Product ID: {product_col}")
        print(f"  Timestamp: {timestamp_col}")

        for idx, row in df.iterrows():
            # Get text
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""

            # Skip empty reviews
            if not text or text == "nan" or len(text.strip()) < 10:
                continue

            # Get label and normalize to 0/1
            label_val = row[label_col]
            label = self._normalize_label(label_val)

            # Get optional fields
            rating = float(row[rating_col]) if rating_col and pd.notna(row[rating_col]) else None
            user_id = str(row[user_col]) if user_col and pd.notna(row[user_col]) else None
            product_id = str(row[product_col]) if product_col and pd.notna(row[product_col]) else None

            # Parse timestamp
            timestamp = None
            if timestamp_col and pd.notna(row[timestamp_col]):
                timestamp = self._parse_timestamp(row[timestamp_col])

            record = ReviewRecord(
                review_id=f"review_{idx}",
                review_text=text.strip(),
                label=label,
                user_id=user_id,
                product_id=product_id,
                rating=rating,
                timestamp=timestamp,
            )

            records.append(record)

        return records

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> str:
        """Find column name from list of candidates (case-insensitive)"""
        df_cols_lower = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            if candidate.lower() in df_cols_lower:
                return df_cols_lower[candidate.lower()]
        return None

    def _normalize_label(self, label_val) -> int:
        """
        Normalize label to binary 0/1
        0 = genuine/truthful/real
        1 = fake/deceptive/spam
        """
        if pd.isna(label_val):
            return 0  # Default to genuine if missing

        # Handle string labels
        if isinstance(label_val, str):
            label_lower = label_val.lower().strip()

            # Fake indicators
            if label_lower in ['fake', 'deceptive', 'spam', 'fraud', '1', 'true', 'yes', 'cg', 'computer_generated']:
                return 1

            # Genuine indicators
            if label_lower in ['genuine', 'real', 'truthful', 'legitimate', '0', 'false', 'no', 'or', 'original']:
                return 0

        # Handle numeric labels
        try:
            numeric = float(label_val)
            # Assume 1 = fake, 0 = genuine
            return 1 if numeric > 0.5 else 0
        except (ValueError, TypeError):
            pass

        # Default to genuine if we can't parse
        print(f"Warning: Could not parse label '{label_val}', defaulting to genuine (0)")
        return 0

    def _parse_timestamp(self, ts_val) -> datetime:
        """Parse timestamp from various formats"""
        if isinstance(ts_val, datetime):
            return ts_val

        if isinstance(ts_val, str):
            # Try common formats
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%Y/%m/%d",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(ts_val, fmt)
                except ValueError:
                    continue

        return None
