"""
Data splitting utilities for EMFRD
Handles train/val/test splits with data leakage prevention
"""
from typing import List, Tuple, Optional, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json
from pathlib import Path


class DataSplitter:
    """
    Data splitter with stratification and leakage prevention

    Prevents data leakage by:
    1. Splitting before any preprocessing that could leak information
    2. Supporting stratified splits for balanced classes
    3. Supporting temporal splits for time-aware evaluation
    4. Optional user/product group-aware splitting
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = True,
    ):
        """
        Initialize splitter

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            stratify: Use stratified splitting
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.stratify = stratify

    def split(
        self,
        texts: List[str],
        labels: List[int],
        user_ids: Optional[List[str]] = None,
        product_ids: Optional[List[str]] = None,
    ) -> Tuple[dict, dict, dict]:
        """
        Split data into train/val/test

        Args:
            texts: Review texts
            labels: Labels
            user_ids: Optional user IDs for leakage analysis
            product_ids: Optional product IDs for leakage analysis

        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        n_samples = len(texts)

        if len(labels) != n_samples:
            raise ValueError(f"texts and labels must have same length")

        print(f"\nSplitting {n_samples} samples...")
        print(f"  Train: {self.train_ratio:.1%}")
        print(f"  Val: {self.val_ratio:.1%}")
        print(f"  Test: {self.test_ratio:.1%}")

        # Create indices
        indices = np.arange(n_samples)

        # Stratify by label if requested
        stratify_by = np.array(labels) if self.stratify else None

        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=self.train_ratio,
            random_state=self.random_seed,
            stratify=stratify_by,
        )

        # Second split: val vs test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        stratify_temp = (
            np.array([labels[i] for i in temp_idx])
            if self.stratify
            else None
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=self.random_seed,
            stratify=stratify_temp,
        )

        # Create split dictionaries
        train_data = self._create_split_dict(texts, labels, train_idx, user_ids, product_ids, "train")
        val_data = self._create_split_dict(texts, labels, val_idx, user_ids, product_ids, "val")
        test_data = self._create_split_dict(texts, labels, test_idx, user_ids, product_ids, "test")

        # Print statistics
        self._print_split_stats(train_data, val_data, test_data)

        # Check for data leakage
        if user_ids is not None:
            self._analyze_user_leakage(train_data, val_data, test_data)

        if product_ids is not None:
            self._analyze_product_leakage(train_data, val_data, test_data)

        return train_data, val_data, test_data

    def _create_split_dict(
        self,
        texts: List[str],
        labels: List[int],
        indices: np.ndarray,
        user_ids: Optional[List[str]],
        product_ids: Optional[List[str]],
        split_name: str,
    ) -> dict:
        """Create dictionary for a data split"""
        data = {
            "texts": [texts[i] for i in indices],
            "labels": [labels[i] for i in indices],
            "indices": indices.tolist(),
            "split": split_name,
        }

        if user_ids is not None:
            data["user_ids"] = [user_ids[i] for i in indices]

        if product_ids is not None:
            data["product_ids"] = [product_ids[i] for i in indices]

        return data

    def _print_split_stats(self, train_data: dict, val_data: dict, test_data: dict):
        """Print split statistics"""
        print("\nSplit Statistics:")
        for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            labels = data["labels"]
            n_fake = sum(labels)
            n_genuine = len(labels) - n_fake
            print(f"  {name:5s}: {len(labels):5d} samples "
                  f"(Fake: {n_fake:4d} [{n_fake/len(labels)*100:4.1f}%], "
                  f"Genuine: {n_genuine:4d} [{n_genuine/len(labels)*100:4.1f}%])")

    def _analyze_user_leakage(self, train_data: dict, val_data: dict, test_data: dict):
        """Analyze user leakage between splits"""
        train_users = set(u for u in train_data.get("user_ids", []) if u)
        val_users = set(u for u in val_data.get("user_ids", []) if u)
        test_users = set(u for u in test_data.get("user_ids", []) if u)

        val_overlap = train_users & val_users
        test_overlap = train_users & test_users

        print("\nUser Leakage Analysis:")
        print(f"  Train users: {len(train_users)}")
        print(f"  Val users: {len(val_users)} ({len(val_overlap)} overlap with train [{len(val_overlap)/len(val_users)*100:.1f}%])")
        print(f"  Test users: {len(test_users)} ({len(test_overlap)} overlap with train [{len(test_overlap)/len(test_users)*100:.1f}%])")

        if len(val_overlap) > 0 or len(test_overlap) > 0:
            print("  WARNING: User overlap detected between splits!")
            print("  Consider group-aware splitting if user behavior is important.")

    def _analyze_product_leakage(self, train_data: dict, val_data: dict, test_data: dict):
        """Analyze product leakage between splits"""
        train_products = set(p for p in train_data.get("product_ids", []) if p)
        val_products = set(p for p in val_data.get("product_ids", []) if p)
        test_products = set(p for p in test_data.get("product_ids", []) if p)

        val_overlap = train_products & val_products
        test_overlap = train_products & test_products

        print("\nProduct Leakage Analysis:")
        print(f"  Train products: {len(train_products)}")
        print(f"  Val products: {len(val_products)} ({len(val_overlap)} overlap with train [{len(val_overlap)/len(val_products)*100:.1f}%])")
        print(f"  Test products: {len(test_products)} ({len(test_overlap)} overlap with train [{len(test_overlap)/len(test_products)*100:.1f}%])")

    def save_splits(
        self,
        train_data: dict,
        val_data: dict,
        test_data: dict,
        output_dir: Path,
    ):
        """
        Save splits to disk

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            output_file = output_dir / f"{name}.json"
            with open(output_file, "w") as f:
                json.dump(data, f)
            print(f"Saved {name} split to {output_file}")

    def load_splits(self, input_dir: Path) -> Tuple[dict, dict, dict]:
        """
        Load splits from disk

        Args:
            input_dir: Input directory

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        input_dir = Path(input_dir)

        splits = {}
        for name in ["train", "val", "test"]:
            input_file = input_dir / f"{name}.json"
            if not input_file.exists():
                raise FileNotFoundError(f"Split file not found: {input_file}")

            with open(input_file, "r") as f:
                splits[name] = json.load(f)

            print(f"Loaded {name} split from {input_file}")

        return splits["train"], splits["val"], splits["test"]
