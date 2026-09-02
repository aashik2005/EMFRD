"""
Canonical data schema for EMFRD
All datasets will be normalized to this schema
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class ReviewRecord:
    """Canonical review record schema"""

    # Required fields
    review_id: str
    review_text: str
    label: int  # 0 = genuine, 1 = fake

    # Optional metadata (for HGNN and fusion)
    user_id: Optional[str] = None
    product_id: Optional[str] = None
    rating: Optional[float] = None
    timestamp: Optional[datetime] = None

    # Additional optional fields
    verified_purchase: Optional[bool] = None
    helpful_votes: Optional[int] = None
    category: Optional[str] = None
    seller_id: Optional[str] = None
    product_title: Optional[str] = None
    metadata: Optional[dict] = None

    def has_graph_fields(self) -> bool:
        """Check if this record has fields required for graph construction"""
        return (
            self.user_id is not None
            and self.product_id is not None
            and self.review_id is not None
        )

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "review_id": self.review_id,
            "review_text": self.review_text,
            "label": self.label,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "rating": self.rating,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "verified_purchase": self.verified_purchase,
            "helpful_votes": self.helpful_votes,
            "category": self.category,
            "seller_id": self.seller_id,
            "product_title": self.product_title,
            "metadata": self.metadata,
        }


@dataclass
class DatasetInfo:
    """Dataset information and statistics"""

    name: str
    total_reviews: int
    fake_count: int
    genuine_count: int
    has_user_id: bool
    has_product_id: bool
    has_timestamp: bool
    has_rating: bool
    train_size: int
    val_size: int
    test_size: int
    missing_values: dict
    duplicate_count: int

    @property
    def class_balance(self) -> float:
        """Calculate class balance ratio"""
        return self.fake_count / self.genuine_count if self.genuine_count > 0 else 0.0

    @property
    def can_build_graph(self) -> bool:
        """Check if dataset can support graph construction"""
        return self.has_user_id and self.has_product_id

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_reviews": self.total_reviews,
            "fake_count": self.fake_count,
            "genuine_count": self.genuine_count,
            "class_balance": self.class_balance,
            "has_user_id": self.has_user_id,
            "has_product_id": self.has_product_id,
            "has_timestamp": self.has_timestamp,
            "has_rating": self.has_rating,
            "can_build_graph": self.can_build_graph,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "missing_values": self.missing_values,
            "duplicate_count": self.duplicate_count,
        }
