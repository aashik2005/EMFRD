"""
Dataset schemas
"""
from pydantic import BaseModel, Field
from typing import Dict


class DatasetValidation(BaseModel):
    """Dataset validation response"""
    name: str
    total_reviews: int
    fake_count: int
    genuine_count: int
    class_balance: float
    has_user_id: bool
    has_product_id: bool
    can_build_graph: bool
    missing_values: Dict[str, int]
    duplicate_count: int
    train_size: int
    val_size: int
    test_size: int
