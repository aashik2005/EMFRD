"""
Models module for EMFRD
"""
from .base import BaseFakeReviewModel
from .roberta_baseline import RoBERTaBaseline

__all__ = ["BaseFakeReviewModel", "RoBERTaBaseline"]
