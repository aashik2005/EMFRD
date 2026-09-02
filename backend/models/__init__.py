"""
Models module for EMFRD
"""
from .base import BaseFakeReviewModel
from .roberta_baseline import RoBERTaBaseline
from .roberta_contrastive import RoBERTaContrastive

__all__ = ["BaseFakeReviewModel", "RoBERTaBaseline", "RoBERTaContrastive"]
