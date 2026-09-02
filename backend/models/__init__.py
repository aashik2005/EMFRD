"""
Models module for EMFRD
"""
from .base import BaseFakeReviewModel
from .roberta_baseline import RoBERTaBaseline
from .roberta_contrastive import RoBERTaContrastive

# HGNN requires DGL - import conditionally
try:
    from .hgnn import HGNN, HGNNWithFeatures
    __all__ = ["BaseFakeReviewModel", "RoBERTaBaseline", "RoBERTaContrastive", "HGNN", "HGNNWithFeatures"]
except ImportError:
    __all__ = ["BaseFakeReviewModel", "RoBERTaBaseline", "RoBERTaContrastive"]
