"""
Models module for EMFRD
"""
from .base import BaseFakeReviewModel
from .roberta_baseline import RoBERTaBaseline
from .roberta_contrastive import RoBERTaContrastive
from .gan_adversarial import GANAdversarial, Generator, Discriminator
from .gated_fusion import GatedMultimodalFusion, GatingNetwork

# HGNN requires DGL - import conditionally
try:
    from .hgnn import HGNN, HGNNWithFeatures
    _HGNN_AVAILABLE = True
except ImportError:
    _HGNN_AVAILABLE = False

# Build __all__ list
__all__ = [
    "BaseFakeReviewModel",
    "RoBERTaBaseline",
    "RoBERTaContrastive",
    "GANAdversarial",
    "Generator",
    "Discriminator",
    "GatedMultimodalFusion",
    "GatingNetwork",
]

if _HGNN_AVAILABLE:
    __all__.extend(["HGNN", "HGNNWithFeatures"])
