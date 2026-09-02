"""
Explainability module for EMFRD

Provides interpretability for fake review detection models through:
- SHAP (SHapley Additive exPlanations)
- Counterfactual generation
- Modality contribution analysis
"""

from .shap_explainer import SHAPExplainer
from .counterfactual import CounterfactualGenerator
from .modality_analyzer import ModalityContributionAnalyzer

__all__ = [
    "SHAPExplainer",
    "CounterfactualGenerator",
    "ModalityContributionAnalyzer",
]
