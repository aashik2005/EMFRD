"""
Explainability API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch

from backend.models import RoBERTaContrastive
from backend.models.roberta_contrastive import RoBERTaContrastiveTokenizer
from backend.explainability import (
    SHAPExplainer,
    CounterfactualGenerator,
    ModalityContributionAnalyzer,
)
from backend.preprocessing import TextPreprocessor
from backend.utils import get_device, CheckpointManager
from backend.config import settings


router = APIRouter()

# Global cache
_explainer_cache = {}


class ExplainRequest(BaseModel):
    review_text: str
    model_name: str = "roberta_contrastive"
    num_samples: int = 100


class CounterfactualRequest(BaseModel):
    review_text: str
    model_name: str = "roberta_contrastive"
    max_changes: int = 5
    num_counterfactuals: int = 3


class ModalityAnalysisRequest(BaseModel):
    review_text: str
    include_graph: bool = False
    include_adversarial: bool = False
    include_metadata: bool = False


def get_shap_explainer():
    """Get or create SHAP explainer"""
    if "shap" not in _explainer_cache:
        print("Loading SHAP explainer...")

        device = get_device(settings.DEVICE)

        # Load model
        model = RoBERTaContrastive(
            model_name=settings.ROBERTA_MODEL,
            num_labels=2,
        )

        checkpoint_dir = settings.MODELS_DIR / "roberta_contrastive"
        checkpoint_manager = CheckpointManager(checkpoint_dir, "roberta_contrastive")

        if checkpoint_manager.exists("best.pt"):
            checkpoint_manager.load(model, "best.pt", device)
        else:
            print("WARNING: No trained model found")

        model = model.to(device)
        model.eval()

        # Create tokenizer
        tokenizer = RoBERTaContrastiveTokenizer(
            model_name=settings.ROBERTA_MODEL,
            max_length=settings.MAX_SEQ_LENGTH,
        ).tokenizer

        # Create explainer
        explainer = SHAPExplainer(model, tokenizer, device=device)

        _explainer_cache["shap"] = explainer

    return _explainer_cache["shap"]


def get_counterfactual_generator():
    """Get or create counterfactual generator"""
    if "counterfactual" not in _explainer_cache:
        print("Loading counterfactual generator...")

        device = get_device(settings.DEVICE)

        # Load model
        model = RoBERTaContrastive(
            model_name=settings.ROBERTA_MODEL,
            num_labels=2,
        )

        checkpoint_dir = settings.MODELS_DIR / "roberta_contrastive"
        checkpoint_manager = CheckpointManager(checkpoint_dir, "roberta_contrastive")

        if checkpoint_manager.exists("best.pt"):
            checkpoint_manager.load(model, "best.pt", device)

        model = model.to(device)
        model.eval()

        # Create tokenizer
        tokenizer = RoBERTaContrastiveTokenizer(
            model_name=settings.ROBERTA_MODEL,
            max_length=settings.MAX_SEQ_LENGTH,
        ).tokenizer

        # Create generator
        generator = CounterfactualGenerator(model, tokenizer, device=device)

        _explainer_cache["counterfactual"] = generator

    return _explainer_cache["counterfactual"]


@router.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """
    Generate SHAP explanation for a prediction

    Provides token-level importance scores showing which words
    contributed most to the prediction.
    """
    try:
        # Preprocess
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.preprocess(request.review_text)

        if not preprocessor.is_valid_review(cleaned_text):
            raise HTTPException(status_code=400, detail="Invalid review text")

        # Get explainer
        explainer = get_shap_explainer()

        # Generate explanation
        explanation = explainer.explain_prediction(
            cleaned_text,
            num_samples=request.num_samples,
        )

        return {
            "success": True,
            "explanation": explanation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counterfactual")
async def generate_counterfactual(request: CounterfactualRequest):
    """
    Generate counterfactual explanation

    Shows minimal changes needed to flip the prediction,
    helping understand model decision boundaries.
    """
    try:
        # Preprocess
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.preprocess(request.review_text)

        if not preprocessor.is_valid_review(cleaned_text):
            raise HTTPException(status_code=400, detail="Invalid review text")

        # Get generator
        generator = get_counterfactual_generator()

        # Generate counterfactual
        if request.num_counterfactuals > 1:
            counterfactuals = generator.generate_multiple_counterfactuals(
                cleaned_text,
                num_counterfactuals=request.num_counterfactuals,
                max_changes=request.max_changes,
            )
        else:
            cf = generator.generate_counterfactual(
                cleaned_text,
                max_changes=request.max_changes,
            )
            counterfactuals = [cf]

        return {
            "success": True,
            "counterfactuals": counterfactuals,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/modality-contribution")
async def analyze_modality_contribution(request: ModalityAnalysisRequest):
    """
    Analyze contribution of each modality (for fusion model)

    Shows how much each modality (semantic, graph, adversarial, metadata)
    contributed to the final prediction.
    """
    try:
        # Preprocess
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.preprocess(request.review_text)

        if not preprocessor.is_valid_review(cleaned_text):
            raise HTTPException(status_code=400, detail="Invalid review text")

        # TODO: Implement full modality analysis with fusion model
        # For now, return placeholder

        return {
            "success": True,
            "contributions": {
                "semantic": 0.65,
                "graph": 0.20,
                "adversarial": 0.10,
                "metadata": 0.05,
            },
            "message": "Modality analysis requires trained fusion model",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance")
async def get_feature_importance(
    texts: Optional[List[str]] = None,
    num_samples: int = 50,
):
    """
    Get aggregate feature importance across multiple texts

    Returns the most important tokens/features for classification.
    """
    try:
        if texts is None or len(texts) == 0:
            raise HTTPException(
                status_code=400,
                detail="Must provide at least one text"
            )

        # Get explainer
        explainer = get_shap_explainer()

        # Compute importance
        importance = explainer.get_feature_importance(texts, num_samples=num_samples)

        # Return top 20
        top_features = dict(list(importance.items())[:20])

        return {
            "success": True,
            "feature_importance": top_features,
            "num_texts_analyzed": len(texts),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def list_explainability_methods():
    """List available explainability methods"""
    return {
        "methods": [
            {
                "name": "SHAP",
                "description": "Token-level importance scores using SHAP values",
                "endpoint": "/explain",
                "supported_models": ["roberta_baseline", "roberta_contrastive"],
            },
            {
                "name": "Counterfactual",
                "description": "Minimal changes to flip prediction",
                "endpoint": "/counterfactual",
                "supported_models": ["roberta_baseline", "roberta_contrastive"],
            },
            {
                "name": "Modality Contribution",
                "description": "Contribution of each modality in fusion model",
                "endpoint": "/modality-contribution",
                "supported_models": ["fusion"],
            },
            {
                "name": "Feature Importance",
                "description": "Aggregate feature importance across texts",
                "endpoint": "/feature-importance",
                "supported_models": ["roberta_baseline", "roberta_contrastive"],
            },
        ]
    }
