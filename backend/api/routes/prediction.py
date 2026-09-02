"""
Prediction endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from backend.schemas import PredictionRequest, PredictionResponse, ModelType
from backend.models import RoBERTaBaseline
from backend.models.roberta_baseline import RoBERTaTokenizer
from backend.preprocessing import TextPreprocessor
from backend.utils import get_device, CheckpointManager
from backend.config import settings
import torch
from pathlib import Path

router = APIRouter()

# Global model cache
_model_cache = {}
_tokenizer_cache = {}


def get_roberta_model():
    """Load RoBERTa model (cached)"""
    model_key = "roberta_baseline"

    if model_key not in _model_cache:
        print("Loading RoBERTa baseline model...")

        # Get device
        device = get_device(settings.DEVICE)

        # Initialize model
        model = RoBERTaBaseline(
            model_name=settings.ROBERTA_MODEL,
            num_labels=2,
            dropout=settings.DROPOUT,
        )

        # Load checkpoint if exists
        checkpoint_dir = settings.MODELS_DIR / "roberta_baseline"
        checkpoint_manager = CheckpointManager(checkpoint_dir, "roberta_baseline")

        if checkpoint_manager.exists("best.pt"):
            checkpoint_manager.load(model, checkpoint_name="best.pt", device=device)
            print("Loaded trained model")
        else:
            print("WARNING: No trained model found. Using untrained model.")
            print(f"Train the model first: python -m backend.training.train_roberta")

        model = model.to(device)
        model.eval()

        _model_cache[model_key] = {"model": model, "device": device}

    return _model_cache[model_key]


def get_tokenizer():
    """Get tokenizer (cached)"""
    if "roberta" not in _tokenizer_cache:
        tokenizer = RoBERTaTokenizer(
            model_name=settings.ROBERTA_MODEL,
            max_length=settings.MAX_SEQ_LENGTH,
        )
        _tokenizer_cache["roberta"] = tokenizer

    return _tokenizer_cache["roberta"]


@router.post("/roberta", response_model=PredictionResponse)
async def predict_roberta(request: PredictionRequest):
    """
    Predict using RoBERTa baseline model

    This endpoint uses only the RoBERTa semantic model
    """
    try:
        # Preprocess
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.preprocess(request.review_text)

        if not preprocessor.is_valid_review(cleaned_text):
            raise HTTPException(status_code=400, detail="Invalid review text")

        # Get model and tokenizer
        model_data = get_roberta_model()
        model = model_data["model"]
        device = model_data["device"]
        tokenizer = get_tokenizer()

        # Tokenize
        encoded = tokenizer.encode_batch([cleaned_text])
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probas = torch.softmax(logits, dim=-1)

        # Get results
        fake_prob = float(probas[0, 1].cpu())
        genuine_prob = float(probas[0, 0].cpu())
        prediction = "FAKE" if fake_prob > 0.5 else "GENUINE"
        confidence = max(fake_prob, genuine_prob)

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            fake_probability=fake_prob,
            genuine_probability=genuine_prob,
            model_used="roberta_baseline",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full", response_model=PredictionResponse)
async def predict_full(request: PredictionRequest):
    """
    Predict using full EMFRD framework

    Currently falls back to RoBERTa baseline (other components in later phases)
    """
    # TODO: Implement full multimodal fusion in later phases
    # For now, use RoBERTa baseline
    return await predict_roberta(request)


@router.get("/models")
async def list_available_models():
    """List available trained models"""
    models_dir = settings.MODELS_DIR

    available = []
    for model_name in ["roberta_baseline", "roberta_contrastive", "hgnn", "gan", "fusion"]:
        model_dir = models_dir / model_name
        checkpoint_path = model_dir / "best.pt"

        if checkpoint_path.exists():
            # Read metadata
            metadata_path = model_dir / "best.json"
            metadata = {}
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)

            available.append({
                "name": model_name,
                "path": str(checkpoint_path),
                "trained": True,
                "metrics": metadata.get("metrics", {}),
            })
        else:
            available.append({
                "name": model_name,
                "trained": False,
            })

    return {"models": available}
