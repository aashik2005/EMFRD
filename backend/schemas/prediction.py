"""
Prediction schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class ModelType(str, Enum):
    """Available model types"""
    ROBERTA_BASELINE = "roberta_baseline"
    ROBERTA_CONTRASTIVE = "roberta_contrastive"
    HGNN = "hgnn"
    GAN_ADVERSARIAL = "gan_adversarial"
    FUSION = "fusion"
    FULL_EMFRD = "full_emfrd"


class PredictionRequest(BaseModel):
    """Request for fake review prediction"""
    review_text: str = Field(..., description="Review text to classify")
    model: ModelType = Field(default=ModelType.FULL_EMFRD, description="Model to use")
    user_id: Optional[str] = Field(None, description="User ID (for HGNN/fusion)")
    product_id: Optional[str] = Field(None, description="Product ID (for HGNN/fusion)")


class PredictionResponse(BaseModel):
    """Response from prediction"""
    prediction: Literal["FAKE", "GENUINE"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    fake_probability: float = Field(..., ge=0.0, le=1.0)
    genuine_probability: float = Field(..., ge=0.0, le=1.0)
    model_used: str

    # Optional additional info
    semantic_confidence: Optional[float] = None
    graph_confidence: Optional[float] = None
    adversarial_confidence: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "FAKE",
                "confidence": 0.94,
                "fake_probability": 0.94,
                "genuine_probability": 0.06,
                "model_used": "full_emfrd",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction"""
    reviews: list[str] = Field(..., description="List of review texts")
    model: ModelType = Field(default=ModelType.FULL_EMFRD)


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction"""
    predictions: list[PredictionResponse]
    total: int
