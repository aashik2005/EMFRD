"""
Experiment schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class ExperimentInfo(BaseModel):
    """Experiment information"""
    experiment_id: str
    model: str
    dataset: str
    status: str  # pending, running, completed, failed
    created_at: str
    completed_at: Optional[str] = None


class MetricsData(BaseModel):
    """Model metrics"""
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    roc_auc: Optional[float] = Field(None, ge=0.0, le=1.0)


class ExperimentResults(BaseModel):
    """Experiment results"""
    experiment_id: str
    model: str
    dataset: str
    metrics: MetricsData
    confusion_matrix: Optional[Dict[str, int]] = None
    config: Optional[Dict[str, Any]] = None
    timestamp: str


class ModelComparison(BaseModel):
    """Model comparison data"""
    model_name: str
    our_metrics: MetricsData
    paper_metrics: Optional[MetricsData] = None
