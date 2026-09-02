"""
Metrics endpoints
"""
from fastapi import APIRouter, HTTPException
from backend.config import settings
from pathlib import Path
import json

router = APIRouter()


@router.get("/comparison")
async def get_model_comparison():
    """Get comparison of all models"""
    results_dir = settings.EXPERIMENTS_DIR / "results"

    if not results_dir.exists():
        return {"models": []}

    models_results = []

    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            test_metrics = data.get("final_test_results", {})
            if test_metrics:
                models_results.append({
                    "model": data.get("model", "unknown"),
                    "metrics": {
                        "accuracy": test_metrics.get("accuracy"),
                        "precision": test_metrics.get("precision"),
                        "recall": test_metrics.get("recall"),
                        "f1": test_metrics.get("f1"),
                        "roc_auc": test_metrics.get("roc_auc"),
                    },
                    "timestamp": data.get("timestamp"),
                })
        except Exception as e:
            continue

    return {"models": models_results}


@router.get("/paper_reference")
async def get_paper_reference():
    """Get paper reference results"""
    reference_file = Path("configs/paper_reference_results.json")

    if not reference_file.exists():
        raise HTTPException(status_code=404, detail="Paper reference results not found")

    with open(reference_file) as f:
        return json.load(f)
