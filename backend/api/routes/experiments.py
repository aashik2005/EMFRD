"""
Experiments endpoints
"""
from fastapi import APIRouter, HTTPException
from backend.schemas import ExperimentInfo, ExperimentResults
from backend.config import settings
from pathlib import Path
import json
from typing import List

router = APIRouter()


@router.get("/", response_model=List[ExperimentInfo])
async def list_experiments():
    """List all experiments"""
    results_dir = settings.EXPERIMENTS_DIR / "results"

    if not results_dir.exists():
        return []

    experiments = []
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            experiments.append(ExperimentInfo(
                experiment_id=data.get("experiment_id", result_file.stem),
                model=data.get("model", "unknown"),
                dataset=data.get("dataset", "unknown"),
                status="completed",
                created_at=data.get("timestamp", ""),
                completed_at=data.get("timestamp", ""),
            ))
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
            continue

    return experiments


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details"""
    results_dir = settings.EXPERIMENTS_DIR / "results"

    # Find experiment file
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            if data.get("experiment_id") == experiment_id:
                return data
        except Exception as e:
            continue

    raise HTTPException(status_code=404, detail="Experiment not found")
