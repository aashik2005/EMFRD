"""
Dataset endpoints
"""
from fastapi import APIRouter, HTTPException
from backend.schemas import DatasetValidation
from backend.data import get_dataset
from backend.config import settings

router = APIRouter()


@router.get("/")
async def list_datasets():
    """List available datasets"""
    from backend.data import DatasetRegistry

    return {
        "datasets": DatasetRegistry.list_datasets(),
        "primary": settings.PRIMARY_DATASET,
    }


@router.post("/validate")
async def validate_dataset(dataset_name: str):
    """Validate dataset and return statistics"""
    try:
        dataset = get_dataset(
            dataset_name,
            data_dir=settings.DATA_DIR / "raw" / dataset_name,
            cache_dir=settings.CACHE_DIR,
        )

        records, info = dataset.prepare()

        return DatasetValidation(**info.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
