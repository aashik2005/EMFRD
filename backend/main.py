"""
EMFRD FastAPI Application
Main entry point for the backend API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.api.routes import prediction, experiments, datasets, metrics, explainability

# Create FastAPI app
app = FastAPI(
    title="EMFRD API",
    description="Explainable Multimodal Framework for Fake Review Detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(explainability.router, prefix="/api/explain", tags=["Explainability"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EMFRD API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": settings.DEVICE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
