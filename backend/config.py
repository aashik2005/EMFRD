"""
Configuration management for EMFRD project
"""
import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Global settings for EMFRD"""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = Field(default_factory=lambda: Path("./data"))
    MODELS_DIR: Path = Field(default_factory=lambda: Path("./models"))
    EXPERIMENTS_DIR: Path = Field(default_factory=lambda: Path("./experiments"))
    CACHE_DIR: Path = Field(default_factory=lambda: Path("./data/cache"))

    # Training
    DEVICE: Literal["cuda", "cpu"] = "cuda"
    MIXED_PRECISION: bool = True
    BATCH_SIZE: int = 8
    MAX_EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5
    RANDOM_SEED: int = 42
    GRADIENT_ACCUMULATION_STEPS: int = 1
    WARMUP_STEPS: int = 100
    WEIGHT_DECAY: float = 0.01
    MAX_GRAD_NORM: float = 1.0

    # Model
    ROBERTA_MODEL: str = "roberta-base"
    MAX_SEQ_LENGTH: int = 256
    DROPOUT: float = 0.1

    # Dataset
    PRIMARY_DATASET: str = "fake_reviews"
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    MIN_REVIEW_LENGTH: int = 10
    MAX_REVIEW_LENGTH: int = 512

    # Contrastive Learning
    CONTRASTIVE_TEMPERATURE: float = 0.07
    CONTRASTIVE_WEIGHT: float = 0.2
    PROJECTION_DIM: int = 128

    # HGNN
    HGNN_HIDDEN_DIM: int = 128
    HGNN_LAYERS: int = 2
    HGNN_DROPOUT: float = 0.2

    # GAN
    GAN_LATENT_DIM: int = 100
    GAN_ENABLED: bool = True

    # Fusion
    FUSION_HIDDEN_DIM: int = 256
    FUSION_DROPOUT: float = 0.2

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]

    # Logging
    LOG_LEVEL: str = "INFO"
    TENSORBOARD_ENABLED: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
