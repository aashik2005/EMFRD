"""
Base model interface for EMFRD
All models should inherit from this class for consistency
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


class BaseFakeReviewModel(ABC, nn.Module):
    """
    Abstract base class for fake review detection models

    Provides consistent interface for:
    - Training
    - Evaluation
    - Prediction
    - Embedding extraction
    - Checkpoint management
    """

    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Returns:
            Dictionary containing:
                - logits: (batch_size, num_classes)
                - embeddings: (batch_size, hidden_dim)
                - loss: (optional) scalar loss
        """
        pass

    def predict(self, *args, **kwargs) -> np.ndarray:
        """
        Make predictions

        Returns:
            Predicted labels (0 or 1)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(*args, **kwargs)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            return predictions.cpu().numpy()

    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        """
        Predict class probabilities

        Returns:
            Class probabilities (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(*args, **kwargs)
            logits = outputs["logits"]
            probas = torch.softmax(logits, dim=-1)
            return probas.cpu().numpy()

    def get_embeddings(self, *args, **kwargs) -> np.ndarray:
        """
        Extract embeddings

        Returns:
            Embeddings (batch_size, hidden_dim)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(*args, **kwargs)
            embeddings = outputs.get("embeddings")
            if embeddings is None:
                raise ValueError("Model does not return embeddings")
            return embeddings.cpu().numpy()

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            "model_name": self.model_name,
            "num_parameters": self.count_parameters(),
        }
