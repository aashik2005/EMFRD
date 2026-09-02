"""
RoBERTa Baseline Model for Fake Review Detection

Simple but effective baseline using RoBERTa for classification
"""
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from typing import Dict, Optional
from .base import BaseFakeReviewModel


class RoBERTaBaseline(BaseFakeReviewModel):
    """
    RoBERTa-based fake review classifier

    Architecture:
        Review text
        ↓
        RoBERTa encoder
        ↓
        [CLS] token representation
        ↓
        Dropout
        ↓
        Linear classifier
        ↓
        Fake/Genuine (2 classes)
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        """
        Initialize RoBERTa baseline

        Args:
            model_name: Pretrained RoBERTa model name
            num_labels: Number of output classes (2 for binary)
            dropout: Dropout probability
            freeze_encoder: Whether to freeze RoBERTa encoder
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_prob = dropout

        # Load pretrained RoBERTa
        print(f"Loading {model_name}...")
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.config = self.roberta.config

        # Freeze encoder if requested (for faster training/less memory)
        if freeze_encoder:
            print("Freezing RoBERTa encoder")
            for param in self.roberta.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        print(f"Model initialized: {self.count_parameters():,} trainable parameters")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) - optional for training

        Returns:
            Dictionary containing:
                - logits: (batch_size, num_labels)
                - embeddings: (batch_size, hidden_size)
                - loss: scalar (if labels provided)
        """
        # RoBERTa encoding
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get [CLS] token representation (first token)
        # Shape: (batch_size, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Classification
        cls_output = self.dropout(cls_embedding)
        logits = self.classifier(cls_output)

        # Prepare output
        result = {
            "logits": logits,
            "embeddings": cls_embedding,  # For fusion model later
        }

        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result["loss"] = loss

        return result

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "dropout": self.dropout_prob,
            "hidden_size": self.config.hidden_size,
            "num_parameters": self.count_parameters(),
        }


class RoBERTaTokenizer:
    """
    Wrapper for RoBERTa tokenizer with dataset-specific preprocessing
    """

    def __init__(self, model_name: str = "roberta-base", max_length: int = 256):
        """
        Initialize tokenizer

        Args:
            model_name: Pretrained model name
            max_length: Maximum sequence length
        """
        from transformers import RobertaTokenizer

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode_batch(
        self,
        texts: list,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode batch of texts

        Args:
            texts: List of review texts
            return_tensors: Return format ("pt" for PyTorch)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

        return encoded

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
