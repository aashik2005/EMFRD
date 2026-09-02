"""
RoBERTa + Supervised Contrastive Learning Model

Enhances RoBERTa baseline with contrastive learning to improve
semantic representations by pulling same-class samples closer and
pushing different-class samples apart.

Architecture:
    Review text
    ↓
    RoBERTa encoder
    ↓
    [CLS] representation
    ├─────────────┬──────────────┐
    ↓             ↓              ↓
    Projection    Classification
    Head          Head
    ↓             ↓
    Contrastive   Classification
    Embedding     Logits
    ↓             ↓
    Contrastive   Cross-Entropy
    Loss          Loss
    └─────────────┴──────────────┘
              ↓
         Combined Loss
"""
import torch
import torch.nn as nn
from transformers import RobertaModel
from typing import Dict, Optional
from .base import BaseFakeReviewModel
from .losses import SupervisedContrastiveLoss


class RoBERTaContrastive(BaseFakeReviewModel):
    """
    RoBERTa with Supervised Contrastive Learning

    Combines classification objective with contrastive learning to
    learn better semantic representations.

    Args:
        model_name: Pretrained RoBERTa model name
        num_labels: Number of output classes (2 for binary)
        dropout: Dropout probability
        projection_dim: Dimension of contrastive projection head (default: 128)
        temperature: Temperature for contrastive loss (default: 0.07)
        contrastive_weight: Weight for contrastive loss (default: 0.2)
        freeze_encoder: Whether to freeze RoBERTa encoder
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        projection_dim: int = 128,
        temperature: float = 0.07,
        contrastive_weight: float = 0.2,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_prob = dropout
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

        # Load pretrained RoBERTa
        print(f"Loading {model_name}...")
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.config = self.roberta.config

        # Freeze encoder if requested
        if freeze_encoder:
            print("Freezing RoBERTa encoder")
            for param in self.roberta.parameters():
                param.requires_grad = False

        hidden_size = self.config.hidden_size

        # Classification head (same as baseline)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Projection head for contrastive learning
        # Two-layer MLP: hidden -> projection_dim -> projection_dim
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Contrastive loss
        self.contrastive_loss_fn = SupervisedContrastiveLoss(temperature=temperature)

        # Initialize weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        for layer in self.projection_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

        print(f"Model initialized: {self.count_parameters():,} trainable parameters")
        print(f"  Projection dim: {projection_dim}")
        print(f"  Temperature: {temperature}")
        print(f"  Contrastive weight: {contrastive_weight}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_contrastive: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) - optional for training
            return_contrastive: Whether to compute contrastive embeddings

        Returns:
            Dictionary containing:
                - logits: (batch_size, num_labels)
                - embeddings: (batch_size, hidden_size) - [CLS] representation
                - contrastive_embeddings: (batch_size, projection_dim) - for contrastive loss
                - loss: scalar (if labels provided)
                - classification_loss: scalar (if labels provided)
                - contrastive_loss: scalar (if labels provided and return_contrastive=True)
        """
        # RoBERTa encoding
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get [CLS] token representation
        # Shape: (batch_size, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Classification branch
        cls_output = self.dropout(cls_embedding)
        logits = self.classifier(cls_output)

        # Prepare output
        result = {
            "logits": logits,
            "embeddings": cls_embedding,  # For fusion model later
        }

        # Contrastive branch
        if return_contrastive:
            # Project to contrastive space
            # Shape: (batch_size, projection_dim)
            contrastive_embeddings = self.projection_head(cls_embedding)

            # L2 normalize for contrastive learning
            contrastive_embeddings = nn.functional.normalize(
                contrastive_embeddings, dim=1
            )

            result["contrastive_embeddings"] = contrastive_embeddings

        # Calculate losses if labels provided
        if labels is not None:
            # Classification loss
            classification_loss_fn = nn.CrossEntropyLoss()
            classification_loss = classification_loss_fn(logits, labels)

            result["classification_loss"] = classification_loss

            # Contrastive loss
            if return_contrastive and "contrastive_embeddings" in result:
                contrastive_loss = self.contrastive_loss_fn(
                    result["contrastive_embeddings"], labels
                )
                result["contrastive_loss"] = contrastive_loss

                # Combined loss
                total_loss = (
                    (1 - self.contrastive_weight) * classification_loss
                    + self.contrastive_weight * contrastive_loss
                )
            else:
                total_loss = classification_loss

            result["loss"] = total_loss

        return result

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "dropout": self.dropout_prob,
            "projection_dim": self.projection_dim,
            "temperature": self.temperature,
            "contrastive_weight": self.contrastive_weight,
            "hidden_size": self.config.hidden_size,
            "num_parameters": self.count_parameters(),
        }


class RoBERTaContrastiveTokenizer:
    """
    Wrapper for RoBERTa tokenizer (same as baseline)
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


# Testing
if __name__ == "__main__":
    print("Testing RoBERTa + Contrastive Learning Model...")

    # Create model
    model = RoBERTaContrastive(
        projection_dim=128, temperature=0.07, contrastive_weight=0.2
    )

    # Dummy input
    batch_size = 4
    seq_len = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 2, (batch_size,))

    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  labels: {labels.shape}")

    # Forward pass
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )

    print(f"\nOutput shapes:")
    print(f"  logits: {outputs['logits'].shape}")
    print(f"  embeddings: {outputs['embeddings'].shape}")
    print(f"  contrastive_embeddings: {outputs['contrastive_embeddings'].shape}")

    print(f"\nLosses:")
    print(f"  Total loss: {outputs['loss'].item():.4f}")
    print(f"  Classification loss: {outputs['classification_loss'].item():.4f}")
    print(f"  Contrastive loss: {outputs['contrastive_loss'].item():.4f}")

    print("\nTest passed!")
