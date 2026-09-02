"""
Gated Multimodal Fusion for EMFRD

Combines multiple modalities with learned gating:
1. Semantic: RoBERTa + Contrastive Learning
2. Behavioral: HGNN (graph-based)
3. Adversarial: GAN-based robustness
4. Metadata: ratings, verified purchase, etc.

Architecture:
    Review → Multiple Modalities → Gating Network → Weighted Fusion → Classifier
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from .base import BaseFakeReviewModel


class GatingNetwork(nn.Module):
    """
    Learned gating mechanism for multimodal fusion

    Computes attention-style gates for each modality based on the input.

    Args:
        semantic_dim: Semantic feature dimension (default: 768)
        graph_dim: Graph feature dimension (default: 128)
        adversarial_dim: Adversarial feature dimension (default: 256)
        metadata_dim: Metadata feature dimension (default: 8)
        gate_hidden_dim: Hidden dimension for gate computation (default: 128)
    """

    def __init__(
        self,
        semantic_dim: int = 768,
        graph_dim: int = 128,
        adversarial_dim: int = 256,
        metadata_dim: int = 8,
        gate_hidden_dim: int = 128,
    ):
        super().__init__()

        # Gate networks for each modality
        self.semantic_gate = nn.Sequential(
            nn.Linear(semantic_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )

        self.graph_gate = nn.Sequential(
            nn.Linear(graph_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )

        self.adversarial_gate = nn.Sequential(
            nn.Linear(adversarial_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )

        self.metadata_gate = nn.Sequential(
            nn.Linear(metadata_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )

    def forward(
        self,
        semantic_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        adversarial_features: Optional[torch.Tensor] = None,
        metadata_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gates for available modalities

        Returns:
            Dictionary with normalized gates for each modality
        """
        gates = []
        gate_dict = {}

        # Compute gates for available modalities
        if semantic_features is not None:
            gate_semantic = self.semantic_gate(semantic_features)
            gates.append(gate_semantic)
            gate_dict["semantic"] = gate_semantic

        if graph_features is not None:
            gate_graph = self.graph_gate(graph_features)
            gates.append(gate_graph)
            gate_dict["graph"] = gate_graph

        if adversarial_features is not None:
            gate_adversarial = self.adversarial_gate(adversarial_features)
            gates.append(gate_adversarial)
            gate_dict["adversarial"] = gate_adversarial

        if metadata_features is not None:
            gate_metadata = self.metadata_gate(metadata_features)
            gates.append(gate_metadata)
            gate_dict["metadata"] = gate_metadata

        # Normalize gates with softmax
        if len(gates) > 0:
            all_gates = torch.cat(gates, dim=1)
            normalized_gates = torch.softmax(all_gates, dim=1)

            # Split normalized gates
            result = {}
            idx = 0
            for key in gate_dict.keys():
                result[key] = normalized_gates[:, idx:idx+1]
                idx += 1

            return result
        else:
            return {}


class GatedMultimodalFusion(BaseFakeReviewModel):
    """
    Gated Multimodal Fusion Model (Complete EMFRD Framework)

    Combines semantic, behavioral, and adversarial features with learned gating.

    Args:
        semantic_dim: Semantic feature dimension (default: 768)
        graph_dim: Graph feature dimension (default: 128)
        adversarial_dim: Adversarial feature dimension (default: 256)
        metadata_dim: Metadata feature dimension (default: 8)
        hidden_dim: Fusion hidden dimension (default: 256)
        dropout: Dropout probability (default: 0.2)
        num_labels: Number of output classes (default: 2)
    """

    def __init__(
        self,
        semantic_dim: int = 768,
        graph_dim: int = 128,
        adversarial_dim: int = 256,
        metadata_dim: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_labels: int = 2,
    ):
        super().__init__()

        self.semantic_dim = semantic_dim
        self.graph_dim = graph_dim
        self.adversarial_dim = adversarial_dim
        self.metadata_dim = metadata_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        self.num_labels = num_labels

        # Gating network
        self.gating = GatingNetwork(
            semantic_dim=semantic_dim,
            graph_dim=graph_dim,
            adversarial_dim=adversarial_dim,
            metadata_dim=metadata_dim,
        )

        # Projection layers to common dimension
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.adversarial_proj = nn.Linear(adversarial_dim, hidden_dim)
        self.metadata_proj = nn.Linear(metadata_dim, hidden_dim)

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim // 2, num_labels)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"GatedMultimodalFusion initialized:")
        print(f"  Semantic dim: {semantic_dim}")
        print(f"  Graph dim: {graph_dim}")
        print(f"  Adversarial dim: {adversarial_dim}")
        print(f"  Metadata dim: {metadata_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Total params: {self.count_parameters():,}")

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        semantic_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        adversarial_features: Optional[torch.Tensor] = None,
        metadata_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multimodal fusion

        Args:
            semantic_features: (batch_size, semantic_dim) - from RoBERTa
            graph_features: (batch_size, graph_dim) - from HGNN
            adversarial_features: (batch_size, adversarial_dim) - from GAN
            metadata_features: (batch_size, metadata_dim) - metadata
            labels: (batch_size,) - optional labels for training

        Returns:
            Dictionary containing:
                - logits: (batch_size, num_labels)
                - embeddings: (batch_size, hidden_dim // 2)
                - gates: dict with gate values for each modality
                - loss: scalar (if labels provided)
        """
        batch_size = (
            semantic_features.size(0) if semantic_features is not None
            else graph_features.size(0) if graph_features is not None
            else adversarial_features.size(0) if adversarial_features is not None
            else metadata_features.size(0)
        )

        # Compute gates
        gates = self.gating(
            semantic_features=semantic_features,
            graph_features=graph_features,
            adversarial_features=adversarial_features,
            metadata_features=metadata_features,
        )

        # Project to common dimension
        projected = []
        gate_values = []

        if semantic_features is not None and "semantic" in gates:
            semantic_proj = self.semantic_proj(semantic_features)
            projected.append(semantic_proj)
            gate_values.append(gates["semantic"])

        if graph_features is not None and "graph" in gates:
            graph_proj = self.graph_proj(graph_features)
            projected.append(graph_proj)
            gate_values.append(gates["graph"])

        if adversarial_features is not None and "adversarial" in gates:
            adversarial_proj = self.adversarial_proj(adversarial_features)
            projected.append(adversarial_proj)
            gate_values.append(gates["adversarial"])

        if metadata_features is not None and "metadata" in gates:
            metadata_proj = self.metadata_proj(metadata_features)
            projected.append(metadata_proj)
            gate_values.append(gates["metadata"])

        # Gated fusion
        if len(projected) == 0:
            raise ValueError("At least one modality must be provided")

        # Weighted combination
        fused = torch.zeros(batch_size, self.hidden_dim, device=projected[0].device)
        for feat, gate in zip(projected, gate_values):
            fused = fused + gate * feat

        # Fusion layers
        fused_features = self.fusion(fused)

        # Classification
        logits = self.classifier(fused_features)

        # Prepare output
        result = {
            "logits": logits,
            "embeddings": fused_features,
            "gates": gates,
        }

        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result["loss"] = loss

        return result

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            "model_name": "GatedMultimodalFusion",
            "semantic_dim": self.semantic_dim,
            "graph_dim": self.graph_dim,
            "adversarial_dim": self.adversarial_dim,
            "metadata_dim": self.metadata_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_prob,
            "num_labels": self.num_labels,
            "num_parameters": self.count_parameters(),
        }


# Testing
if __name__ == "__main__":
    print("Testing Gated Multimodal Fusion...")

    # Create model
    model = GatedMultimodalFusion(
        semantic_dim=768,
        graph_dim=128,
        adversarial_dim=256,
        metadata_dim=8,
        hidden_dim=256,
    )

    # Test with all modalities
    print("\nTest 1: All modalities available")
    batch_size = 4
    semantic_feats = torch.randn(batch_size, 768)
    graph_feats = torch.randn(batch_size, 128)
    adversarial_feats = torch.randn(batch_size, 256)
    metadata_feats = torch.randn(batch_size, 8)
    labels = torch.randint(0, 2, (batch_size,))

    outputs = model(
        semantic_features=semantic_feats,
        graph_features=graph_feats,
        adversarial_features=adversarial_feats,
        metadata_features=metadata_feats,
        labels=labels,
    )

    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Embeddings: {outputs['embeddings'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Gates: {list(outputs['gates'].keys())}")
    for key, gate in outputs['gates'].items():
        print(f"    {key}: {gate.mean().item():.4f}")

    # Test with missing modalities
    print("\nTest 2: Semantic + Metadata only")
    outputs = model(
        semantic_features=semantic_feats,
        metadata_features=metadata_feats,
        labels=labels,
    )

    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Gates: {list(outputs['gates'].keys())}")
    for key, gate in outputs['gates'].items():
        print(f"    {key}: {gate.mean().item():.4f}")

    print("\nTest passed!")
