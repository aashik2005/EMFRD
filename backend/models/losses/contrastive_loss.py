"""
Supervised Contrastive Loss for EMFRD

Based on: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
https://arxiv.org/abs/2004.11362

This loss function encourages representations of samples with the same label
to be close together, while pushing apart representations of samples with
different labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss

    Given a batch of samples with labels, this loss:
    1. Treats samples with the same label as positive pairs
    2. Treats samples with different labels as negative pairs
    3. Pulls positive pairs closer in the embedding space
    4. Pushes negative pairs apart

    Args:
        temperature: Temperature parameter for scaling (default: 0.07)
        base_temperature: Base temperature for normalization (default: 0.07)

    Shape:
        - features: (batch_size, embedding_dim) - L2 normalized embeddings
        - labels: (batch_size,) - Class labels
        - mask: (batch_size, batch_size) - Optional mask for valid pairs

    Returns:
        Scalar loss value
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss

        Args:
            features: Embeddings (batch_size, embedding_dim), should be L2 normalized
            labels: Class labels (batch_size,)
            mask: Optional mask for valid pairs (batch_size, batch_size)

        Returns:
            Contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]

        # Ensure features are normalized
        features = F.normalize(features, dim=1)

        # Labels should be (batch_size, 1) for broadcasting
        labels = labels.contiguous().view(-1, 1)

        # Create mask for positive pairs (same label)
        # Shape: (batch_size, batch_size)
        # mask[i, j] = 1 if labels[i] == labels[j], else 0
        labels_mask = torch.eq(labels, labels.T).float().to(device)

        # Apply external mask if provided
        if mask is not None:
            labels_mask = labels_mask * mask

        # Compute similarity matrix (dot product of normalized features)
        # Shape: (batch_size, batch_size)
        similarity_matrix = torch.matmul(features, features.T)

        # For numerical stability, subtract max
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Scale by temperature
        logits = logits / self.temperature

        # Create mask to exclude self-comparisons (diagonal)
        # Shape: (batch_size, batch_size)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        labels_mask = labels_mask.masked_fill(self_mask, 0)

        # Compute log probabilities
        # exp_logits: (batch_size, batch_size)
        exp_logits = torch.exp(logits)

        # Mask out self-comparisons from denominator
        exp_logits = exp_logits.masked_fill(self_mask, 0)

        # Sum over all samples (excluding self) for denominator
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        # Compute mean of log-likelihood over positive pairs
        # Only consider samples that have at least one positive pair
        mask_sum = labels_mask.sum(dim=1)

        # Avoid division by zero
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (labels_mask * log_prob).sum(dim=1) / mask_sum

        # Loss is negative mean log-likelihood
        loss = -mean_log_prob_pos

        # Average over batch
        loss = loss.mean()

        # Scale by temperature ratio (from original paper)
        loss = (self.temperature / self.base_temperature) * loss

        return loss


class ContrastiveLearningMetrics:
    """
    Utility class to compute contrastive learning metrics
    """

    @staticmethod
    def compute_alignment(features: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute alignment metric: average similarity between positive pairs

        Higher is better (embeddings of same class are more similar)
        """
        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)

        # Positive pair mask
        pos_mask = torch.eq(labels, labels.T).float()

        # Exclude self-comparisons
        batch_size = features.shape[0]
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        pos_mask = pos_mask.masked_fill(self_mask, 0)

        # Compute similarities
        similarity_matrix = torch.matmul(features, features.T)

        # Average similarity over positive pairs
        if pos_mask.sum() > 0:
            alignment = (pos_mask * similarity_matrix).sum() / pos_mask.sum()
            return alignment.item()
        return 0.0

    @staticmethod
    def compute_uniformity(features: torch.Tensor) -> float:
        """
        Compute uniformity metric: how uniformly distributed embeddings are

        Lower is better (embeddings are more uniformly distributed on hypersphere)
        """
        features = F.normalize(features, dim=1)

        # Compute pairwise squared distances
        # ||f_i - f_j||^2 = 2 - 2 * f_i · f_j
        similarity_matrix = torch.matmul(features, features.T)
        sq_dist = 2 - 2 * similarity_matrix

        # Average over all pairs (excluding diagonal)
        batch_size = features.shape[0]
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)

        uniformity = torch.exp(-2 * sq_dist[mask]).mean()
        return torch.log(uniformity).item()

    @staticmethod
    def compute_metrics(features: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Compute all contrastive learning metrics

        Returns:
            Dictionary with alignment and uniformity metrics
        """
        return {
            "alignment": ContrastiveLearningMetrics.compute_alignment(features, labels),
            "uniformity": ContrastiveLearningMetrics.compute_uniformity(features),
        }


# Example usage and testing
if __name__ == "__main__":
    # Test contrastive loss
    print("Testing Supervised Contrastive Loss...")

    # Create dummy data
    batch_size = 8
    embedding_dim = 128
    num_classes = 2

    # Random embeddings (will be normalized)
    features = torch.randn(batch_size, embedding_dim)

    # Random labels (half fake, half genuine)
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"Features shape: {features.shape}")
    print(f"Labels: {labels}")

    # Compute loss
    criterion = SupervisedContrastiveLoss(temperature=0.07)
    loss = criterion(features, labels)

    print(f"Contrastive Loss: {loss.item():.4f}")

    # Compute metrics
    metrics = ContrastiveLearningMetrics.compute_metrics(features, labels)
    print(f"Alignment: {metrics['alignment']:.4f}")
    print(f"Uniformity: {metrics['uniformity']:.4f}")

    print("\nTest passed!")
