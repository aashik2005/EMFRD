"""
GAN-based Adversarial Training for Fake Review Detection

Implements adversarial training to improve robustness against sophisticated
and AI-generated fake reviews.

Architecture:
    Generator: Creates synthetic adversarial review representations
    Discriminator: Detects fake reviews (including synthetic ones)

This improves the detector's robustness by training on adversarial examples.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .base import BaseFakeReviewModel


class Generator(nn.Module):
    """
    Generator network that creates synthetic fake review representations

    Takes noise + label as input and generates review-like embeddings
    that can fool the discriminator.

    Args:
        latent_dim: Dimension of input noise (default: 100)
        hidden_dim: Hidden layer dimension (default: 256)
        output_dim: Output embedding dimension (default: 768, for RoBERTa)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 768,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Generator network
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for label conditioning
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic review embeddings

        Args:
            noise: (batch_size, latent_dim) - random noise
            labels: (batch_size,) - target labels (0 or 1)

        Returns:
            (batch_size, output_dim) - synthetic embeddings
        """
        # Condition on labels
        labels_expanded = labels.unsqueeze(1).float()
        gen_input = torch.cat([noise, labels_expanded], dim=1)

        # Generate
        synthetic_embeddings = self.model(gen_input)

        return synthetic_embeddings


class Discriminator(nn.Module):
    """
    Discriminator network that detects fake reviews

    Takes review embeddings and classifies them as real/fake.
    Also outputs fake/genuine prediction for real reviews.

    Args:
        input_dim: Input embedding dimension (default: 768)
        hidden_dim: Hidden layer dimension (default: 256)
        num_labels: Number of output classes (default: 2)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_labels: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )

        # Real/Synthetic detector
        self.real_fake_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Fake/Genuine classifier (for real reviews)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, num_labels),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            embeddings: (batch_size, input_dim) - review embeddings
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - real_fake_prob: (batch_size, 1) - probability of being real
                - logits: (batch_size, num_labels) - fake/genuine logits
                - features: (batch_size, hidden_dim) - if return_features
        """
        # Extract features
        features = self.feature_extractor(embeddings)

        # Real/Synthetic detection
        real_fake_prob = self.real_fake_head(features)

        # Fake/Genuine classification
        logits = self.classifier(features)

        result = {
            "real_fake_prob": real_fake_prob,
            "logits": logits,
        }

        if return_features:
            result["features"] = features

        return result


class GANAdversarial(BaseFakeReviewModel):
    """
    GAN-based adversarial training framework

    Combines generator and discriminator for robust fake review detection.

    Args:
        latent_dim: Generator noise dimension (default: 100)
        hidden_dim: Hidden layer dimension (default: 256)
        embedding_dim: Review embedding dimension (default: 768)
        num_labels: Number of output classes (default: 2)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        hidden_dim: int = 256,
        embedding_dim: int = 768,
        num_labels: int = 2,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels

        # Generator
        self.generator = Generator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        )

        # Discriminator (detector)
        self.discriminator = Discriminator(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
        )

        print(f"GANAdversarial initialized:")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"  Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"  Total params: {self.count_parameters():,}")

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "discriminator",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            embeddings: (batch_size, embedding_dim) - real review embeddings
            labels: (batch_size,) - labels (optional)
            mode: "discriminator" or "generator"

        Returns:
            Dictionary with outputs depending on mode
        """
        if mode == "generator":
            return self._generator_forward(embeddings, labels)
        else:
            return self._discriminator_forward(embeddings, labels)

    def _generator_forward(
        self,
        real_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generator forward pass"""
        batch_size = real_embeddings.size(0)
        device = real_embeddings.device

        # Sample noise
        noise = torch.randn(batch_size, self.latent_dim, device=device)

        # Generate synthetic embeddings (fake reviews)
        synthetic_embeddings = self.generator(noise, labels)

        # Try to fool discriminator
        disc_output = self.discriminator(synthetic_embeddings)

        return {
            "synthetic_embeddings": synthetic_embeddings,
            "real_fake_prob": disc_output["real_fake_prob"],
        }

    def _discriminator_forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Discriminator forward pass"""
        # Classify real embeddings
        disc_output = self.discriminator(embeddings, return_features=True)

        result = {
            "logits": disc_output["logits"],
            "embeddings": disc_output["features"],
            "real_fake_prob": disc_output["real_fake_prob"],
        }

        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(disc_output["logits"], labels)
            result["loss"] = loss

        return result

    def generate_adversarial_samples(
        self,
        num_samples: int,
        target_label: int = 1,  # Generate fake reviews
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate adversarial samples

        Args:
            num_samples: Number of samples to generate
            target_label: Target label (0=genuine, 1=fake)
            device: Device to generate on

        Returns:
            (num_samples, embedding_dim) - synthetic embeddings
        """
        if device is None:
            device = next(self.parameters()).device

        self.generator.eval()

        with torch.no_grad():
            # Sample noise
            noise = torch.randn(num_samples, self.latent_dim, device=device)
            labels = torch.full((num_samples,), target_label, device=device, dtype=torch.long)

            # Generate
            synthetic_embeddings = self.generator(noise, labels)

        return synthetic_embeddings

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            "model_name": "GANAdversarial",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "num_labels": self.num_labels,
            "num_parameters": self.count_parameters(),
        }


# Testing
if __name__ == "__main__":
    print("Testing GAN Adversarial Model...")

    # Create model
    model = GANAdversarial(
        latent_dim=100,
        hidden_dim=256,
        embedding_dim=768,
    )

    # Dummy input
    batch_size = 4
    embeddings = torch.randn(batch_size, 768)
    labels = torch.randint(0, 2, (batch_size,))

    print(f"\nInput shapes:")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Labels: {labels.shape}")

    # Test discriminator
    print("\nTesting Discriminator...")
    disc_output = model(embeddings, labels, mode="discriminator")
    print(f"  Logits: {disc_output['logits'].shape}")
    print(f"  Features: {disc_output['embeddings'].shape}")
    print(f"  Real/Fake prob: {disc_output['real_fake_prob'].shape}")
    print(f"  Loss: {disc_output['loss'].item():.4f}")

    # Test generator
    print("\nTesting Generator...")
    gen_output = model(embeddings, labels, mode="generator")
    print(f"  Synthetic embeddings: {gen_output['synthetic_embeddings'].shape}")
    print(f"  Real/Fake prob: {gen_output['real_fake_prob'].shape}")

    # Test generation
    print("\nTesting Adversarial Generation...")
    synthetic = model.generate_adversarial_samples(num_samples=10, target_label=1)
    print(f"  Generated: {synthetic.shape}")

    print("\nTest passed!")
