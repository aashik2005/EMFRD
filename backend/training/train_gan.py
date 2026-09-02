"""
Training script for GAN Adversarial Model

Trains the Generator and Discriminator for adversarial robustness.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from backend.models import GANAdversarial
from backend.models.roberta_baseline import RoBERTaBaseline
from backend.data import get_dataset
from backend.config import settings
from backend.utils import get_device, set_seed, CheckpointManager
from backend.evaluation import MetricsCalculator


def train_gan(
    dataset_name: str = "fake_reviews",
    num_epochs: int = 20,
    batch_size: int = 32,
    latent_dim: int = 100,
    hidden_dim: int = 256,
    lr_generator: float = 0.0002,
    lr_discriminator: float = 0.0002,
    device: str = "auto",
    seed: int = 42,
    save_dir: str = None,
):
    """
    Train GAN Adversarial Model

    Args:
        dataset_name: Name of dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        latent_dim: Latent noise dimension
        hidden_dim: Hidden dimension
        lr_generator: Generator learning rate
        lr_discriminator: Discriminator learning rate
        device: Device for training
        seed: Random seed
        save_dir: Directory to save checkpoints
    """
    # Setup
    set_seed(seed)
    device = get_device(device)
    print(f"Using device: {device}")

    if save_dir is None:
        save_dir = settings.MODELS_DIR / "gan_adversarial"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    dataset = get_dataset(dataset_name)
    dataset.load()

    train_loader = DataLoader(
        dataset.splits["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset.splits["val"],
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Train samples: {len(dataset.splits['train'])}")
    print(f"Val samples: {len(dataset.splits['val'])}")

    # Load RoBERTa to extract embeddings
    print("\nLoading RoBERTa for embedding extraction...")
    roberta_model = RoBERTaBaseline(
        model_name=settings.ROBERTA_MODEL,
        num_labels=2,
    ).to(device)

    # Load trained RoBERTa if available
    roberta_checkpoint_dir = settings.MODELS_DIR / "roberta_baseline"
    roberta_checkpoint = CheckpointManager(roberta_checkpoint_dir, "roberta_baseline")

    if roberta_checkpoint.exists("best.pt"):
        roberta_checkpoint.load(roberta_model, "best.pt", device)
        print("Loaded trained RoBERTa embeddings")
    else:
        print("Warning: Using untrained RoBERTa embeddings")

    roberta_model.eval()

    # Initialize GAN
    print("\nInitializing GAN...")
    gan = GANAdversarial(
        embedding_dim=768,  # RoBERTa embedding size
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_labels=2,
    ).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(gan.generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(gan.discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))

    # Loss functions
    criterion_real_fake = nn.BCEWithLogitsLoss()
    criterion_classification = nn.CrossEntropyLoss()

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(save_dir, "gan_adversarial")

    # Training loop
    print(f"\nTraining GAN for {num_epochs} epochs...")

    best_val_acc = 0.0
    history = {
        "train_loss_G": [],
        "train_loss_D": [],
        "val_accuracy": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):
        # Training
        gan.train()
        roberta_model.eval()

        train_loss_G = 0.0
        train_loss_D = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_size = input_ids.size(0)

            # Extract real embeddings from RoBERTa
            with torch.no_grad():
                roberta_outputs = roberta_model(input_ids, attention_mask)
                real_embeddings = roberta_outputs["embeddings"]

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real embeddings
            real_outputs = gan.discriminator(real_embeddings)
            real_pred = real_outputs["real_fake_logits"]
            real_labels_disc = torch.ones(batch_size, 1, device=device)
            loss_D_real_fake = criterion_real_fake(real_pred, real_labels_disc)

            # Classification loss on real samples
            real_class_logits = real_outputs["class_logits"]
            loss_D_classification = criterion_classification(real_class_logits, labels)

            # Fake embeddings
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_embeddings = gan.generator(noise, labels)
            fake_outputs = gan.discriminator(fake_embeddings.detach())
            fake_pred = fake_outputs["real_fake_logits"]
            fake_labels_disc = torch.zeros(batch_size, 1, device=device)
            loss_D_fake = criterion_real_fake(fake_pred, fake_labels_disc)

            # Total discriminator loss
            loss_D = loss_D_real_fake + loss_D_fake + loss_D_classification
            loss_D.backward()
            optimizer_D.step()

            train_loss_D += loss_D.item()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Generate fake embeddings
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_embeddings = gan.generator(noise, labels)

            # Try to fool discriminator
            fake_outputs = gan.discriminator(fake_embeddings)
            fake_pred = fake_outputs["real_fake_logits"]
            real_labels_disc = torch.ones(batch_size, 1, device=device)  # Want to be classified as real
            loss_G_fool = criterion_real_fake(fake_pred, real_labels_disc)

            # Classification loss for generated samples
            fake_class_logits = fake_outputs["class_logits"]
            loss_G_classification = criterion_classification(fake_class_logits, labels)

            # Total generator loss
            loss_G = loss_G_fool + loss_G_classification
            loss_G.backward()
            optimizer_G.step()

            train_loss_G += loss_G.item()

            # Update progress bar
            progress_bar.set_postfix({
                "loss_G": f"{loss_G.item():.4f}",
                "loss_D": f"{loss_D.item():.4f}",
            })

        # Average losses
        avg_loss_G = train_loss_G / len(train_loader)
        avg_loss_D = train_loss_D / len(train_loader)

        history["train_loss_G"].append(avg_loss_G)
        history["train_loss_D"].append(avg_loss_D)

        # Validation
        gan.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Extract embeddings
                roberta_outputs = roberta_model(input_ids, attention_mask)
                embeddings = roberta_outputs["embeddings"]

                # Discriminator classification
                disc_outputs = gan.discriminator(embeddings)
                class_logits = disc_outputs["class_logits"]

                loss = criterion_classification(class_logits, labels)
                val_loss += loss.item()

                predictions = torch.argmax(class_logits, dim=-1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        # Calculate metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_all(val_labels_list, val_predictions)

        val_accuracy = metrics["accuracy"]
        history["val_accuracy"].append(val_accuracy)

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss G: {avg_loss_G:.4f}")
        print(f"  Train Loss D: {avg_loss_D:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val Precision: {metrics['precision']:.4f}")
        print(f"  Val Recall: {metrics['recall']:.4f}")
        print(f"  Val F1: {metrics['f1']:.4f}")

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint_manager.save(
                model=gan,
                optimizer=None,
                epoch=epoch,
                metrics=metrics,
                checkpoint_name="best.pt",
            )
            print(f"  ✓ Best model saved (accuracy: {val_accuracy:.4f})")

    # Save final results
    results = {
        "model": "gan_adversarial",
        "dataset": dataset_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "best_val_accuracy": best_val_acc,
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = settings.EXPERIMENTS_DIR / "results" / f"gan_adversarial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_dir}")
    print(f"Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAN Adversarial Model")
    parser.add_argument("--dataset", type=str, default="fake_reviews", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--lr-g", type=float, default=0.0002, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=0.0002, help="Discriminator learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_gan(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lr_generator=args.lr_g,
        lr_discriminator=args.lr_d,
        device=args.device,
        seed=args.seed,
    )
