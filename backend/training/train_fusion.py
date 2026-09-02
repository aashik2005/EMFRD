"""
Training script for Gated Multimodal Fusion Model (Complete EMFRD)

Trains the complete EMFRD framework combining all modalities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional

from backend.models import (
    GatedMultimodalFusion,
    RoBERTaContrastive,
    GANAdversarial,
)
from backend.data import get_dataset
from backend.config import settings
from backend.utils import get_device, set_seed, CheckpointManager
from backend.evaluation import MetricsCalculator


def extract_features(
    batch: dict,
    semantic_model: RoBERTaContrastive,
    gan_model: Optional[GANAdversarial],
    device: str,
) -> dict:
    """
    Extract features from all modalities

    Args:
        batch: Data batch
        semantic_model: RoBERTa contrastive model
        gan_model: GAN model (optional)
        device: Computation device

    Returns:
        Dictionary with features from all modalities
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    features = {}

    # Semantic features from RoBERTa + Contrastive
    with torch.no_grad():
        semantic_outputs = semantic_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_contrastive=True,
        )
        features["semantic"] = semantic_outputs["embeddings"]  # (batch, 768)

    # Graph features (placeholder - requires HGNN)
    # In production, extract from HGNN model
    features["graph"] = None

    # Adversarial features from GAN discriminator
    if gan_model is not None:
        with torch.no_grad():
            disc_outputs = gan_model.discriminator(features["semantic"])
            # Use discriminator's hidden representation
            features["adversarial"] = disc_outputs.get("hidden", None)

    if features["adversarial"] is None:
        # Placeholder if GAN not available
        batch_size = input_ids.size(0)
        features["adversarial"] = torch.randn(batch_size, 256, device=device)

    # Metadata features (if available in batch)
    metadata_list = []
    for item in batch.get("metadata", [{}] * input_ids.size(0)):
        # Extract metadata: [rating, verified, helpful_count, ...]
        rating = item.get("rating", 0.0) / 5.0  # Normalize to [0, 1]
        verified = float(item.get("verified", 0))

        # Create 8-dimensional metadata vector
        meta_vec = [
            rating,
            verified,
            0.0,  # Placeholder for other metadata
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        metadata_list.append(meta_vec)

    features["metadata"] = torch.tensor(metadata_list, dtype=torch.float32, device=device)

    return features


def train_fusion(
    dataset_name: str = "fake_reviews",
    num_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    device: str = "auto",
    seed: int = 42,
    save_dir: str = None,
    load_components: bool = True,
):
    """
    Train Gated Multimodal Fusion Model

    Args:
        dataset_name: Name of dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim: Fusion hidden dimension
        dropout: Dropout probability
        device: Device for training
        seed: Random seed
        save_dir: Directory to save checkpoints
        load_components: Whether to load pre-trained component models
    """
    # Setup
    set_seed(seed)
    device = get_device(device)
    print(f"Using device: {device}")

    if save_dir is None:
        save_dir = settings.MODELS_DIR / "fusion"
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

    # Load component models
    print("\n" + "="*60)
    print("Loading Component Models")
    print("="*60)

    # 1. Semantic Model (RoBERTa + Contrastive)
    print("\n1. Loading Semantic Model (RoBERTa + Contrastive)...")
    semantic_model = RoBERTaContrastive(
        model_name=settings.ROBERTA_MODEL,
        num_labels=2,
        projection_dim=settings.PROJECTION_DIM,
    ).to(device)

    if load_components:
        semantic_checkpoint_dir = settings.MODELS_DIR / "roberta_contrastive"
        semantic_checkpoint = CheckpointManager(semantic_checkpoint_dir, "roberta_contrastive")

        if semantic_checkpoint.exists("best.pt"):
            semantic_checkpoint.load(semantic_model, "best.pt", device)
            print("   ✓ Loaded trained contrastive model")
        else:
            print("   ⚠ Warning: Using untrained contrastive model")

    semantic_model.eval()  # Freeze during fusion training

    # 2. GAN Model (optional)
    print("\n2. Loading Adversarial Model (GAN)...")
    gan_model = None

    if load_components:
        gan_checkpoint_dir = settings.MODELS_DIR / "gan_adversarial"
        gan_checkpoint = CheckpointManager(gan_checkpoint_dir, "gan_adversarial")

        if gan_checkpoint.exists("best.pt"):
            gan_model = GANAdversarial(
                embedding_dim=768,
                latent_dim=100,
                hidden_dim=256,
            ).to(device)
            gan_checkpoint.load(gan_model, "best.pt", device)
            gan_model.eval()
            print("   ✓ Loaded trained GAN model")
        else:
            print("   ⚠ GAN model not found, using placeholder adversarial features")
    else:
        print("   ⚠ Skipping GAN loading (load_components=False)")

    # 3. Initialize Fusion Model
    print("\n3. Initializing Fusion Model...")
    fusion_model = GatedMultimodalFusion(
        semantic_dim=768,
        graph_dim=128,
        adversarial_dim=256,
        metadata_dim=8,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_labels=2,
    ).to(device)

    print(f"   ✓ Fusion model initialized ({fusion_model.count_parameters():,} parameters)")

    # Optimizer
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(save_dir, "fusion")

    # Training loop
    print("\n" + "="*60)
    print(f"Training Fusion Model for {num_epochs} epochs")
    print("="*60)

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    for epoch in range(num_epochs):
        # Training
        fusion_model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            labels = batch["labels"].to(device)

            # Extract features from all modalities
            features = extract_features(batch, semantic_model, gan_model, device)

            # Forward pass through fusion model
            outputs = fusion_model(
                semantic_features=features["semantic"],
                graph_features=features["graph"],
                adversarial_features=features["adversarial"],
                metadata_features=features["metadata"],
                labels=labels,
            )

            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            predictions = torch.argmax(outputs["logits"], dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{train_correct/train_total:.4f}",
            })

        # Average metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(train_accuracy)

        # Validation
        fusion_model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(device)

                # Extract features
                features = extract_features(batch, semantic_model, gan_model, device)

                # Forward pass
                outputs = fusion_model(
                    semantic_features=features["semantic"],
                    graph_features=features["graph"],
                    adversarial_features=features["adversarial"],
                    metadata_features=features["metadata"],
                    labels=labels,
                )

                loss = outputs["loss"]
                val_loss += loss.item()

                predictions = torch.argmax(outputs["logits"], dim=-1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        # Calculate metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_all(val_labels_list, val_predictions)

        val_accuracy = metrics["accuracy"]
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(metrics["f1"])

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        print(f"  Val Precision: {metrics['precision']:.4f}")
        print(f"  Val Recall: {metrics['recall']:.4f}")
        print(f"  Val F1: {metrics['f1']:.4f}")

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint_manager.save(
                model=fusion_model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                checkpoint_name="best.pt",
            )
            print(f"  ✓ Best model saved (accuracy: {val_accuracy:.4f})")

    # Save final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    # Compare with paper reference
    paper_reference = {
        "accuracy": 0.978,
        "precision": 0.975,
        "recall": 0.976,
        "f1": 0.976,
    }

    print("\nPaper Reference vs. Reproduction:")
    print("-" * 60)
    print(f"{'Metric':<15} {'Paper':<15} {'Reproduced':<15} {'Diff':<15}")
    print("-" * 60)

    final_metrics = {
        "accuracy": val_accuracy,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }

    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        paper_val = paper_reference[metric_name]
        repro_val = final_metrics[metric_name]
        diff = repro_val - paper_val
        print(f"{metric_name:<15} {paper_val:<15.4f} {repro_val:<15.4f} {diff:+.4f}")

    results = {
        "model": "fusion",
        "dataset": dataset_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "best_val_accuracy": best_val_acc,
        "final_metrics": final_metrics,
        "paper_reference": paper_reference,
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = settings.EXPERIMENTS_DIR / "results" / f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to: {save_dir}")
    print(f"Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gated Multimodal Fusion Model")
    parser.add_argument("--dataset", type=str, default="fake_reviews", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Fusion hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-load-components", action="store_true", help="Don't load pre-trained components")

    args = parser.parse_args()

    train_fusion(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
        load_components=not args.no_load_components,
    )
