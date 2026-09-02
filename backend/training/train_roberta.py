"""
Training script for RoBERTa baseline model

Usage:
    python -m backend.training.train_roberta
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
from datetime import datetime
import argparse

from backend.config import settings
from backend.data import get_dataset
from backend.preprocessing import TextPreprocessor, DataSplitter
from backend.models import RoBERTaBaseline
from backend.models.roberta_baseline import RoBERTaTokenizer
from backend.evaluation import MetricsCalculator
from backend.utils import get_device, set_seed, CheckpointManager


class ReviewDataset(Dataset):
    """PyTorch Dataset for reviews"""

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoded = self.tokenizer.encode_batch(
            [text],
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            probas = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probas.extend(probas.cpu().numpy())

    # Calculate metrics
    import numpy as np
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)

    results = MetricsCalculator.calculate(all_labels, all_preds, all_probas)
    return results


def main(args):
    """Main training function"""
    print("="*80)
    print("EMFRD - RoBERTa Baseline Training")
    print("="*80)

    # Set seed for reproducibility
    set_seed(settings.RANDOM_SEED)

    # Get device
    device = get_device(settings.DEVICE)

    # Load dataset
    print(f"\nLoading {settings.PRIMARY_DATASET} dataset...")
    dataset = get_dataset(
        settings.PRIMARY_DATASET,
        data_dir=settings.DATA_DIR / "raw" / settings.PRIMARY_DATASET,
        cache_dir=settings.CACHE_DIR,
    )

    records, dataset_info = dataset.prepare()
    print(f"\nDataset info:")
    for key, value in dataset_info.to_dict().items():
        print(f"  {key}: {value}")

    # Preprocess
    print("\nPreprocessing...")
    preprocessor = TextPreprocessor()
    texts = [preprocessor.preprocess(r.review_text) for r in records]
    labels = [r.label for r in records]

    # Filter valid reviews
    texts, labels = preprocessor.filter_valid_reviews(
        texts,
        labels,
        min_length=settings.MIN_REVIEW_LENGTH,
        max_length=settings.MAX_REVIEW_LENGTH,
    )

    # Split data
    print("\nSplitting data...")
    splitter = DataSplitter(
        train_ratio=settings.TRAIN_RATIO,
        val_ratio=settings.VAL_RATIO,
        test_ratio=settings.TEST_RATIO,
        random_seed=settings.RANDOM_SEED,
    )

    user_ids = [r.user_id for r in records] if records and records[0].user_id else None
    product_ids = [r.product_id for r in records] if records and records[0].product_id else None

    train_data, val_data, test_data = splitter.split(texts, labels, user_ids, product_ids)

    # Save splits
    splits_dir = settings.DATA_DIR / "splits" / settings.PRIMARY_DATASET
    splitter.save_splits(train_data, val_data, test_data, splits_dir)

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = RoBERTaTokenizer(
        model_name=settings.ROBERTA_MODEL,
        max_length=settings.MAX_SEQ_LENGTH,
    )

    # Create datasets
    train_dataset = ReviewDataset(train_data["texts"], train_data["labels"], tokenizer)
    val_dataset = ReviewDataset(val_data["texts"], val_data["labels"], tokenizer)
    test_dataset = ReviewDataset(test_data["texts"], test_data["labels"], tokenizer)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows compatibility
    )
    val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, num_workers=0)

    # Initialize model
    print("\nInitializing model...")
    model = RoBERTaBaseline(
        model_name=settings.ROBERTA_MODEL,
        num_labels=2,
        dropout=settings.DROPOUT,
        freeze_encoder=args.freeze_encoder,
    )
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=settings.LEARNING_RATE,
        weight_decay=settings.WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * settings.MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=settings.WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # Checkpoint manager
    checkpoint_dir = settings.MODELS_DIR / "roberta_baseline"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "roberta_baseline")

    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    best_val_f1 = 0.0
    training_history = {
        "train_loss": [],
        "val_metrics": [],
        "test_metrics": None,
    }

    for epoch in range(1, settings.MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{settings.MAX_EPOCHS}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        training_history["train_loss"].append(train_loss)

        # Validate
        val_results = evaluate(model, val_loader, device)
        val_results.print_summary("Validation")
        training_history["val_metrics"].append(val_results.to_dict())

        # Save checkpoint
        is_best = val_results.f1 > best_val_f1
        if is_best:
            best_val_f1 = val_results.f1
            print(f"New best model! F1: {best_val_f1:.4f}")

        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_results.to_dict(),
            config=model.get_config(),
            is_best=is_best,
            checkpoint_name=f"epoch_{epoch}.pt",
        )

    # Final evaluation on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)

    # Load best model
    checkpoint_manager.load(model, checkpoint_name="best.pt", device=device)

    # Evaluate
    test_results = evaluate(model, test_loader, device)
    test_results.print_summary("Test Set")
    training_history["test_metrics"] = test_results.to_dict()

    # Save results
    results_dir = settings.EXPERIMENTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"roberta_baseline_{timestamp}.json"

    experiment_results = {
        "experiment_id": f"roberta_baseline_{timestamp}",
        "model": "roberta_baseline",
        "dataset": settings.PRIMARY_DATASET,
        "config": {
            "seed": settings.RANDOM_SEED,
            "batch_size": settings.BATCH_SIZE,
            "learning_rate": settings.LEARNING_RATE,
            "max_epochs": settings.MAX_EPOCHS,
            "max_seq_length": settings.MAX_SEQ_LENGTH,
        },
        "training_history": training_history,
        "final_test_results": test_results.to_dict(),
        "timestamp": timestamp,
    }

    with open(results_file, "w") as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa Baseline")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze RoBERTa encoder")
    args = parser.parse_args()

    main(args)
