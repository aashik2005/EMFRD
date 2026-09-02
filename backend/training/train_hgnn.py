"""
Training script for HGNN (Heterogeneous Graph Neural Network)

NOTE: This requires a dataset with graph structure (user_id, product_id).
Use FraudAmazon dataset for graph experiments.

Usage:
    python -m backend.training.train_hgnn --dataset fraud_amazon
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from tqdm import tqdm
import json
from datetime import datetime
import argparse

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    print("ERROR: DGL is required for HGNN training.")
    print("Install with: pip install dgl")
    sys.exit(1)

from backend.config import settings
from backend.data import get_dataset
from backend.models import HGNNWithFeatures
from backend.evaluation import MetricsCalculator
from backend.utils import get_device, set_seed, CheckpointManager


def train_epoch(model, graph, train_mask, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    graph = graph.to(device)

    # Get labels
    labels = graph.nodes['review'].data['label'][train_mask] if 'review' in graph.ntypes else None
    if labels is None:
        # Try to find labeled node type
        for ntype in graph.ntypes:
            if 'label' in graph.nodes[ntype].data:
                labels = graph.nodes[ntype].data['label'][train_mask]
                break

    # Forward pass
    outputs = model(graph, labels=labels)
    loss = outputs['loss']

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.MAX_GRAD_NORM)
    optimizer.step()

    return loss.item()


def evaluate(model, graph, mask, device):
    """Evaluate model"""
    model.eval()
    graph = graph.to(device)

    with torch.no_grad():
        outputs = model(graph)
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=-1)[mask]

        # Get labels
        labels = graph.nodes['review'].data['label'][mask] if 'review' in graph.ntypes else None
        if labels is None:
            for ntype in graph.ntypes:
                if 'label' in graph.nodes[ntype].data:
                    labels = graph.nodes[ntype].data['label'][mask]
                    break

        # Calculate metrics
        import numpy as np
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Get probabilities
        probas = torch.softmax(logits, dim=-1)[mask].cpu().numpy()

        results = MetricsCalculator.calculate(labels_np, preds_np, probas)

    return results


def main(args):
    """Main training function"""
    print("="*80)
    print("EMFRD - HGNN Training")
    print("="*80)

    if not DGL_AVAILABLE:
        print("ERROR: DGL is required. Install with: pip install dgl")
        return

    # Set seed
    set_seed(settings.RANDOM_SEED)

    # Get device
    device = get_device(settings.DEVICE)

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    dataset = get_dataset(
        args.dataset,
        data_dir=settings.DATA_DIR / "raw" / args.dataset,
        cache_dir=settings.CACHE_DIR,
    )

    graph, dataset_info = dataset.prepare()

    print(f"\nGraph info:")
    print(f"  Node types: {graph.ntypes}")
    print(f"  Edge types: {graph.etypes}")
    print(f"  Total nodes: {graph.num_nodes()}")
    print(f"  Total edges: {graph.num_edges()}")

    # Create train/val/test masks
    num_nodes = graph.num_nodes('review') if 'review' in graph.ntypes else graph.num_nodes()
    indices = torch.randperm(num_nodes)

    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True

    print(f"\nSplit sizes:")
    print(f"  Train: {train_mask.sum()}")
    print(f"  Val: {val_mask.sum()}")
    print(f"  Test: {test_mask.sum()}")

    # Initialize model
    print("\nInitializing HGNN...")
    model = HGNNWithFeatures(
        num_users=graph.num_nodes('user') if 'user' in graph.ntypes else graph.num_nodes('U'),
        num_products=graph.num_nodes('product') if 'product' in graph.ntypes else graph.num_nodes('P'),
        num_reviews=num_nodes,
        hidden_dim=settings.HGNN_HIDDEN_DIM,
        num_layers=settings.HGNN_LAYERS,
        dropout=settings.HGNN_DROPOUT,
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.LEARNING_RATE,
        weight_decay=settings.WEIGHT_DECAY,
    )

    # Checkpoint manager
    checkpoint_dir = settings.MODELS_DIR / "hgnn"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "hgnn")

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

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-"*80)

        # Train
        train_loss = train_epoch(model, graph, train_mask, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        training_history["train_loss"].append(train_loss)

        # Validate
        val_results = evaluate(model, graph, val_mask, device)
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

    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)

    checkpoint_manager.load(model, checkpoint_name="best.pt", device=device)
    test_results = evaluate(model, graph, test_mask, device)
    test_results.print_summary("Test Set")
    training_history["test_metrics"] = test_results.to_dict()

    # Compare with paper
    paper_reference = {
        "accuracy": 0.952,
        "precision": 0.949,
        "recall": 0.953,
        "f1": 0.951,
    }
    MetricsCalculator.compare_with_paper(test_results, paper_reference, "HGNN")

    # Save results
    results_dir = settings.EXPERIMENTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"hgnn_{timestamp}.json"

    experiment_results = {
        "experiment_id": f"hgnn_{timestamp}",
        "model": "hgnn",
        "dataset": args.dataset,
        "config": model.get_config(),
        "training_history": training_history,
        "final_test_results": test_results.to_dict(),
        "paper_reference": paper_reference,
        "timestamp": timestamp,
    }

    with open(results_file, "w") as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HGNN")
    parser.add_argument("--dataset", default="fraud_amazon", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    main(args)
