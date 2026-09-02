"""
Checkpoint management utilities
"""
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints with best model tracking

    Prevents accidental overwriting and tracks training metadata
    """

    def __init__(self, checkpoint_dir: Path, model_name: str):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory for checkpoints
            model_name: Model identifier
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        checkpoint_name: str = "last.pt",
    ) -> Path:
        """
        Save model checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Training metrics
            config: Model configuration
            is_best: Whether this is the best model so far
            checkpoint_name: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = {
            "model_name": self.model_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "config": config or {},
            "timestamp": datetime.now().isoformat(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save metadata as JSON for easy inspection
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump({
                "model_name": self.model_name,
                "epoch": epoch,
                "metrics": metrics,
                "config": config,
                "timestamp": checkpoint["timestamp"],
            }, f, indent=2)

        # Save as best if specified
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")

            # Update best metadata
            best_metadata_path = best_path.with_suffix(".json")
            with open(best_metadata_path, "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "epoch": epoch,
                    "metrics": metrics,
                    "config": config,
                    "timestamp": checkpoint["timestamp"],
                }, f, indent=2)

        return checkpoint_path

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_name: str = "best.pt",
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            checkpoint_name: Checkpoint filename
            device: Device to load to

        Returns:
            Checkpoint metadata dictionary
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        if device is None:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded {checkpoint['model_name']} from epoch {checkpoint['epoch']}")
        if "metrics" in checkpoint and checkpoint["metrics"]:
            print(f"  Metrics: {checkpoint['metrics']}")

        return checkpoint

    def exists(self, checkpoint_name: str = "best.pt") -> bool:
        """Check if checkpoint exists"""
        return (self.checkpoint_dir / checkpoint_name).exists()

    def get_best_metric(self, metric_name: str) -> Optional[float]:
        """Get best metric value from saved checkpoint"""
        metadata_path = self.checkpoint_dir / "best.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return metadata.get("metrics", {}).get(metric_name)

    def list_checkpoints(self) -> list:
        """List all checkpoint files"""
        return sorted([
            p.name for p in self.checkpoint_dir.glob("*.pt")
        ])
