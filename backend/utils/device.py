"""
Device detection and management utilities
"""
import torch
from typing import Literal


def get_device(preferred: Literal["cuda", "cpu", "auto"] = "auto") -> torch.device:
    """
    Get the appropriate device for training/inference

    Args:
        preferred: Preferred device ("cuda", "cpu", "auto")

    Returns:
        torch.device object
    """
    if preferred == "cpu":
        device = torch.device("cpu")
        print("Using device: CPU (forced)")
        return device

    if preferred == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
        return device

    # Auto-detect
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using device: CPU (CUDA not available)")

    return device


def get_device_info() -> dict:
    """
    Get detailed device information

    Returns:
        Dictionary with device info
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": torch.cuda.get_device_capability(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        })

    return info


def empty_cache():
    """Clear CUDA cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_info() -> dict:
    """
    Get GPU memory information

    Returns:
        Dictionary with memory info
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    return {
        "cuda_available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
