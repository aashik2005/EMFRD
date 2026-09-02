"""
Reproducibility utilities for EMFRD
Ensures consistent results across runs
"""
import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    print(f"Setting random seed: {seed}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Additional PyTorch settings for reproducibility
    # Note: These may reduce performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_deterministic(enabled: bool = True):
    """
    Enable/disable deterministic mode in PyTorch

    Args:
        enabled: Whether to enable deterministic mode

    Note:
        Deterministic mode may reduce performance but ensures reproducibility
    """
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("Deterministic mode enabled (may reduce performance)")
    else:
        torch.use_deterministic_algorithms(False)
        print("Deterministic mode disabled")


def get_env_info() -> dict:
    """
    Get environment information for reproducibility

    Returns:
        Dictionary with environment info
    """
    return {
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
