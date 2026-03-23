"""Hardware selection helpers for PathoLogic.

Priority order is CUDA -> MPS -> CPU.
"""

from __future__ import annotations


def detect_preferred_device() -> str:
    """Select a preferred compute backend with graceful fallback."""
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        # MAC_OPTIMIZATION: Explicitly detect and return 'mps' to enable
        # Apple Silicon GPU acceleration via Metal (MPS) in PyTorch.
        return "mps"

    return "cpu"
