"""Model registry and factory exports."""

from pathologic.models.builder import EnsembleSpec, ModelBuilder
from pathologic.models.factory import create_model
from pathologic.models.registry import get_model_metadata, list_registered_models, register

__all__ = [
    "create_model",
    "get_model_metadata",
    "list_registered_models",
    "register",
    "ModelBuilder",
    "EnsembleSpec",
]
