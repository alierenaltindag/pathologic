"""Model registry for plug-and-play model integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing model capabilities and intent."""

    family: str
    explainability_supported: bool
    supports_predict_proba: bool
    supports_layer_freezing: bool


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry containing model constructor and metadata."""

    name: str
    constructor: Callable[..., Any]
    metadata: ModelMetadata


_MODEL_REGISTRY: dict[str, ModelSpec] = {}


def _ensure_builtin_models_loaded() -> None:
    """Load built-in model zoo registrations lazily."""
    if _MODEL_REGISTRY:
        return
    from pathologic.models import zoo  # noqa: F401


def register(
    name: str,
    *,
    family: str,
    explainability_supported: bool = True,
    supports_predict_proba: bool = True,
    supports_layer_freezing: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register model class/function in global registry."""

    def decorator(constructor: Callable[..., Any]) -> Callable[..., Any]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model alias already registered: {name}")

        _MODEL_REGISTRY[name] = ModelSpec(
            name=name,
            constructor=constructor,
            metadata=ModelMetadata(
                family=family,
                explainability_supported=explainability_supported,
                supports_predict_proba=supports_predict_proba,
                supports_layer_freezing=supports_layer_freezing,
            ),
        )
        return constructor

    return decorator


def build_model(name: str, **kwargs: Any) -> Any:
    """Instantiate model by alias."""
    _ensure_builtin_models_loaded()
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model alias '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name].constructor(**kwargs)


def get_model_metadata(name: str) -> ModelMetadata:
    """Return metadata for registered model."""
    _ensure_builtin_models_loaded()
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model alias '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name].metadata


def list_registered_models() -> list[str]:
    """List all registered aliases sorted for stable output."""
    _ensure_builtin_models_loaded()
    return sorted(_MODEL_REGISTRY)
