"""Factory for creating single or hybrid models from aliases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pathologic.models.hybrid import build_default_hybrid
from pathologic.models.registry import build_model


def _sanitize_model_params(model_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize model params and prevent random_state collisions."""
    if model_params is None:
        return {}
    params = dict(model_params)
    params.pop("random_state", None)
    return params


def create_model(
    alias: str,
    *,
    random_state: int,
    model_params: Mapping[str, Any] | None = None,
) -> Any:
    """Create model object from alias.

    Hybrid aliases use dedicated builders; single aliases are loaded from registry.
    """
    params = _sanitize_model_params(model_params)
    if "+" in alias:
        return build_default_hybrid(alias, random_state=random_state, model_params=params)
    return build_model(alias, random_state=random_state, **params)
