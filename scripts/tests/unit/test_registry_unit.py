"""Unit tests for model registry behavior."""

from __future__ import annotations

import pytest

from pathologic.models.registry import get_model_metadata, list_registered_models, register


def test_registry_contains_phase3_models() -> None:
    aliases = list_registered_models()
    assert "xgboost" in aliases
    assert "catboost" in aliases
    assert "mlp" in aliases
    assert "tabnet" in aliases
    assert "random_forest" in aliases
    assert "hist_gbdt" in aliases
    assert "logreg" in aliases


def test_registry_metadata_available() -> None:
    metadata = get_model_metadata("xgboost")
    assert metadata.family == "gbdt"
    assert metadata.supports_predict_proba is True


def test_registry_rejects_duplicate_alias() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register(name="mlp", family="neural-network")
        class _DuplicateAlias:
            pass
