"""Unit and integration tests for LightGBM model integration."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from pathologic.models.factory import create_model
from pathologic.models.registry import list_registered_models
from pathologic.models.zoo.lightgbm_model import LightGBMWrapper
from pathologic.explain.shap_engine import ShapAttributionEngine


def _sample_binary_data() -> tuple[np.ndarray, np.ndarray]:
    x, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=42,
    )
    return x.astype(np.float32), y


def test_lightgbm_registration() -> None:
    """Verify that lightgbm is registered in the model factory."""
    registered_names = list_registered_models()
    assert "lightgbm" in registered_names


def test_lightgbm_instantiation() -> None:
    """Verify that lightgbm can be instantiated via the factory."""
    model = create_model("lightgbm", random_state=42)
    assert isinstance(model, LightGBMWrapper)
    assert model._random_state == 42


def test_lightgbm_fit_predict_contract() -> None:
    """Verify fit, predict, and predict_proba contracts."""
    x, y = _sample_binary_data()
    model = create_model("lightgbm", random_state=42)

    # Test fit
    model.fit(x, y)

    # Test predict
    preds = model.predict(x)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Test predict_proba
    probs = model.predict_proba(x)
    assert probs.shape == (100, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_lightgbm_feature_importances() -> None:
    """Verify feature_importances_ property."""
    x, y = _sample_binary_data()
    model = create_model("lightgbm", random_state=42)
    model.fit(x, y)

    importances = model.feature_importances_
    assert importances.shape == (10,)
    assert np.all(importances >= 0)


def test_lightgbm_explainability_compatibility() -> None:
    """Verify that ShapAttributionEngine can handle LightGBMWrapper."""
    x, y = _sample_binary_data()
    model = create_model("lightgbm", random_state=42)
    model.fit(x, y)

    engine = ShapAttributionEngine(background_size=10, random_state=42)

    # Use a small subset for explanation
    x_background = x[:10]
    x_target = x[10:15]

    result = engine.compute(
        model=model,
        x_background=x_background,
        x_target=x_target
    )

    assert result.contributions.shape == (5, 10)
    assert result.global_importance.shape == (10,)
    # If lightgbm is installed, it should ideally use tree_shap or similar
    # If not, it falls back to proxy_contributions
    assert result.backend in ["tree_shap", "proxy_contributions"]


def test_lightgbm_params_override() -> None:
    """Verify that custom parameters are passed to the underlying estimator."""
    model = create_model(
        "lightgbm",
        random_state=42,
        model_params={
            "n_estimators": 50,
            "num_leaves": 20,
            "learning_rate": 0.01
        }
    )
    # Check if params reach the estimator
    if not model._using_fallback:
        assert model.estimator.n_estimators == 50
        assert model.estimator.num_leaves == 20
        assert model.estimator.learning_rate == 0.01
    else:
        # Check mapping for HistGradientBoostingClassifier fallback
        assert model.estimator.max_iter == 50
        assert model.estimator.max_leaf_nodes == 20
        assert model.estimator.learning_rate == 0.01
