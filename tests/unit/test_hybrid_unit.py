"""Unit tests for voting and stacking hybrid strategies."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from pathologic.models.hybrid import StackingEnsembleModel, VotingEnsembleModel, build_default_hybrid


def _sample_binary_data() -> tuple[np.ndarray, np.ndarray]:
    x, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    return x, y


def test_voting_hybrid_strategy_fit_predict() -> None:
    x, y = _sample_binary_data()
    model = VotingEnsembleModel(member_aliases=["mlp", "xgboost"], random_state=42)
    model.fit(x, y)

    preds = model.predict(x)
    probs = model.predict_proba(x)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == x.shape[0]
    assert probs.shape[0] == x.shape[0]


def test_stacking_hybrid_strategy_fit_predict() -> None:
    x, y = _sample_binary_data()
    model = StackingEnsembleModel(member_aliases=["mlp", "xgboost"], random_state=42)
    model.fit(x, y)

    preds = model.predict(x)
    probs = model.predict_proba(x)

    assert isinstance(preds, np.ndarray)
    assert probs.shape[1] >= 2


def test_hybrid_requires_multiple_members() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        VotingEnsembleModel(member_aliases=["mlp"], random_state=42)


def test_voting_hybrid_supports_member_feature_routing() -> None:
    x, y = _sample_binary_data()
    model = VotingEnsembleModel(
        member_aliases=["mlp", "logreg"],
        random_state=42,
        member_feature_indices={
            "mlp": np.array([0]),
            "logreg": np.array([1]),
        },
    )
    model.fit(x, y)

    preds = model.predict(x)
    probs = model.predict_proba(x)

    assert preds.shape[0] == x.shape[0]
    assert probs.shape == (x.shape[0], 2)


def test_voting_hybrid_exposes_normalized_manual_member_weights() -> None:
    x, y = _sample_binary_data()
    model = VotingEnsembleModel(
        member_aliases=["mlp", "logreg"],
        random_state=42,
        weights=[2.0, 1.0],
        weighting_policy="manual",
    )
    model.fit(x, y)

    weights = model.effective_member_weights()
    assert set(weights.keys()) == {"mlp", "logreg"}
    assert abs(weights["mlp"] - (2.0 / 3.0)) < 1e-6
    assert abs(weights["logreg"] - (1.0 / 3.0)) < 1e-6


def test_voting_hybrid_dynamic_weighting_prefers_better_member() -> None:
    class _GoodModel:
        def fit(self, x: np.ndarray, y: np.ndarray) -> _GoodModel:
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return (x[:, 0] > 0).astype(int)

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            pred = self.predict(x)
            prob = np.where(pred == 1, 0.9, 0.1)
            return np.column_stack([1.0 - prob, prob])

    class _BadModel:
        def fit(self, x: np.ndarray, y: np.ndarray) -> _BadModel:
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return (x[:, 0] <= 0).astype(int)

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            pred = self.predict(x)
            prob = np.where(pred == 1, 0.9, 0.1)
            return np.column_stack([1.0 - prob, prob])

    x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)
    model = VotingEnsembleModel(
        member_aliases=["mlp", "logreg"],
        random_state=42,
        weighting_policy="objective_proportional",
        weighting_objective="accuracy",
    )
    model._member_models = [_GoodModel(), _BadModel()]  # noqa: SLF001
    model.fit(x, y)

    weights = model.effective_member_weights()
    scores = model.member_weight_scores()
    assert weights["mlp"] > weights["logreg"]
    assert scores["mlp"] > scores["logreg"]


def test_build_default_hybrid_supports_weight_ratio_for_two_member_voting() -> None:
    model = build_default_hybrid(
        "mlp+logreg",
        model_params={
            "strategy": "soft_voting",
            "strategy_params": {
                "weight_ratio": 0.8,
                "weighting_policy": "manual",
            },
        },
    )

    assert isinstance(model, VotingEnsembleModel)
    assert model.weights is not None
    assert model.weights[0] == pytest.approx(0.8)
    assert model.weights[1] == pytest.approx(0.2)


def test_build_default_hybrid_rejects_invalid_weight_ratio() -> None:
    with pytest.raises(ValueError, match="0 < ratio < 1"):
        build_default_hybrid(
            "mlp+logreg",
            model_params={
                "strategy": "soft_voting",
                "strategy_params": {"weight_ratio": 1.0},
            },
        )
