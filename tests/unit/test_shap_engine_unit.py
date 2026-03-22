"""Unit tests for explainability attribution backend selection and determinism."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pathologic.explain.shap_engine import ShapAttributionEngine


class _DummyProbModel:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x.sum(axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


class _TreeWrapper:
    def __init__(self) -> None:
        self.estimator = RandomForestClassifier(n_estimators=10, random_state=42)


class _LinearWrapper:
    def __init__(self) -> None:
        self.estimator = LogisticRegression(random_state=42)


class _LightGBMEstimatorLike:
    __module__ = "lightgbm.sklearn"


class _LightGBMWrapper:
    def __init__(self) -> None:
        self.estimator = _LightGBMEstimatorLike()


class _HybridLikeWrapper:
    def __init__(self) -> None:
        self.model = self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x.sum(axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


class _HybridEnsembleWrapper:
    def __init__(self) -> None:
        self.member_aliases = ["tabnet", "xgboost"]
        self._member_models = [object(), object()]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x.sum(axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


class _TabNetEstimatorLike:
    __module__ = "pytorch_tabnet.tab_model"

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x.sum(axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


class _TabNetWrapperLike:
    def __init__(self, *, native: bool = True) -> None:
        self._is_native_tabnet = native
        self.estimator = _TabNetEstimatorLike()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(x)


class _TorchLeaf:
    def parameters(self):
        return iter([1.0])

    def eval(self) -> None:
        return None


class _NestedTorchWrapper:
    def __init__(self) -> None:
        self.network = _TorchLeaf()


class _TupleParamTorchLeaf:
    def parameters(self):
        class _Param:
            device = "cpu"

        return iter([("w", _Param())])

    def eval(self) -> None:
        return None


def test_shap_engine_auto_backend_selection_by_model_type() -> None:
    engine = ShapAttributionEngine(backend="auto", random_state=42)

    assert engine._select_backend(_TreeWrapper()) == "tree"
    assert engine._select_backend(_LightGBMWrapper()) == "tree"
    assert engine._select_backend(_LinearWrapper()) == "linear"
    assert engine._select_backend(_DummyProbModel()) == "shap"
    assert engine._select_backend(_HybridLikeWrapper()) == "shap"
    assert engine._select_backend(_HybridEnsembleWrapper()) == "proxy"
    assert engine._select_backend(_TabNetWrapperLike()) == "deep"


def test_shap_engine_proxy_is_deterministic_for_same_seed() -> None:
    engine = ShapAttributionEngine(backend="proxy", random_state=123)
    model = _DummyProbModel()

    x_background = np.array(
        [[0.1, 0.2], [0.2, 0.1], [0.4, 0.3], [0.3, 0.5]],
        dtype=float,
    )
    x_target = np.array(
        [[0.15, 0.25], [0.35, 0.45]],
        dtype=float,
    )

    result_a = engine.compute(model=model, x_background=x_background, x_target=x_target)
    result_b = engine.compute(model=model, x_background=x_background, x_target=x_target)

    assert result_a.backend == "proxy"
    assert np.allclose(result_a.contributions, result_b.contributions)
    assert np.allclose(result_a.global_importance, result_b.global_importance)


def test_shap_engine_auto_fallback_records_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = ShapAttributionEngine(backend="auto", random_state=42)
    model = _DummyProbModel()

    monkeypatch.setattr(engine, "_select_backend", lambda _: "tree")
    monkeypatch.setattr(
        engine,
        "_try_shap",
        lambda **_: (None, "shap_runtime_error: synthetic_failure"),
    )

    x_background = np.array([[0.1, 0.2], [0.2, 0.1]], dtype=float)
    x_target = np.array([[0.15, 0.25]], dtype=float)

    result = engine.compute(model=model, x_background=x_background, x_target=x_target)

    assert result.backend == "proxy"
    assert result.diagnostics.get("fallback_from") == "tree"
    assert "synthetic_failure" in str(result.diagnostics.get("fallback_reason", ""))


def test_shap_engine_explicit_backend_raises_with_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ShapAttributionEngine(backend="tree", random_state=42)
    model = _DummyProbModel()

    monkeypatch.setattr(
        engine,
        "_try_shap",
        lambda **_: (None, "import_error: shap unavailable"),
    )

    x_background = np.array([[0.1, 0.2], [0.2, 0.1]], dtype=float)
    x_target = np.array([[0.15, 0.25]], dtype=float)

    with pytest.raises(RuntimeError, match="shap unavailable"):
        engine.compute(model=model, x_background=x_background, x_target=x_target)


def test_shap_engine_resolve_torch_model_from_nested_attributes() -> None:
    engine = ShapAttributionEngine(backend="auto", random_state=42)
    model = _NestedTorchWrapper()

    torch_model = engine._resolve_torch_model(model)

    assert torch_model is not None
    assert hasattr(torch_model, "parameters")


def test_shap_engine_resolve_first_parameter_tensor_supports_tuple_style() -> None:
    engine = ShapAttributionEngine(backend="auto", random_state=42)

    first_param = engine._resolve_first_parameter_tensor(_TupleParamTorchLeaf())

    assert first_param is not None
    assert hasattr(first_param, "device")


def test_tabnet_deep_shap_failure_raises_in_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ShapAttributionEngine(backend="auto", random_state=42)
    model = _TabNetWrapperLike(native=True)

    monkeypatch.setattr(engine, "_try_shap", lambda **_: (None, "torch_model_unavailable"))

    x_background = np.array([[0.1, 0.2], [0.2, 0.1]], dtype=float)
    x_target = np.array([[0.15, 0.25]], dtype=float)

    with pytest.raises(RuntimeError, match="DeepSHAP is required for TabNet"):
        engine.compute(model=model, x_background=x_background, x_target=x_target)
