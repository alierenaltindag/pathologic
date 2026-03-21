from __future__ import annotations

import numpy as np

from pathologic.models.zoo.lightgbm_model import LightGBMWrapper


class _FakeLGBMClassifier:
    def __init__(self, **params):
        self.params = params
        self.fit_calls: list[dict[str, object]] = []

    def fit(self, x, y, **kwargs):
        self.fit_calls.append(dict(kwargs))
        return self


class _FakeLightGBMModule:
    LGBMClassifier = _FakeLGBMClassifier


def test_lightgbm_wrapper_does_not_pass_early_stopping_dict_to_constructor(monkeypatch) -> None:
    def _fake_import_module(name: str):
        if name == "lightgbm":
            return _FakeLightGBMModule
        raise ImportError(name)

    monkeypatch.setattr(
        "pathologic.models.zoo.lightgbm_model.importlib.import_module",
        _fake_import_module,
    )

    model = LightGBMWrapper(
        n_estimators=20,
        random_state=7,
        early_stopping={"enabled": True, "validation_split": 0.25, "patience": 6},
    )

    assert isinstance(model.estimator, _FakeLGBMClassifier)
    assert "early_stopping" not in model.estimator.params

    x = np.random.RandomState(0).randn(40, 6)
    y = np.array([0] * 20 + [1] * 20)
    model.fit(x, y)

    assert model.estimator.fit_calls
    fit_kwargs = model.estimator.fit_calls[-1]
    assert "eval_set" in fit_kwargs
    assert fit_kwargs.get("early_stopping_rounds") == 6
