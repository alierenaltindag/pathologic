from __future__ import annotations

import numpy as np

from pathologic.models.zoo.lightgbm_model import LightGBMWrapper


class _FakeLGBMClassifier:
    init_params: list[dict[str, object]] = []

    def __init__(self, **params):
        self.params = params
        self.fit_calls: list[dict[str, object]] = []
        self._failed_once = False
        _FakeLGBMClassifier.init_params.append(dict(params))

    def fit(self, x, y, **kwargs):
        device = str(self.params.get("device", "")).lower()
        device_type = str(self.params.get("device_type", "")).lower()
        uses_gpu = device in {"gpu", "cuda"} or device_type in {"gpu", "cuda"}
        if uses_gpu and not self._failed_once:
            self._failed_once = True
            if device_type == "cuda":
                raise RuntimeError(
                    "CUDA Tree Learner was not enabled in this build. "
                    "Please recompile with CMake option -DUSE_CUDA=1"
                )
            raise RuntimeError("No OpenCL device found")
        self.fit_calls.append(dict(kwargs))
        return self

    def get_params(self, deep: bool = True):
        return dict(self.params)


class _FakeLightGBMModule:
    LGBMClassifier = _FakeLGBMClassifier


def test_lightgbm_wrapper_does_not_pass_early_stopping_dict_to_constructor(monkeypatch) -> None:
    LightGBMWrapper._cuda_backend_status = "unknown"

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
        device="cpu",
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


def test_lightgbm_wrapper_falls_back_to_cpu_when_gpu_runtime_fails(monkeypatch) -> None:
    LightGBMWrapper._cuda_backend_status = "unknown"
    _FakeLGBMClassifier.init_params = []

    def _fake_import_module(name: str):
        if name == "lightgbm":
            return _FakeLightGBMModule
        raise ImportError(name)

    monkeypatch.setattr(
        "pathologic.models.zoo.lightgbm_model.importlib.import_module",
        _fake_import_module,
    )

    model = LightGBMWrapper(device="cuda", random_state=7)
    x = np.random.RandomState(1).randn(20, 4)
    y = np.array([0] * 10 + [1] * 10)

    model.fit(x, y)

    assert isinstance(model.estimator, _FakeLGBMClassifier)
    assert str(model.estimator.params.get("device", "")).lower() != "gpu"
    assert str(model.estimator.params.get("device_type", "")).lower() != "cuda"
    assert len(_FakeLGBMClassifier.init_params) >= 2


def test_lightgbm_wrapper_caches_cuda_unsupported_state(monkeypatch) -> None:
    LightGBMWrapper._cuda_backend_status = "unknown"
    _FakeLGBMClassifier.init_params = []

    def _fake_import_module(name: str):
        if name == "lightgbm":
            return _FakeLightGBMModule
        raise ImportError(name)

    monkeypatch.setattr(
        "pathologic.models.zoo.lightgbm_model.importlib.import_module",
        _fake_import_module,
    )

    first = LightGBMWrapper(device="cuda", random_state=7)
    x = np.random.RandomState(2).randn(20, 4)
    y = np.array([0] * 10 + [1] * 10)
    first.fit(x, y)

    init_len_before_second = len(_FakeLGBMClassifier.init_params)
    second = LightGBMWrapper(device="cuda", random_state=8)
    new_inits = _FakeLGBMClassifier.init_params[init_len_before_second:]

    assert LightGBMWrapper._cuda_backend_status == "unsupported"
    assert isinstance(second.estimator, _FakeLGBMClassifier)
    assert any(str(params.get("device", "")).lower() == "gpu" for params in new_inits)
    assert all("device_type" not in params for params in new_inits)
