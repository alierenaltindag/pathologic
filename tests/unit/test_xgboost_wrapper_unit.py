from __future__ import annotations

import numpy as np
import pandas as pd

from pathologic.models.zoo.xgboost_model import XGBoostWrapper


class _FakeCupyArray:
    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array)


class _FakeCuPyModule:
    @staticmethod
    def asarray(array: object) -> _FakeCupyArray:
        return _FakeCupyArray(np.asarray(array))


class _FakeCuDFFrame:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame


class _FakeCuDFDataFrameFactory:
    @staticmethod
    def from_pandas(frame: pd.DataFrame) -> _FakeCuDFFrame:
        return _FakeCuDFFrame(frame)


class _FakeCuDFModule:
    DataFrame = _FakeCuDFDataFrameFactory


class _CudaEstimator:
    def __init__(self) -> None:
        self.seen_input: object | None = None

    def get_xgb_params(self) -> dict[str, str]:
        return {"device": "cuda"}

    def predict_proba(self, x: object) -> np.ndarray:
        self.seen_input = x
        return np.array([[0.4, 0.6], [0.6, 0.4]], dtype=float)


def _make_wrapper(estimator: object) -> XGBoostWrapper:
    wrapper = XGBoostWrapper.__new__(XGBoostWrapper)
    wrapper.estimator = estimator
    return wrapper


def test_xgboost_wrapper_uses_cupy_for_cuda_numpy_inference(monkeypatch) -> None:
    estimator = _CudaEstimator()
    wrapper = _make_wrapper(estimator)

    def _fake_import_module(name: str):
        if name == "cupy":
            return _FakeCuPyModule
        raise ImportError(name)

    monkeypatch.setattr(
        "pathologic.models.zoo.xgboost_model.importlib.import_module",
        _fake_import_module,
    )

    x = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    _ = wrapper.predict_proba(x)

    assert isinstance(estimator.seen_input, _FakeCupyArray)


def test_xgboost_wrapper_uses_cudf_for_cuda_dataframe_inference(monkeypatch) -> None:
    estimator = _CudaEstimator()
    wrapper = _make_wrapper(estimator)

    def _fake_import_module(name: str):
        if name == "cudf":
            return _FakeCuDFModule
        raise ImportError(name)

    monkeypatch.setattr(
        "pathologic.models.zoo.xgboost_model.importlib.import_module",
        _fake_import_module,
    )

    frame = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    _ = wrapper.predict_proba(frame)

    assert isinstance(estimator.seen_input, _FakeCuDFFrame)
