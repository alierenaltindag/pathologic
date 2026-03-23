"""Unit tests for trainer device fallback and validation metrics."""

from __future__ import annotations

import os

import numpy as np

from pathologic.engine import Trainer, TrainerConfig
from pathologic.models import create_model


def test_trainer_falls_back_when_requested_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("pathologic.engine.trainer.detect_preferred_device", lambda: "cpu")

    trainer = Trainer(TrainerConfig(device="cuda"))

    assert trainer.device == "cpu"


def test_trainer_fit_returns_validation_metrics() -> None:
    x_train = np.array([[0.1, 1.0], [0.2, 1.1], [0.8, 0.3], [0.9, 0.2]])
    y_train = np.array([0, 0, 1, 1])
    x_val = np.array([[0.15, 1.05], [0.85, 0.25]])
    y_val = np.array([0, 1])

    trainer = Trainer(TrainerConfig(device="cpu"))
    model = create_model("logreg", random_state=42)
    result = trainer.fit(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )

    assert result.device == "cpu"
    assert "f1" in result.metrics
    assert "precision" in result.metrics


def test_trainer_sets_cuda_visible_devices_from_gpu_ids(monkeypatch) -> None:
    monkeypatch.setattr("pathologic.engine.trainer.detect_preferred_device", lambda: "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    trainer = Trainer(TrainerConfig(device="cuda", gpu_ids=[2, 3]))

    assert trainer.device == "cuda"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_trainer_forwards_explicit_validation_to_model_fit() -> None:
    class _ValAwareModel:
        def __init__(self) -> None:
            self.seen_x_val: np.ndarray | None = None
            self.seen_y_val: np.ndarray | None = None

        def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            *,
            x_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
        ) -> None:
            self.seen_x_val = x_val
            self.seen_y_val = y_val

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.array([0, 1], dtype=int)

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)

    x_train = np.array([[0.1, 1.0], [0.9, 0.2]])
    y_train = np.array([0, 1])
    x_val = np.array([[0.2, 1.1], [0.8, 0.3]])
    y_val = np.array([0, 1])

    model = _ValAwareModel()
    trainer = Trainer(TrainerConfig(device="cpu"))
    result = trainer.fit(model=model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    assert result.metrics
    assert model.seen_x_val is x_val
    assert model.seen_y_val is y_val


def test_trainer_fallbacks_when_model_fit_does_not_accept_validation_kwargs() -> None:
    class _NoValKwargModel:
        def __init__(self) -> None:
            self.fit_calls = 0

        def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
            self.fit_calls += 1

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.array([0, 1], dtype=int)

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)

    x_train = np.array([[0.1, 1.0], [0.9, 0.2]])
    y_train = np.array([0, 1])
    x_val = np.array([[0.2, 1.1], [0.8, 0.3]])
    y_val = np.array([0, 1])

    model = _NoValKwargModel()
    trainer = Trainer(TrainerConfig(device="cpu"))
    result = trainer.fit(model=model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    assert model.fit_calls == 1
    assert result.metrics
