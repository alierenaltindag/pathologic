"""Unit tests for the Phase 1 API skeleton."""

from __future__ import annotations

import pytest

from pathologic import PathoLogic


def test_import_and_public_export() -> None:
    model = PathoLogic("xgboost")
    assert model.model_name == "xgboost"


def test_unsupported_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        PathoLogic("nonexistent-model")


def test_train_marks_model_as_trained(fake_dataset_path: str) -> None:
    model = PathoLogic("tabnet+xgb")
    trained = model.train(fake_dataset_path)

    assert trained is model
    assert model.is_trained is True
    assert model.last_train_source == fake_dataset_path


def test_predict_requires_training(fake_dataset_path: str) -> None:
    model = PathoLogic("tabnet")
    with pytest.raises(RuntimeError, match="Call train"):
        model.predict(fake_dataset_path)
