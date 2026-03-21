"""Unit tests for Phase 7 fine-tuning contracts."""

from __future__ import annotations

import numpy as np
import pytest

from pathologic import PathoLogic
from pathologic.models.factory import create_model


def test_mlp_fine_tune_freezes_backbone_layers() -> None:
    x = np.array(
        [
            [0.1, 1.0],
            [0.2, 1.1],
            [0.8, 0.3],
            [0.9, 0.2],
            [0.15, 1.05],
            [0.85, 0.25],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 0, 1], dtype=int)

    model = create_model("mlp", random_state=42)
    model.fit(x, y)
    model.fine_tune(x, y, freeze_layers="backbone_last2", epochs=2)

    assert model.model is not None
    trainable = [param.requires_grad for param in model.model.parameters()]
    assert any(trainable)
    assert not all(trainable)


def test_fine_tune_raises_for_unsupported_freeze_strategy_on_sklearn_model(
    fake_dataset_path: str,
) -> None:
    model = PathoLogic("logreg")
    model.train(fake_dataset_path)

    with pytest.raises(ValueError, match="does not support layer freezing"):
        model.fine_tune(fake_dataset_path)


def test_pathologic_fine_tune_returns_metric_delta(fake_dataset_path: str) -> None:
    model = PathoLogic("mlp")
    model.train(fake_dataset_path)

    report = model.fine_tune(fake_dataset_path)

    assert "before" in report
    assert "after" in report
    assert "metric_delta" in report
    assert "f1" in report["metric_delta"]
    assert "roc_auc" in report["metric_delta"]


def test_pathologic_resolves_finetune_runtime_config() -> None:
    model = PathoLogic("mlp")

    config = model._resolved_finetune_config()

    assert isinstance(config, dict)
    assert config.get("freeze_layers") == "backbone_last2"
    assert "metric_delta" in config
