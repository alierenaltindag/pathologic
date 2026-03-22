"""Unit tests for the Phase 1 API skeleton."""

from __future__ import annotations

from pathlib import Path

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


def test_tabnet_auto_missingness_adds_indicator_feature(tmp_path: Path) -> None:
    data_path = tmp_path / "tabnet_missing.csv"
    data_path.write_text(
        (
            "gene_id,label,feature__GERP_Score,feature__REVEL_Score\n"
            "G1,1,,0.1\n"
            "G1,0,2.4,0.2\n"
            "G2,1,,0.3\n"
            "G2,0,1.8,0.4\n"
            "G3,1,2.1,0.5\n"
            "G3,0,,0.6\n"
        ),
        encoding="utf-8",
    )

    model = PathoLogic("tabnet")
    model.train(
        str(data_path),
        label_column="label",
        gene_column="gene_id",
        required_features=["feature__GERP_Score", "feature__REVEL_Score"],
    )

    assert "feature__GERP_Score__is_missing" in model._feature_columns  # noqa: SLF001
