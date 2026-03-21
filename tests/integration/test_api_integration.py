"""Integration tests for Phase 1 end-to-end API skeleton behavior."""

from __future__ import annotations

import pytest

from pathologic import PathoLogic


@pytest.mark.integration
def test_train_predict_smoke_flow(fake_dataset_path: str) -> None:
    model = PathoLogic("tabnet+xgb")
    model.train(fake_dataset_path)

    predictions = model.predict(fake_dataset_path)

    assert len(predictions) >= 1
    row = predictions[0]
    assert row["source"] == fake_dataset_path
    assert row["model_name"] == "tabnet+xgb"
    assert row["device"] in {"cuda", "mps", "cpu"}
    assert isinstance(row["row_index"], int)
    assert row["predicted_label"] in {"0", "1"}
    assert isinstance(row["score"], float)


@pytest.mark.integration
def test_blank_data_path_is_rejected() -> None:
    model = PathoLogic("xgboost")
    with pytest.raises(ValueError, match="non-empty"):
        model.train("   ")
