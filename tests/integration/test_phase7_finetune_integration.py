"""Integration tests for Phase 7 fine-tune workflow."""

from __future__ import annotations

import pytest

from pathologic import PathoLogic


@pytest.mark.integration
def test_phase7_mlp_train_then_finetune_reports_metric_delta(variant_csv_path: str) -> None:
    model = PathoLogic("mlp")
    model.train(variant_csv_path)

    report = model.fine_tune(variant_csv_path)

    assert report["model_name"] == "mlp"
    assert "before" in report
    assert "after" in report
    assert "metric_delta" in report
    assert "f1" in report["metric_delta"]
    assert "roc_auc" in report["metric_delta"]


@pytest.mark.integration
def test_phase7_logreg_finetune_without_freezing_works(variant_csv_path: str) -> None:
    model = PathoLogic("logreg")
    model.train(variant_csv_path)

    report = model.fine_tune(variant_csv_path, freeze_layers="none")

    assert report["model_name"] == "logreg"
    assert isinstance(report["metric_delta"], dict)


@pytest.mark.integration
def test_phase7_hybrid_finetune_reports_hybrid_policy(variant_csv_path: str) -> None:
    model = PathoLogic("tabnet+xgb")
    model.train(variant_csv_path)

    report = model.fine_tune(variant_csv_path, freeze_layers="none")

    assert report["model_name"] == "tabnet+xgb"
    assert report["hybrid_policy"] == "all_members_common"
