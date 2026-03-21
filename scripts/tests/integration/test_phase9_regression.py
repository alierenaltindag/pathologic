"""Phase 9 regression suite covering critical workflows across phases."""

from __future__ import annotations

import pytest

from pathologic import PathoLogic
from pathologic.data.loader import build_folds, load_dataset


@pytest.mark.integration
@pytest.mark.regression
def test_phase9_regression_train_predict_evaluate_logreg(variant_csv_path: str) -> None:
    model = PathoLogic("logreg")
    model.train(variant_csv_path)

    predictions = model.predict(variant_csv_path)
    report = model.evaluate(variant_csv_path)

    assert len(predictions) > 0
    assert predictions[0]["model_name"] == "logreg"
    assert predictions[0]["device"] in {"cuda", "mps", "cpu"}
    assert "metrics" in report
    assert "f1" in report["metrics"]


@pytest.mark.integration
@pytest.mark.regression
def test_phase9_regression_hybrid_predict_and_explain(variant_csv_path: str) -> None:
    model = PathoLogic("tabnet+xgb")
    model.train(variant_csv_path)

    predictions = model.predict(variant_csv_path)
    explain_report = model.explain(variant_csv_path)

    assert len(predictions) > 0
    assert predictions[0]["model_name"] == "tabnet+xgb"
    assert "backend" in explain_report
    assert "global_feature_importance" in explain_report


@pytest.mark.integration
@pytest.mark.regression
def test_phase9_regression_gene_split_remains_leakage_safe(variant_csv_path: str) -> None:
    dataset = load_dataset(variant_csv_path)
    folds = build_folds(
        dataset,
        label_column="label",
        gene_column="gene_id",
        n_splits=3,
        stratified=True,
        random_state=42,
    )

    for train_idx, val_idx in folds:
        train_genes = set(dataset.iloc[train_idx]["gene_id"].astype(str))
        val_genes = set(dataset.iloc[val_idx]["gene_id"].astype(str))
        assert train_genes.isdisjoint(val_genes)


@pytest.mark.integration
@pytest.mark.regression
def test_phase9_regression_finetune_metric_delta_available(variant_csv_path: str) -> None:
    model = PathoLogic("mlp")
    model.train(variant_csv_path)

    report = model.fine_tune(variant_csv_path, freeze_layers="backbone_last2", epochs=3)

    assert report["model_name"] == "mlp"
    assert "metric_delta" in report
    assert isinstance(report["metric_delta"], dict)


@pytest.mark.integration
@pytest.mark.regression
def test_phase9_regression_quickstart_examples_still_run(variant_csv_path: str) -> None:
    from tests.integration.test_phase8_quickstart_integration import _run_example

    payload = _run_example("example_01_basic_workflow.py", variant_csv_path)

    assert payload["model_name"] == "logreg"
    assert int(payload["prediction_count"]) > 0
