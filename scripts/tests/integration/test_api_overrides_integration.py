"""Integration tests for kwargs-based API overrides with autocomplete-friendly signatures."""

from __future__ import annotations

import pandas as pd
import pytest

from pathologic import PathoLogic


@pytest.mark.integration
def test_train_and_evaluate_accept_kwargs_overrides(fake_dataset_path: str) -> None:
    model = PathoLogic("logreg")
    model.train(
        fake_dataset_path,
        validation_split=0.5,
        split={"mode": "holdout", "holdout": {"test_size": 0.2, "val_size": 0.2}},
        preprocess={"per_gene": True, "on_missing_gene_column": "disable"},
    )

    report = model.evaluate(
        fake_dataset_path,
        threshold=0.0,
        metrics=["f1"],
        top_k_hotspots=3,
        group_column="gene_id",
    )

    assert set(report["metrics"]) == {"f1"}


@pytest.mark.integration
def test_predict_threshold_override_changes_labeling(fake_dataset_path: str) -> None:
    model = PathoLogic("xgboost")
    model.train(fake_dataset_path)

    threshold_predictions = model.predict(fake_dataset_path, threshold=0.0)

    assert len(threshold_predictions) > 0
    assert all(row["predicted_label"] == "1" for row in threshold_predictions)


@pytest.mark.integration
def test_fine_tune_accepts_kwargs_overrides(fake_dataset_path: str) -> None:
    model = PathoLogic("mlp")
    model.train(fake_dataset_path)

    report = model.fine_tune(
        fake_dataset_path,
        freeze_layers="none",
        learning_rate=0.001,
        epochs=2,
        scheduler={"name": "reduce_on_plateau", "patience": 2},
    )

    assert report["freeze_layers"] == "none"
    assert report["epochs"] == 2


@pytest.mark.integration
def test_unknown_override_key_is_rejected(fake_dataset_path: str) -> None:
    model = PathoLogic("logreg")

    with pytest.raises(ValueError, match=r"Unsupported train\(\) override keys"):
        model.train(fake_dataset_path, unknown_param=1)


@pytest.mark.integration
def test_train_accepts_class_imbalance_and_early_stopping_overrides(
    fake_dataset_path: str,
) -> None:
    model = PathoLogic("logreg")
    model.train(
        fake_dataset_path,
        class_imbalance={"enabled": True, "mode": "balanced"},
        early_stopping={"enabled": True, "patience": 2, "validation_split": 0.3},
    )

    assert model.is_trained is True


@pytest.mark.integration
def test_train_with_drop_rows_missing_policy_reduces_prediction_rows(
    variant_csv_path: str,
) -> None:
    model = PathoLogic("logreg")
    model.train(
        variant_csv_path,
        preprocess={
            "missing_value_policy": "drop_rows",
            "impute_strategy": "none",
            "per_gene": True,
            "on_missing_gene_column": "disable",
        },
    )

    predictions = model.predict(variant_csv_path)
    source_rows = len(pd.read_csv(variant_csv_path))

    assert len(predictions) < source_rows


@pytest.mark.integration
def test_train_accepts_single_model_feature_routing_override(
    variant_csv_path: str,
) -> None:
    model = PathoLogic("logreg")
    model.train(
        variant_csv_path,
        feature_routing={
            "single": {
                "logreg": ["feat_a"],
            }
        },
    )

    predictions = model.predict(variant_csv_path)

    assert model._feature_columns == ["feat_a"]  # noqa: SLF001
    assert len(predictions) > 0


@pytest.mark.integration
def test_train_accepts_hybrid_member_feature_routing_override(
    variant_csv_path: str,
) -> None:
    model = PathoLogic("logreg+xgboost")
    model.train(
        variant_csv_path,
        feature_routing={
            "hybrid": {
                "logreg+xgboost": {
                    "members": {
                        "logreg": ["feat_a"],
                        "xgboost": ["feat_b"],
                    }
                }
            }
        },
    )

    predictions = model.predict(variant_csv_path)

    assert model._feature_columns == ["feat_a", "feat_b"]  # noqa: SLF001
    assert len(predictions) > 0


@pytest.mark.integration
def test_train_filters_out_excluded_columns_override(
    variant_csv_path: str,
) -> None:
    model = PathoLogic("logreg")
    model.train(
        variant_csv_path,
        excluded_columns=["feat_b"],
    )

    predictions = model.predict(variant_csv_path)

    assert model._feature_columns == ["feat_a"]  # noqa: SLF001
    assert len(predictions) > 0
