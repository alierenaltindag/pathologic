"""Integration test to ensure gene leakage is prevented across folds."""

from __future__ import annotations

import pytest

from pathologic.data.loader import build_folds, load_dataset, validate_schema


@pytest.mark.integration
def test_gene_variants_never_cross_train_validation(variant_csv_path: str) -> None:
    df = load_dataset(variant_csv_path)
    validate_schema(df, required_feature_columns=["revel_score", "cadd_phred"])

    folds = build_folds(
        df,
        label_column="label",
        gene_column="gene_id",
        n_splits=3,
        stratified=True,
        random_state=42,
    )

    for train_idx, val_idx in folds:
        train_genes = set(df.iloc[train_idx]["gene_id"])
        val_genes = set(df.iloc[val_idx]["gene_id"])
        assert train_genes.isdisjoint(val_genes)


@pytest.mark.integration
def test_pathologic_train_generates_split_summary(variant_csv_path: str) -> None:
    from pathologic import PathoLogic

    model = PathoLogic("tabnet+xgb")
    model.train(variant_csv_path)

    assert len(model.last_split_summary) == 3
    assert all(item["shared_genes"] == 0 for item in model.last_split_summary)


@pytest.mark.integration
def test_pathologic_train_supports_holdout_split_mode(
    monkeypatch: pytest.MonkeyPatch,
    variant_csv_path: str,
) -> None:
    from pathologic import PathoLogic

    custom_defaults = {
        "seed": 42,
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["revel_score", "cadd_phred"],
        },
        "split": {
            "mode": "holdout",
            "holdout": {
                "test_size": 0.2,
                "val_size": 0.2,
                "stratified": True,
            },
        },
        "preprocess": {
            "impute_strategy": "median",
            "scaler": "standard",
            "per_gene": True,
        },
        "models": {
            "tabnet+xgb": {
                "members": {
                    "tabnet": {},
                    "xgboost": {"n_estimators": 10, "max_depth": 2},
                }
            }
        },
    }

    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model = PathoLogic("tabnet+xgb")
    model.train(variant_csv_path)

    assert len(model.last_split_summary) == 1
    summary = model.last_split_summary[0]
    assert summary["split_mode"] == "holdout"
    assert summary["train_val_shared_genes"] == 0
    assert summary["train_test_shared_genes"] == 0
    assert summary["val_test_shared_genes"] == 0

