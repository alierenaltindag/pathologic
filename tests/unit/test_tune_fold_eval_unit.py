from __future__ import annotations

import numpy as np
import pytest

from pathologic import PathoLogic


class _FoldEvalCaptureModel:
    def __init__(self) -> None:
        self.fit_calls: list[dict[str, int]] = []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "_FoldEvalCaptureModel":
        self.fit_calls.append(
            {
                "train_rows": int(len(x)),
                "val_rows": int(len(x_val) if x_val is not None else 0),
                "train_labels": int(len(y)),
                "val_labels": int(len(y_val) if y_val is not None else 0),
            }
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Return deterministic binary predictions.
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Provide stable probabilities for evaluator.
        p1 = np.full(len(x), 0.25, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def test_tune_passes_fold_validation_set_to_model_fit(
    monkeypatch: pytest.MonkeyPatch,
    variant_csv_path: str,
) -> None:
    defaults = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["revel_score", "cadd_phred"],
            "excluded_columns": [],
        },
        "split": {
            "mode": "cross_validation",
            "cross_validation": {"n_splits": 3, "stratified": True},
        },
        "preprocess": {
            "missing_value_policy": "impute",
            "impute_strategy": "median",
            "scaler": "standard",
            "per_gene": True,
            "add_missing_indicators": False,
        },
        "train": {
            "split": {
                "mode": "cross_validation",
                "cross_validation": {"n_splits": 3, "stratified": True},
            },
            "preprocess": {
                "missing_value_policy": "impute",
                "impute_strategy": "median",
                "scaler": "standard",
                "per_gene": True,
                "add_missing_indicators": False,
            },
            "early_stopping": {"enabled": True, "validation_split": 0.2, "patience": 2},
            "class_imbalance": {"enabled": False},
        },
        "tune": {
            "engine": "random",
            "n_trials": 1,
            "max_trials": 1,
            "objective": "f1",
            "timeout_minutes": 1,
            "early_stopping": {"enabled": False, "patience": 2},
            "split": {
                "mode": "cross_validation",
                "cross_validation": {"n_splits": 3, "stratified": True},
            },
            "preprocess": {
                "missing_value_policy": "impute",
                "impute_strategy": "median",
                "scaler": "standard",
                "per_gene": True,
                "add_missing_indicators": False,
            },
            "class_imbalance": {"enabled": False},
        },
        "models": {
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "tuning_search_space": {
                    "max_depth": {"type": "int", "low": 3, "high": 3},
                },
            }
        },
    }

    capture_model = _FoldEvalCaptureModel()

    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: defaults))

    import pathologic.core as core_module

    monkeypatch.setattr(
        core_module,
        "create_model",
        lambda *args, **kwargs: capture_model,
    )

    model = PathoLogic("xgboost")
    result = model.tune(variant_csv_path)

    assert isinstance(result, dict)
    assert capture_model.fit_calls, "Expected tune() to call model.fit at least once."
    assert all(call["val_rows"] > 0 for call in capture_model.fit_calls)
    assert all(call["val_labels"] == call["val_rows"] for call in capture_model.fit_calls)
