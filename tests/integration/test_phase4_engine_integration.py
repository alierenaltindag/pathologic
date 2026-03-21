"""Integration tests for Phase 4 engine-backed core APIs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pathologic import PathoLogic


@pytest.mark.integration
def test_core_evaluate_returns_metrics(variant_csv_path: str) -> None:
    model = PathoLogic("logreg")
    model.train(variant_csv_path)

    report = model.evaluate(variant_csv_path)

    assert "metrics" in report
    assert "f1" in report["metrics"]
    assert "grouped_metrics" in report


@pytest.mark.integration
def test_core_tune_runs_random_engine(
    monkeypatch: pytest.MonkeyPatch,
    variant_csv_path: str,
) -> None:
    custom_defaults = {
        "seed": 42,
        "device": "auto",
        "mixed_precision": False,
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["feat_a", "feat_b"],
        },
        "split": {
            "mode": "cross_validation",
            "stratified": True,
            "cross_validation": {"n_splits": 3, "stratified": True},
        },
        "preprocess": {
            "impute_strategy": "median",
            "scaler": "standard",
            "per_gene": True,
        },
        "train": {
            "epochs": 5,
            "batch_size": 16,
            "validation_split": 0.2,
            "early_stopping": {"enabled": False, "patience": 3},
            "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "ddp": {"enabled": False, "backend": "nccl", "rank": 0, "world_size": 1},
        },
        "test": {
            "threshold": 0.5,
            "metrics": ["roc_auc", "auprc", "f1", "mcc", "precision", "recall"],
            "top_k_hotspots": 10,
        },
        "tune": {
            "engine": "random",
            "n_trials": 4,
            "objective": "roc_auc",
            "timeout_minutes": 1,
        },
        "models": {
            "logreg": {
                "c": 1.0,
                "max_iter": 300,
                "tuning_search_space": {
                    "c": {"type": "float", "low": 0.2, "high": 2.5},
                    "max_iter": {"type": "int", "low": 200, "high": 400, "step": 200},
                },
            }
        },
    }

    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model = PathoLogic("logreg")
    result = model.tune(variant_csv_path)

    assert result["engine"] == "random"
    assert "best_params" in result
    assert len(result["trials"]) > 0


@pytest.mark.integration
def test_core_tune_is_reproducible_with_same_seed(
    monkeypatch: pytest.MonkeyPatch,
    variant_csv_path: str,
) -> None:
    custom_defaults = {
        "seed": 123,
        "device": "auto",
        "mixed_precision": False,
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["feat_a", "feat_b"],
        },
        "split": {
            "mode": "cross_validation",
            "stratified": True,
            "cross_validation": {"n_splits": 3, "stratified": True},
        },
        "preprocess": {
            "impute_strategy": "median",
            "scaler": "standard",
            "per_gene": True,
        },
        "train": {
            "epochs": 5,
            "batch_size": 16,
            "validation_split": 0.2,
            "early_stopping": {"enabled": False, "patience": 3},
            "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "gpu_ids": [],
            "ddp": {
                "enabled": False,
                "backend": "nccl",
                "rank": 0,
                "world_size": 1,
                "gpu_ids": [],
            },
        },
        "test": {
            "threshold": 0.5,
            "metrics": ["roc_auc", "auprc", "f1", "mcc", "precision", "recall"],
            "top_k_hotspots": 10,
        },
        "tune": {
            "engine": "random",
            "n_trials": 6,
            "max_trials": 6,
            "objective": "roc_auc",
            "timeout_minutes": 1,
            "early_stopping": {
                "enabled": False,
                "patience": 3,
                "min_improvement": 0.0,
            },
        },
        "models": {
            "logreg": {
                "c": 1.0,
                "max_iter": 300,
                "tuning_search_space": {
                    "c": {"type": "float", "low": 0.2, "high": 2.5},
                    "max_iter": {"type": "int", "low": 200, "high": 400, "step": 200},
                },
            }
        },
    }
    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model_a = PathoLogic("logreg")
    model_b = PathoLogic("logreg")

    result_a = model_a.tune(variant_csv_path)
    result_b = model_b.tune(variant_csv_path)

    assert result_a["best_params"] == result_b["best_params"]
    assert result_a["best_score"] == result_b["best_score"]


@pytest.mark.integration
def test_core_train_works_with_cuda_request_and_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
    variant_csv_path: str,
) -> None:
    monkeypatch.setattr("pathologic.core.detect_preferred_device", lambda: "cpu")
    monkeypatch.setattr("pathologic.engine.trainer.detect_preferred_device", lambda: "cpu")

    custom_defaults = {
        "seed": 42,
        "device": "cuda",
        "mixed_precision": False,
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["feat_a", "feat_b"],
        },
        "split": {
            "mode": "cross_validation",
            "stratified": True,
            "cross_validation": {"n_splits": 3, "stratified": True},
        },
        "preprocess": {
            "impute_strategy": "median",
            "scaler": "standard",
            "per_gene": True,
        },
        "train": {
            "epochs": 5,
            "batch_size": 16,
            "validation_split": 0.2,
            "early_stopping": {"enabled": False, "patience": 3},
            "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "gpu_ids": [],
            "ddp": {
                "enabled": False,
                "backend": "nccl",
                "rank": 0,
                "world_size": 1,
                "gpu_ids": [],
            },
        },
        "test": {
            "threshold": 0.5,
            "metrics": ["roc_auc", "auprc", "f1", "mcc", "precision", "recall"],
            "top_k_hotspots": 10,
        },
        "tune": {
            "engine": "random",
            "n_trials": 4,
            "max_trials": 4,
            "objective": "roc_auc",
            "timeout_minutes": 1,
            "early_stopping": {
                "enabled": False,
                "patience": 3,
                "min_improvement": 0.0,
            },
        },
        "models": {"logreg": {"c": 1.0, "max_iter": 300}},
    }
    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model = PathoLogic("logreg")
    model.train(variant_csv_path)
    predictions = model.predict(variant_csv_path)

    assert model.last_train_metrics
    assert len(predictions) > 0


@pytest.mark.integration
def test_core_train_falls_back_when_per_gene_enabled_but_gene_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "variants_no_gene.csv"
    pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3", "v4", "v5", "v6"],
            "label": [1, 0, 1, 0, 1, 0],
            "feat_a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feat_b": [1.0, 1.1, 0.9, 1.2, 1.3, 0.8],
        }
    ).to_csv(csv_path, index=False)

    custom_defaults = {
        "seed": 42,
        "device": "auto",
        "mixed_precision": False,
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["feat_a", "feat_b"],
        },
        "train": {
            "epochs": 5,
            "batch_size": 16,
            "validation_split": 0.2,
            "split": {
                "mode": "cross_validation",
                "stratified": True,
                "cross_validation": {"n_splits": 3, "stratified": True},
            },
            "preprocess": {
                "impute_strategy": "none",
                "scaler": "standard",
                "per_gene": True,
                "on_missing_gene_column": "disable",
            },
            "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "ddp": {"enabled": False, "backend": "nccl", "rank": 0, "world_size": 1},
        },
        "test": {
            "threshold": 0.5,
            "metrics": ["roc_auc", "auprc", "f1", "mcc", "precision", "recall"],
            "top_k_hotspots": 10,
        },
        "tune": {
            "engine": "random",
            "n_trials": 3,
            "max_trials": 3,
            "objective": "roc_auc",
            "timeout_minutes": 1,
            "early_stopping": {
                "enabled": False,
                "patience": 2,
                "min_improvement": 0.0,
            },
        },
        "models": {"logreg": {"c": 1.0, "max_iter": 200}},
    }
    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model = PathoLogic("logreg")
    model.train(str(csv_path))

    assert model.is_trained is True
    assert model.last_train_metrics
