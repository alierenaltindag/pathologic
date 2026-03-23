"""Integration tests for config-driven model parameter flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathologic import PathoLogic


@pytest.mark.integration
def test_runtime_sections_are_resolved_from_modular_defaults() -> None:
    model = PathoLogic("logreg")

    assert isinstance(model.defaults.get("train"), dict)
    assert isinstance(model.defaults.get("test"), dict)
    assert isinstance(model.defaults.get("tune"), dict)
    assert "epochs" in model.defaults["train"]
    assert "threshold" in model.defaults["test"]
    assert "n_trials" in model.defaults["tune"]


@pytest.mark.integration
def test_mlp_params_are_loaded_from_defaults(
    monkeypatch: pytest.MonkeyPatch,
    variant_csv_path: str,
) -> None:
    custom_defaults = {
        "seed": 42,
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["revel_score", "cadd_phred"],
        },
        "split": {"n_splits": 3, "stratified": True},
        "preprocess": {
            "impute_strategy": "median",
            "scaler": "standard",
            "per_gene": True,
        },
        "models": {
            "mlp": {
                "hidden_layer_sizes": [12, 6],
                "activation": "tanh",
                "max_iter": 111,
                "learning_rate_init": 0.007,
            }
        },
    }

    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model = PathoLogic("mlp")
    model.train(variant_csv_path)

    assert model._trained_model is not None
    assert model._trained_model.layer_specs[0]["units"] == 12
    assert model._trained_model.layer_specs[1]["units"] == 6
    assert model._trained_model._default_activation == "tanh"
    assert model._trained_model._max_epochs == 111
    assert model._trained_model._learning_rate == 0.007


@pytest.mark.integration
def test_mlp_max_epochs_falls_back_to_train_epochs(
    monkeypatch: pytest.MonkeyPatch,
    variant_csv_path: str,
) -> None:
    custom_defaults = {
        "seed": 42,
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["revel_score", "cadd_phred"],
        },
        "split": {"n_splits": 3, "stratified": True},
        "preprocess": {
            "impute_strategy": "median",
            "scaler": "standard",
            "per_gene": True,
        },
        "train": {
            "epochs": 17,
            "batch_size": 16,
            "optimizer": {"name": "adam", "lr": 0.003, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "early_stopping": {"enabled": False, "patience": 4},
        },
        "models": {
            "mlp": {
                "hidden_layer_sizes": [12, 6],
                "activation": "relu",
            }
        },
    }

    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model = PathoLogic("mlp")
    model.train(variant_csv_path)

    assert model._trained_model is not None
    assert model._trained_model._max_epochs == 17


@pytest.mark.integration
def test_mlp_architecture_max_epochs_overrides_model_and_train(
    monkeypatch: pytest.MonkeyPatch,
    variant_csv_path: str,
    tmp_path: Path,
) -> None:
    architecture_path = tmp_path / "mlp_arch_priority.yaml"
    architecture_path.write_text(
        (
            "version: 1\n"
            "model:\n"
            "  architecture:\n"
            "    layers:\n"
            "      - type: dense\n"
            "        units: 16\n"
            "  max_epochs: 3\n"
        ),
        encoding="utf-8",
    )

    custom_defaults = {
        "seed": 42,
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["revel_score", "cadd_phred"],
        },
        "split": {"n_splits": 3, "stratified": True},
        "preprocess": {
            "impute_strategy": "median",
            "scaler": "standard",
            "per_gene": True,
        },
        "train": {
            "epochs": 99,
            "batch_size": 16,
            "optimizer": {"name": "adam", "lr": 0.003, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "early_stopping": {"enabled": False, "patience": 4},
        },
        "models": {
            "mlp": {
                "architecture_path": str(architecture_path),
                "max_epochs": 21,
            }
        },
    }

    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model = PathoLogic("mlp")
    model.train(variant_csv_path)

    assert model._trained_model is not None
    assert model._trained_model._max_epochs == 3


@pytest.mark.integration
def test_train_runtime_early_stopping_false_overrides_model_config(
    monkeypatch: pytest.MonkeyPatch,
    variant_csv_path: str,
) -> None:
    custom_defaults = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "label_column": "label",
            "gene_column": "gene_id",
            "required_features": ["revel_score", "cadd_phred"],
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
        },
        "train": {
            "epochs": 5,
            "batch_size": 16,
            "validation_split": 0.2,
            "mixed_precision": False,
            "split": {
                "mode": "cross_validation",
                "cross_validation": {"n_splits": 3, "stratified": True},
            },
            "preprocess": {
                "missing_value_policy": "impute",
                "impute_strategy": "median",
                "scaler": "standard",
                "per_gene": True,
            },
            "early_stopping": {
                "enabled": False,
                "patience": 2,
                "validation_split": 0.2,
            },
            "class_imbalance": {"enabled": False},
        },
        "models": {
            "xgboost": {
                "n_estimators": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
                "early_stopping": {
                    "enabled": True,
                    "patience": 25,
                    "validation_split": 0.4,
                },
            }
        },
    }

    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model = PathoLogic("xgboost")
    model.train(variant_csv_path)

    assert model._trained_model is not None
    assert model._trained_model._early_stopping_cfg["enabled"] is False
    assert model._trained_model._early_stopping_cfg["patience"] == 2

