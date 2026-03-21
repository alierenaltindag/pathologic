"""Unit tests for model zoo fit/predict/predict_proba contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

from pathologic.models.factory import create_model
from pathologic.models.zoo.mlp import extract_mlp_preprocess_hints


def _sample_binary_data() -> tuple[np.ndarray, np.ndarray]:
    x, y = make_classification(
        n_samples=80,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    return x, y


def test_wrapper_contracts_for_single_models() -> None:
    x, y = _sample_binary_data()
    for alias in [
        "xgboost",
        "catboost",
        "mlp",
        "tabnet",
        "random_forest",
        "hist_gbdt",
        "logreg",
    ]:
        model = create_model(alias, random_state=42)
        model.fit(x, y)

        preds = model.predict(x)
        probs = model.predict_proba(x)

        assert preds.shape[0] == x.shape[0]
        assert probs.shape[0] == x.shape[0]
        assert probs.shape[1] >= 2


def test_mlp_wrapper_accepts_configurable_params() -> None:
    model = create_model(
        "mlp",
        random_state=42,
        model_params={
            "hidden_layer_sizes": [8, 4],
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.005,
            "learning_rate_init": 0.002,
            "max_iter": 123,
        },
    )

    assert model.layer_specs[0]["units"] == 8
    assert model.layer_specs[1]["units"] == 4
    assert model._default_activation == "tanh"
    assert model._learning_rate == 0.002
    assert model._max_epochs == 123


def test_mlp_wrapper_reads_dedicated_architecture_yaml(tmp_path: Path) -> None:
    arch_path = tmp_path / "mlp_custom.yaml"
    arch_path.write_text(
        (
            "version: 1\n"
            "model:\n"
            "  architecture:\n"
            "    layers:\n"
            "      - type: dense\n"
            "        units: 48\n"
            "      - type: batch_norm\n"
            "      - type: dense\n"
            "        units: 24\n"
            "      - type: dropout\n"
            "        p: 0.25\n"
            "  activation: tanh\n"
            "  max_epochs: 12\n"
        ),
        encoding="utf-8",
    )

    model = create_model(
        "mlp",
        random_state=42,
        model_params={"architecture_path": str(arch_path)},
    )

    assert model.layer_specs[0]["type"] == "dense"
    assert model.layer_specs[0]["units"] == 48
    assert model.layer_specs[1]["type"] == "batch_norm"
    assert model._default_activation == "tanh"
    assert model._max_epochs == 12


def test_mlp_architecture_extracts_gene_batch_norm_hints(tmp_path: Path) -> None:
    arch_path = tmp_path / "mlp_hints.yaml"
    arch_path.write_text(
        (
            "version: 1\n"
            "model:\n"
            "  architecture:\n"
            "    layers:\n"
            "      - type: dense\n"
            "        units: 32\n"
            "      - type: batch_norm\n"
            "        features: [cadd_phred]\n"
            "      - type: gene_batch_norm\n"
            "        features: [revel_score]\n"
        ),
        encoding="utf-8",
    )

    hints = extract_mlp_preprocess_hints(str(arch_path))

    assert hints["scaler"] == "standard"
    assert hints["per_gene"] is True
    assert hints["per_gene_features"] == ["revel_score"]
    assert hints["scaler_features"] == ["cadd_phred"]


def test_mlp_wrapper_supports_optional_early_stopping() -> None:
    x, y = _sample_binary_data()
    model = create_model(
        "mlp",
        random_state=42,
        model_params={
            "hidden_layer_sizes": [16, 8],
            "max_epochs": 20,
            "batch_size": 16,
            "early_stopping": {
                "enabled": True,
                "patience": 1,
                "min_delta": 1000.0,
                "validation_split": 0.3,
                "restore_best_weights": True,
            },
            "scheduler": {"name": "none"},
        },
    )

    model.fit(x, y)

    assert model._trained_epochs < model._max_epochs


def test_sklearn_wrappers_forward_class_weight() -> None:
    rf_model = create_model(
        "random_forest",
        random_state=42,
        model_params={"class_weight": "balanced"},
    )
    lr_model = create_model(
        "logreg",
        random_state=42,
        model_params={"class_weight": "balanced"},
    )

    assert rf_model.estimator.get_params()["class_weight"] == "balanced"
    assert lr_model.estimator.get_params()["class_weight"] == "balanced"


def test_xgboost_wrapper_accepts_early_stopping_and_scale_pos_weight() -> None:
    x, y = _sample_binary_data()
    model = create_model(
        "xgboost",
        random_state=42,
        model_params={
            "scale_pos_weight": 2.0,
            "early_stopping": {
                "enabled": True,
                "patience": 1,
                "validation_split": 0.25,
            },
        },
    )
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape[0] == x.shape[0]


def test_catboost_wrapper_accepts_early_stopping_and_class_weight() -> None:
    x, y = _sample_binary_data()
    model = create_model(
        "catboost",
        random_state=42,
        model_params={
            "class_weight": "balanced",
            "early_stopping": {
                "enabled": True,
                "patience": 1,
                "validation_split": 0.25,
            },
        },
    )
    model.fit(x, y)
    probs = model.predict_proba(x)
    assert probs.shape[0] == x.shape[0]


def test_xgboost_wrapper_accepts_extended_regularization_params() -> None:
    x, y = _sample_binary_data()
    model = create_model(
        "xgboost",
        random_state=42,
        model_params={
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_weight": 3.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.5,
            "reg_alpha": 0.001,
            "reg_lambda": 1.2,
        },
    )
    model.fit(x, y)
    probs = model.predict_proba(x)
    assert probs.shape[0] == x.shape[0]


def test_catboost_wrapper_accepts_extended_regularization_params() -> None:
    x, y = _sample_binary_data()
    model = create_model(
        "catboost",
        random_state=42,
        model_params={
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "subsample": 0.8,
        },
    )
    model.fit(x, y)
    probs = model.predict_proba(x)
    assert probs.shape[0] == x.shape[0]


def test_sklearn_wrappers_accept_extended_hyperparameters() -> None:
    x, y = _sample_binary_data()

    rf_model = create_model(
        "random_forest",
        random_state=42,
        model_params={
            "n_estimators": 400,
            "max_depth": 12,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    )
    rf_model.fit(x, y)
    rf_probs = rf_model.predict_proba(x)
    assert rf_probs.shape[0] == x.shape[0]

    hgbdt_model = create_model(
        "hist_gbdt",
        random_state=42,
        model_params={
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "max_depth": 6,
            "min_samples_leaf": 10,
            "l2_regularization": 0.01,
            "early_stopping": True,
            "validation_fraction": 0.2,
            "n_iter_no_change": 10,
            "tol": 1e-4,
            "class_weight": "balanced",
        },
    )
    hgbdt_model.fit(x, y)
    hgbdt_probs = hgbdt_model.predict_proba(x)
    assert hgbdt_probs.shape[0] == x.shape[0]

    logreg_model = create_model(
        "logreg",
        random_state=42,
        model_params={
            "c": 0.5,
            "max_iter": 1200,
            "solver": "lbfgs",
            "tol": 1e-4,
            "fit_intercept": True,
            "class_weight": "balanced",
        },
    )
    logreg_model.fit(x, y)
    logreg_probs = logreg_model.predict_proba(x)
    assert logreg_probs.shape[0] == x.shape[0]


def test_tabnet_wrapper_accepts_optimizer_scheduler_and_lr_params() -> None:
    x, y = _sample_binary_data()
    model = create_model(
        "tabnet",
        random_state=42,
        model_params={
            "max_epochs": 15,
            "patience": 3,
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "optimizer_name": "adamw",
            "scheduler_name": "step",
            "scheduler_step_size": 5,
            "scheduler_gamma": 0.8,
            "n_d": 16,
            "n_a": 16,
            "n_steps": 4,
            "gamma": 1.4,
            "lambda_sparse": 0.001,
            "batch_size": 256,
            "virtual_batch_size": 64,
        },
    )

    model.fit(x, y)
    preds = model.predict(x)
    probs = model.predict_proba(x)

    assert preds.shape[0] == x.shape[0]
    assert probs.shape[0] == x.shape[0]
    assert model._learning_rate == 0.01
    assert model._weight_decay == 0.001


def test_tabnet_wrapper_handles_reduce_on_plateau_scheduler_request() -> None:
    x, y = _sample_binary_data()
    model = create_model(
        "tabnet",
        random_state=42,
        model_params={
            "max_epochs": 10,
            "scheduler_name": "reduce_on_plateau",
            "scheduler_patience": 2,
            "scheduler_factor": 0.5,
        },
    )

    model.fit(x, y)
    probs = model.predict_proba(x)
    assert probs.shape[0] == x.shape[0]

