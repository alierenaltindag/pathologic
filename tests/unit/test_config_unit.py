"""Unit tests for default configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from pathologic import PathoLogic


def test_defaults_yaml_exists() -> None:
    defaults = Path("pathologic/configs/defaults.yaml")
    assert defaults.exists()


def test_defaults_yaml_has_required_keys() -> None:
    defaults = Path("pathologic/configs/defaults.yaml")
    parsed = yaml.safe_load(defaults.read_text(encoding="utf-8"))

    assert parsed["model"] == "xgboost"
    assert parsed["seed"] == 42
    assert parsed["device"] == "auto"


def test_defaults_yaml_uses_modular_train_test_tune_paths() -> None:
    defaults = Path("pathologic/configs/defaults.yaml")
    parsed = yaml.safe_load(defaults.read_text(encoding="utf-8"))

    assert parsed["train"] == "runtime/train.yaml"
    assert parsed["test"] == "runtime/test.yaml"
    assert parsed["tune"] == "runtime/tune.yaml"


def test_model_loads_defaults_on_init() -> None:
    model = PathoLogic("mlp")
    assert model.defaults["seed"] == 42
    assert "train" in model.defaults
    assert "test" in model.defaults
    assert "tune" in model.defaults
    assert isinstance(model.defaults["train"], dict)
    assert isinstance(model.defaults["test"], dict)
    assert isinstance(model.defaults["tune"], dict)
    assert "epochs" in model.defaults["train"]
    assert "preprocess" in model.defaults["train"]
    preprocess = model.defaults["train"]["preprocess"]
    assert preprocess["missing_value_policy"] == "drop_rows"
    assert preprocess["add_missing_indicators"] is False
    assert "threshold" in model.defaults["test"]
    assert "n_trials" in model.defaults["tune"]
    assert "logging" in model.defaults
    assert "hardware" in model.defaults
    assert isinstance(model.defaults["models"]["mlp"], str)

    model_params = model._model_params_from_defaults()
    assert model_params["architecture_path"] == "mlp_architecture.yaml"
    assert model_params["optimizer"]["name"] == "adamw"


def test_error_analysis_columns_propagate_to_explain_group_columns() -> None:
    model = PathoLogic("logreg")

    explain_config = model._resolved_explain_config()  # noqa: SLF001

    assert "Veri_Kaynagi_Paneli" in explain_config["group_columns"]
    assert "Veri_Kaynagi_Paneli" in explain_config["false_positive"]["group_columns"]
