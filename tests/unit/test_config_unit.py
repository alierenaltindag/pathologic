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
    assert preprocess["missing_value_policy"] == "none"
    assert preprocess["add_missing_indicators"] is False
    assert preprocess["tabnet_missingness_mode"] == "auto"
    assert preprocess["tabnet_impute_strategy"] == "median"
    assert "feature__GERP_Score" in preprocess["tabnet_missing_indicator_features"]
    assert "threshold" in model.defaults["test"]
    assert "n_trials" in model.defaults["tune"]
    assert "logging" in model.defaults
    assert "hardware" in model.defaults
    assert isinstance(model.defaults["models"]["mlp"], str)

    model_params = model._model_params_from_defaults()
    architecture_path = str(model_params["architecture_path"])
    assert architecture_path.endswith(".yaml")
    assert model_params["optimizer"]["name"] == "adamw"


def test_error_analysis_columns_propagate_to_explain_group_columns() -> None:
    model = PathoLogic("logreg")

    explain_config = model._resolved_explain_config()  # noqa: SLF001

    assert "Veri_Kaynagi_Paneli" in explain_config["group_columns"]
    assert "Veri_Kaynagi_Paneli" in explain_config["false_positive"]["group_columns"]


def test_defaults_data_schema_matches_engineered_feature_contract() -> None:
    model = PathoLogic("logreg")
    data_config = model.defaults["data"]

    required_features = list(data_config["required_features"])
    excluded_columns = list(data_config["excluded_columns"])
    error_columns = list(data_config["error_analysis_columns"])

    assert "gnomAD_log" in required_features
    assert "gnomAD_is_zero" in required_features
    assert "cpg_flag" in required_features
    assert "proline_intro" in required_features
    assert "cysteine_intro" in required_features
    assert "proline_remove" in required_features
    assert "gnomAD_AF" not in required_features

    assert "Protein change" in excluded_columns
    assert "VariationID" in excluded_columns
    assert "AA_Position" in excluded_columns

    assert "VariationID" in error_columns
    assert "Gene(s)" in error_columns
    assert "Protein change" in error_columns
    assert "Veri_Kaynagi_Paneli" in error_columns


def test_runtime_defaults_enable_early_stopping_for_train_and_tune() -> None:
    model = PathoLogic("xgboost")

    assert model.defaults["train"]["early_stopping"]["enabled"] is True
    assert model.defaults["tune"]["early_stopping"]["enabled"] is True


def test_model_configs_start_with_mild_regularization_defaults() -> None:
    xgb = PathoLogic("xgboost")._model_params_from_defaults()  # noqa: SLF001
    lgb = PathoLogic("lightgbm")._model_params_from_defaults()  # noqa: SLF001
    cat = PathoLogic("catboost")._model_params_from_defaults()  # noqa: SLF001
    tabnet = PathoLogic("tabnet")._model_params_from_defaults()  # noqa: SLF001
    mlp = PathoLogic("mlp")._model_params_from_defaults()  # noqa: SLF001
    logreg = PathoLogic("logreg")._model_params_from_defaults()  # noqa: SLF001
    rf = PathoLogic("random_forest")._model_params_from_defaults()  # noqa: SLF001
    hgb = PathoLogic("hist_gbdt")._model_params_from_defaults()  # noqa: SLF001

    assert float(xgb["reg_alpha"]) > 0.0
    assert float(xgb["reg_lambda"]) > 0.0

    assert float(lgb["reg_alpha"]) > 0.0
    assert float(lgb["reg_lambda"]) > 0.0

    assert float(cat["l2_leaf_reg"]) > 0.0

    assert float(tabnet["weight_decay"]) > 0.0
    assert float(tabnet["lambda_sparse"]) > 0.0

    assert float(mlp["alpha"]) > 0.0
    optimizer_cfg = mlp.get("optimizer", {})
    assert isinstance(optimizer_cfg, dict)
    assert float(optimizer_cfg.get("weight_decay", 0.0)) > 0.0

    assert float(logreg["c"]) <= 1.0

    assert int(rf["min_samples_leaf"]) >= 2
    assert int(rf["min_samples_split"]) >= 4

    assert float(hgb["l2_regularization"]) > 0.0
