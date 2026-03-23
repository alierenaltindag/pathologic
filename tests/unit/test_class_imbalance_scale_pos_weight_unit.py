from __future__ import annotations

import numpy as np

from pathologic import PathoLogic


def test_scale_pos_weight_mode_sets_xgboost_and_lightgbm_params() -> None:
    y = np.asarray([0, 0, 0, 1], dtype=int)
    cfg = {
        "enabled": True,
        "mode": "scale_pos_weight",
        "positive_class_weight": 2.08,
    }

    xgb_model = PathoLogic("xgboost")
    xgb_params = xgb_model._with_runtime_model_params(  # noqa: SLF001
        model_params={},
        y=y,
        early_stopping_config=None,
        class_imbalance_config=cfg,
    )

    lgb_model = PathoLogic("lightgbm")
    lgb_params = lgb_model._with_runtime_model_params(  # noqa: SLF001
        model_params={},
        y=y,
        early_stopping_config=None,
        class_imbalance_config=cfg,
    )

    assert xgb_params["scale_pos_weight"] == 2.08
    assert lgb_params["scale_pos_weight"] == 2.08


def test_scale_pos_weight_mode_does_not_force_balanced_class_weight() -> None:
    y = np.asarray([0, 0, 0, 1], dtype=int)
    cfg = {
        "enabled": True,
        "mode": "scale_pos_weight",
        "positive_class_weight": 2.08,
    }

    cat_model = PathoLogic("catboost")
    cat_params = cat_model._with_runtime_model_params(  # noqa: SLF001
        model_params={},
        y=y,
        early_stopping_config=None,
        class_imbalance_config=cfg,
    )

    assert "class_weight" not in cat_params


def test_runtime_early_stopping_false_overrides_model_default_single() -> None:
    y = np.asarray([0, 1, 0, 1], dtype=int)
    model = PathoLogic("xgboost")

    params = model._with_runtime_model_params(  # noqa: SLF001
        model_params={
            "early_stopping": {
                "enabled": True,
                "patience": 99,
                "validation_split": 0.4,
            }
        },
        y=y,
        early_stopping_config={
            "enabled": False,
            "patience": 3,
            "validation_split": 0.2,
        },
        class_imbalance_config=None,
    )

    assert params["early_stopping"]["enabled"] is False
    assert params["early_stopping"]["patience"] == 3
    assert params["early_stopping"]["validation_split"] == 0.2


def test_runtime_early_stopping_false_overrides_model_default_hybrid_members() -> None:
    y = np.asarray([0, 1, 0, 1], dtype=int)
    model = PathoLogic("xgboost+lightgbm")

    params = model._with_runtime_model_params(  # noqa: SLF001
        model_params={
            "member__xgboost__early_stopping": {
                "enabled": True,
                "patience": 77,
            },
            "member__lightgbm__early_stopping": {
                "enabled": True,
                "patience": 88,
            },
        },
        y=y,
        early_stopping_config={
            "enabled": False,
            "patience": 4,
            "validation_split": 0.2,
        },
        class_imbalance_config=None,
    )

    assert params["member__xgboost__early_stopping"]["enabled"] is False
    assert params["member__xgboost__early_stopping"]["patience"] == 4
    assert params["member__lightgbm__early_stopping"]["enabled"] is False
    assert params["member__lightgbm__early_stopping"]["patience"] == 4
