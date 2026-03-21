"""Unit tests for fluent hybrid ModelBuilder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathologic import ModelBuilder


def test_builder_requires_at_least_two_members() -> None:
    builder = ModelBuilder().add_model("mlp")
    with pytest.raises(ValueError, match="at least 2"):
        builder.build()


def test_builder_requires_meta_model_for_stacking() -> None:
    builder = (
        ModelBuilder()
        .add_model("mlp")
        .add_model("catboost")
        .strategy("stacking")
    )
    with pytest.raises(ValueError, match="requires a meta model"):
        builder.build()


def test_builder_serializes_hybrid_spec() -> None:
    spec = (
        ModelBuilder()
        .add_model("mlp", max_epochs=10)
        .add_model("catboost", depth=4)
        .add_model("logreg", c=0.5)
        .strategy("stacking", cv=4)
        .meta_model("logreg", c=1.2)
        .tuning_search_space(
            {
                "member__mlp__max_epochs": {"type": "int", "low": 5, "high": 20, "step": 5},
                "meta__c": {"type": "float", "low": 0.5, "high": 2.0},
            }
        )
        .build()
    )

    payload = spec.to_model_config()
    assert spec.alias == "mlp+catboost+logreg"
    assert payload["strategy"] == "stacking"
    assert payload["meta_model"]["alias"] == "logreg"
    assert payload["members"]["mlp"]["max_epochs"] == 10
    assert "member__mlp__max_epochs" in payload["tuning_search_space"]


def test_builder_export_snapshot(tmp_path: Path) -> None:
    spec = (
        ModelBuilder()
        .add_model("mlp")
        .add_model("xgboost")
        .strategy("soft_voting")
        .build()
    )
    output_path = tmp_path / "spec.json"
    spec.export(str(output_path))

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["alias"] == "mlp+xgboost"
    assert loaded["strategy"] == "soft_voting"


def test_builder_serializes_member_weights_and_dynamic_policy() -> None:
    spec = (
        ModelBuilder()
        .add_model("tabnet")
        .add_model("xgboost")
        .strategy("soft_voting")
        .member_weights({"tabnet": 0.25, "xgboost": 0.75}, normalize=True)
        .dynamic_weighting("objective_proportional", objective="roc_auc")
        .build()
    )

    payload = spec.to_model_config()
    strategy_params = payload["strategy_params"]
    assert strategy_params["weights"]["tabnet"] == 0.25
    assert strategy_params["weights"]["xgboost"] == 0.75
    assert strategy_params["normalize_weights"] is True
    assert strategy_params["weighting_policy"] == "objective_proportional"
    assert strategy_params["weighting_objective"] == "roc_auc"
