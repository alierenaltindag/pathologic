"""Integration tests for model registry + API training with configured aliases."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathologic import PathoLogic


@pytest.mark.integration
def test_config_model_alias_train_predict(variant_csv_path: str) -> None:
    model = PathoLogic("mlp")
    model.train(variant_csv_path)
    predictions = model.predict(variant_csv_path)

    assert len(predictions) > 0
    assert predictions[0]["model_name"] == "mlp"
    assert "score" in predictions[0]


@pytest.mark.integration
def test_hybrid_alias_train_predict_schema(variant_csv_path: str) -> None:
    model = PathoLogic("tabnet+xgb")
    model.train(variant_csv_path)
    predictions = model.predict(variant_csv_path)

    assert len(predictions) > 0
    row = predictions[0]
    assert row["model_name"] == "tabnet+xgb"
    assert row["predicted_label"] in {"0", "1"}
    assert 0.0 <= float(row["score"]) <= 1.0


@pytest.mark.integration
def test_builder_hybrid_train_predict_without_config_file(variant_csv_path: str) -> None:
    builder = (
        PathoLogic.builder()
        .add_model("mlp", max_epochs=8)
        .add_model("catboost", depth=4)
        .add_model("logreg", c=1.0)
        .strategy("stacking", cv=3)
        .meta_model("logreg", c=0.8)
    )
    model = PathoLogic.from_builder(builder)

    model.train(variant_csv_path)
    predictions = model.predict(variant_csv_path)

    assert model.model_name == "mlp+catboost+logreg"
    assert model.is_trained is True
    assert len(predictions) > 0
    assert predictions[0]["model_name"] == "mlp+catboost+logreg"


@pytest.mark.integration
def test_builder_blending_strategy_train_predict(variant_csv_path: str) -> None:
    builder = (
        PathoLogic.builder()
        .add_model("mlp", max_epochs=6)
        .add_model("xgboost", n_estimators=50)
        .add_model("logreg", c=1.0)
        .strategy("blending", blend_size=0.2)
        .meta_model("logreg", c=1.2)
    )
    model = PathoLogic.from_builder(builder)

    model.train(variant_csv_path)
    report = model.evaluate(variant_csv_path)

    assert "metrics" in report
    assert "f1" in report["metrics"]


@pytest.mark.integration
def test_sklearn_aliases_train_predict(variant_csv_path: str) -> None:
    for alias in ["random_forest", "hist_gbdt", "logreg"]:
        model = PathoLogic(alias)
        model.train(variant_csv_path)
        predictions = model.predict(variant_csv_path)

        assert len(predictions) > 0
        assert predictions[0]["model_name"] == alias


@pytest.mark.integration
def test_mlp_architecture_file_from_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "variants.csv"
    data_path.write_text(
        (
            "variant_id,gene_id,label,revel_score,cadd_phred\n"
            "1,G1,1,0.1,1.0\n"
            "2,G1,0,0.2,1.1\n"
            "3,G2,1,0.3,0.9\n"
            "4,G2,0,0.4,1.2\n"
            "5,G3,1,0.5,1.3\n"
            "6,G3,0,0.6,0.8\n"
        ),
        encoding="utf-8",
    )

    architecture_path = tmp_path / "mlp_arch.yaml"
    architecture_path.write_text(
        (
            "version: 1\n"
            "model:\n"
            "  architecture:\n"
            "    layers:\n"
            "      - type: dense\n"
            "        units: 40\n"
            "      - type: gene_batch_norm\n"
            "        features: [revel_score]\n"
            "      - type: dense\n"
            "        units: 20\n"
            "  activation: tanh\n"
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
        "models": {
            "mlp": {
                "architecture_path": str(architecture_path),
            }
        },
    }

    monkeypatch.setattr(PathoLogic, "_load_defaults", staticmethod(lambda: custom_defaults))

    model = PathoLogic("mlp")
    model.train(str(data_path))

    assert model._trained_model is not None
    assert model._preprocessor is not None
    assert model._preprocessor.per_gene is True
    assert model._preprocessor.per_gene_features == ["revel_score"]
    assert model._trained_model.layer_specs[0]["units"] == 40
    assert model._trained_model.layer_specs[2]["units"] == 20

