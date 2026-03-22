"""Unit tests for phase 6 explainability helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathologic.explain.biological_mapper import BiologicalMapper
from pathologic.explain.false_positive_analyzer import FalsePositiveAnalyzer
from pathologic.explain.schemas import ExplainabilityReport, FeatureAttribution, SampleExplanation
from pathologic.explain.service import ExplainabilityService
from pathologic.explain.visualizer import ExplainabilityVisualizer


class _DummyMemberModel:
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = float(offset)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x.sum(axis=1) + self._offset
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


class _DummyHybridModel:
    def __init__(self) -> None:
        self.member_aliases = ["tabnet", "xgboost"]
        self._member_models = [_DummyMemberModel(0.1), _DummyMemberModel(-0.1)]
        self._member_feature_indices = {
            "tabnet": np.array([0, 1], dtype=int),
            "xgboost": np.array([2, 3], dtype=int),
        }

    def _member_input(self, alias: str, x: np.ndarray) -> np.ndarray:
        indices = self._member_feature_indices[alias]
        return x[:, indices]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = []
        for alias, model in zip(self.member_aliases, self._member_models, strict=True):
            proba = model.predict_proba(self._member_input(alias, x))[:, -1]
            probs.append(proba)
        stacked = np.vstack(probs)
        positive = np.mean(stacked, axis=0)
        return np.column_stack([1.0 - positive, positive])

    def effective_member_weights(self) -> dict[str, float]:
        return {"tabnet": 0.6, "xgboost": 0.4}

    def member_weight_scores(self) -> dict[str, float]:
        return {"tabnet": 0.82, "xgboost": 0.71}


def test_biological_mapper_generates_labels_and_narratives() -> None:
    mapper = BiologicalMapper(mapping={"revel_score": "Domain conservation score"})

    assert mapper.label_for("revel_score") == "Domain conservation score"
    assert mapper.label_for("domain_length") == "Protein domain feature"

    narrative = mapper.narrative_for_top_features(
        top_labels=["Domain conservation score", "Protein domain feature"]
    )
    assert "combined effect" in narrative.lower()


def test_false_positive_analyzer_returns_risk_ratio_sorted_hotspots() -> None:
    analyzer = FalsePositiveAnalyzer()
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    groups = pd.Series(["G1", "G1", "G2", "G2", "G1", "G2", "G3", "G3"])

    hotspots = analyzer.analyze(
        y_true=y_true,
        y_pred=y_pred,
        group_values=groups,
        group_name="gene_id",
        top_k=5,
        minimum_negative_count=1,
    )

    assert len(hotspots) > 0
    assert "false_positive_risk_ratio" in hotspots[0]
    assert "gene_id" in hotspots[0]


def test_explainability_schema_to_dict_contract() -> None:
    feature_item = FeatureAttribution(
        feature="revel_score",
        contribution=0.2,
        absolute_contribution=0.2,
        biological_label="Engineered quantitative feature",
    )
    sample_item = SampleExplanation(
        row_index=0,
        score=0.9,
        predicted_label=1,
        top_features=[feature_item],
        narrative="Prediction is mainly driven by engineered quantitative feature.",
    )
    report = ExplainabilityReport(
        backend="proxy",
        global_feature_importance=[feature_item],
        sample_explanations=[sample_item],
        false_positive_hotspots=[],
        metadata={"seed": 42},
    )

    payload = report.to_dict()

    assert payload["backend"] == "proxy"
    assert payload["global_feature_importance"][0]["feature"] == "revel_score"
    assert payload["sample_explanations"][0]["predicted_label"] == 1


def test_service_ignores_missing_optional_group_columns() -> None:
    service = ExplainabilityService(
        config={
            "group_columns": ["gene_id", "domain_id", "protein_family"],
            "false_positive": {
                "enabled": True,
                "top_k_hotspots": 5,
                "minimum_negative_count": 1,
                "group_columns": ["gene_id", "domain_id", "protein_family"],
            },
        },
        seed=42,
    )
    dataset = pd.DataFrame({"gene_id": ["G1", "G1", "G2", "G2"]})

    hotspots = service._build_false_positive_hotspots(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([1, 0, 1, 0]),
        dataset=dataset,
    )

    assert isinstance(hotspots, list)


def test_visualizer_generates_html_document() -> None:
    feature_item = FeatureAttribution(
        feature="revel_score",
        contribution=0.4,
        absolute_contribution=0.4,
        biological_label="Engineered quantitative feature",
    )
    report = ExplainabilityReport(
        backend="proxy",
        global_feature_importance=[feature_item],
        sample_explanations=[
            SampleExplanation(
                row_index=0,
                score=0.75,
                predicted_label=1,
                top_features=[feature_item],
                narrative="Prediction is mainly driven by engineered quantitative feature.",
            )
        ],
        false_positive_hotspots=[],
        metadata={"seed": 42},
    )

    html = ExplainabilityVisualizer().render_html(report)

    assert "<html>" in html
    assert "PathoLogic Explainability Report" in html
    assert "Yontem Ozeti (Turkce)" in html
    assert "revel_score" in html
    assert "relative_strength" in html
    assert "cards" in html
    assert "False-Positive Hotspots" not in html
    assert "Hata Analizi Desenleri" not in html
    assert "false_positive_rate / overall_false_positive_rate" not in html


def test_visualizer_handles_heterogeneous_hotspot_columns() -> None:
    feature_item = FeatureAttribution(
        feature="revel_score",
        contribution=0.3,
        absolute_contribution=0.3,
        biological_label="Engineered quantitative feature",
    )
    report = ExplainabilityReport(
        backend="proxy",
        global_feature_importance=[feature_item],
        sample_explanations=[],
        false_positive_hotspots=[
            {
                "group_column": "gene_id",
                "gene_id": "G1",
                "false_positive_count": 2,
                "negative_count": 4,
                "false_positive_rate": 0.5,
                "overall_false_positive_rate": 0.25,
                "false_positive_risk_ratio": 2.0,
            },
            {
                "group_column": "Protein change",
                "Protein change": "A1708E",
                "false_positive_count": 1,
                "negative_count": 1,
                "false_positive_rate": 1.0,
                "overall_false_positive_rate": 0.25,
                "false_positive_risk_ratio": 4.0,
            },
        ],
        metadata={"seed": 42},
    )

    html = ExplainabilityVisualizer().render_html(report)

    assert "False-Positive Hotspots" not in html
    assert "<th>gene_id</th>" not in html
    assert "<th>Protein change</th>" not in html
    assert "A1708E" not in html


def test_visualizer_renders_member_explainability_section() -> None:
    feature_item = FeatureAttribution(
        feature="revel_score",
        contribution=0.3,
        absolute_contribution=0.3,
        biological_label="Engineered quantitative feature",
    )
    report = ExplainabilityReport(
        backend="proxy",
        global_feature_importance=[feature_item],
        sample_explanations=[],
        false_positive_hotspots=[],
        metadata={"seed": 42},
        member_explainability={
            "status": "ok",
            "members": {
                "tabnet": {
                    "status": "ok",
                    "backend": "proxy",
                    "attribution_diagnostics": {"fallback_reason": "synthetic"},
                    "global_feature_importance": [
                        {
                            "feature": "revel_score",
                            "absolute_contribution": 0.3,
                            "biological_label": "Engineered quantitative feature",
                        }
                    ],
                }
            },
        },
    )

    html = ExplainabilityVisualizer().render_html(report)

    assert "Member Explainability" in html
    assert "tabnet" in html
    assert "backend:" in html
    assert "attribution_diagnostics:" in html


def test_service_builds_member_explainability_for_hybrid_model() -> None:
    service = ExplainabilityService(
        config={
            "backend": "proxy",
            "top_k_features": 3,
            "top_k_samples": 2,
            "false_positive": {"enabled": False},
        },
        seed=42,
    )
    model = _DummyHybridModel()

    x_background = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.1, 0.4, 0.3],
            [0.5, 0.4, 0.2, 0.1],
            [0.6, 0.3, 0.1, 0.2],
        ],
        dtype=float,
    )
    x_target = np.array(
        [
            [0.15, 0.25, 0.35, 0.45],
            [0.22, 0.18, 0.41, 0.29],
            [0.55, 0.38, 0.22, 0.14],
        ],
        dtype=float,
    )
    ensemble_scores = model.predict_proba(x_target)[:, -1]
    y_pred = (ensemble_scores >= 0.5).astype(int)
    y_true = np.array([1, 0, 1], dtype=int)
    dataset = pd.DataFrame({"gene_id": ["G1", "G2", "G3"]})

    report = service.build_report(
        model=model,
        feature_names=["f0", "f1", "f2", "f3"],
        x_background=x_background,
        x_target=x_target,
        y_score=ensemble_scores,
        y_pred=y_pred,
        y_true=y_true,
        dataset=dataset,
    )
    payload = report.to_dict()

    members = payload["member_explainability"]["members"]
    assert payload["member_explainability"]["status"] == "ok"
    assert "tabnet" in members
    assert "xgboost" in members
    assert members["tabnet"]["status"] == "ok"
    assert members["xgboost"]["status"] == "ok"
    assert len(members["tabnet"]["global_feature_importance"]) > 0
    assert len(members["xgboost"]["global_feature_importance"]) > 0
    assert abs(float(members["tabnet"]["weight"]) - 0.6) < 1e-6
    assert abs(float(members["xgboost"]["weight"]) - 0.4) < 1e-6
    assert "effective_member_weights" in payload["metadata"]
    assert "member_weight_scores" in payload["metadata"]

