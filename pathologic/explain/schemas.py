"""Serializable schemas for explainability outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureAttribution:
    """Per-feature attribution item."""

    feature: str
    contribution: float
    absolute_contribution: float
    biological_label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "contribution": float(self.contribution),
            "absolute_contribution": float(self.absolute_contribution),
            "biological_label": self.biological_label,
        }


@dataclass(frozen=True)
class SampleExplanation:
    """Sample-level explanation payload."""

    row_index: int
    score: float
    predicted_label: int
    top_features: list[FeatureAttribution]
    narrative: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": int(self.row_index),
            "score": float(self.score),
            "predicted_label": int(self.predicted_label),
            "top_features": [item.to_dict() for item in self.top_features],
            "narrative": self.narrative,
        }


@dataclass(frozen=True)
class ExplainabilityReport:
    """Top-level explainability report."""

    backend: str
    global_feature_importance: list[FeatureAttribution]
    sample_explanations: list[SampleExplanation]
    false_positive_hotspots: list[dict[str, float | int | str]]
    metadata: dict[str, Any]
    member_explainability: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "global_feature_importance": [
                item.to_dict() for item in self.global_feature_importance
            ],
            "sample_explanations": [item.to_dict() for item in self.sample_explanations],
            "false_positive_hotspots": [dict(item) for item in self.false_positive_hotspots],
            "metadata": dict(self.metadata),
        }
        if self.member_explainability is not None:
            payload["member_explainability"] = dict(self.member_explainability)
        return payload
