"""Biological mapping helpers for explainability narratives."""

from __future__ import annotations


class BiologicalMapper:
    """Map technical feature names to biological labels and narrative text."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._mapping = dict(mapping or {})

    def label_for(self, feature_name: str) -> str:
        if feature_name in self._mapping:
            return str(self._mapping[feature_name])

        normalized = feature_name.strip().lower()
        if normalized.startswith("gene") or normalized.endswith("_gene"):
            return "Gene-level signal"
        if "domain" in normalized:
            return "Protein domain feature"
        if "idr" in normalized:
            return "Intrinsic disorder region feature"
        if normalized.startswith("feat_"):
            return "Engineered quantitative feature"
        return "General biological feature"

    def narrative_for_top_features(self, *, top_labels: list[str]) -> str:
        if not top_labels:
            return "Prediction is not strongly driven by a dominant biological feature group."

        unique_labels = list(dict.fromkeys(top_labels))
        if len(unique_labels) == 1:
            return f"Prediction is mainly driven by {unique_labels[0].lower()}."

        if len(unique_labels) == 2:
            return (
                "Prediction reflects a combined effect from "
                f"{unique_labels[0].lower()} and {unique_labels[1].lower()}."
            )

        leading = ", ".join(label.lower() for label in unique_labels[:2])
        return (
            "Prediction is influenced by multiple biological signals, primarily "
            f"{leading}, and additional supporting factors."
        )
