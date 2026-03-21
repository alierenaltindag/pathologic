"""High-level explainability service orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pathologic.explain.biological_mapper import BiologicalMapper
from pathologic.explain.false_positive_analyzer import FalsePositiveAnalyzer
from pathologic.explain.schemas import ExplainabilityReport, FeatureAttribution, SampleExplanation
from pathologic.explain.shap_engine import ShapAttributionEngine


class ExplainabilityService:
    """Build explainability reports from trained models and prepared features."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        seed: int,
    ) -> None:
        self.config = dict(config)
        self.seed = int(seed)

        mapper_config_raw = self.config.get("biological_mapping")
        mapper_config = mapper_config_raw if isinstance(mapper_config_raw, dict) else {}
        self._mapper = BiologicalMapper(mapping={str(k): str(v) for k, v in mapper_config.items()})

        self._engine = ShapAttributionEngine(
            backend=str(self.config.get("backend", "auto")),
            background_size=int(self.config.get("background_size", 100)),
            random_state=self.seed,
        )
        self._fp_analyzer = FalsePositiveAnalyzer()

    def build_report(
        self,
        *,
        model: Any,
        feature_names: list[str],
        x_background: np.ndarray,
        x_target: np.ndarray,
        y_score: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        dataset: pd.DataFrame,
    ) -> ExplainabilityReport:
        attribution = self._engine.compute(
            model=model,
            x_background=x_background,
            x_target=x_target,
        )

        top_k_features = max(int(self.config.get("top_k_features", 5)), 1)
        top_k_samples = max(int(self.config.get("top_k_samples", 10)), 1)

        global_items = self._build_global_importance(
            feature_names=feature_names,
            importances=attribution.global_importance,
            top_k=top_k_features,
        )
        sample_items = self._build_sample_explanations(
            feature_names=feature_names,
            contributions=attribution.contributions,
            y_score=y_score,
            y_pred=y_pred,
            top_k_features=top_k_features,
            top_k_samples=top_k_samples,
        )

        resolved_group_columns = self._resolve_fp_group_columns(dataset)

        hotspots = self._build_false_positive_hotspots(
            y_true=y_true,
            y_pred=y_pred,
            dataset=dataset,
            group_columns=resolved_group_columns,
        )
        member_explainability = self._build_member_explainability(
            model=model,
            feature_names=feature_names,
            x_background=x_background,
            x_target=x_target,
            top_k_features=top_k_features,
            top_k_samples=top_k_samples,
        )

        metadata = {
            "seed": self.seed,
            "backend_policy": str(self.config.get("backend", "auto")),
            "resolved_backend": attribution.backend,
            "background_size": int(self.config.get("background_size", 100)),
            "top_k_features": top_k_features,
            "top_k_samples": top_k_samples,
            "sample_count": int(x_target.shape[0]),
            "feature_count": int(x_target.shape[1]),
            "group_columns": resolved_group_columns,
        }
        if attribution.diagnostics:
            metadata["attribution_diagnostics"] = dict(attribution.diagnostics)
        if isinstance(member_explainability, dict):
            member_weights = member_explainability.get("effective_member_weights")
            if isinstance(member_weights, dict):
                metadata["effective_member_weights"] = {
                    str(alias): float(value)
                    for alias, value in member_weights.items()
                }

        return ExplainabilityReport(
            backend=attribution.backend,
            global_feature_importance=global_items,
            sample_explanations=sample_items,
            false_positive_hotspots=hotspots,
            metadata=metadata,
            member_explainability=member_explainability or None,
        )

    def _build_member_explainability(
        self,
        *,
        model: Any,
        feature_names: list[str],
        x_background: np.ndarray,
        x_target: np.ndarray,
        top_k_features: int,
        top_k_samples: int,
    ) -> dict[str, Any]:
        aliases_raw = getattr(model, "member_aliases", None)
        models_raw = getattr(model, "_member_models", None)
        if not isinstance(aliases_raw, list) or not isinstance(models_raw, list):
            return {}
        if len(aliases_raw) != len(models_raw) or not aliases_raw:
            return {}

        results: dict[str, Any] = {"status": "ok", "members": {}}
        member_weights = self._resolve_member_weights(model)
        if member_weights:
            results["effective_member_weights"] = dict(member_weights)
        member_scores = self._resolve_member_weight_scores(model)
        if member_scores:
            results["member_weight_scores"] = dict(member_scores)

        for alias, member_model in zip(aliases_raw, models_raw, strict=True):
            member_alias = str(alias)
            member_x_target, member_features = self._resolve_member_input(
                model=model,
                alias=member_alias,
                x_values=x_target,
                feature_names=feature_names,
            )
            member_x_background, _ = self._resolve_member_input(
                model=model,
                alias=member_alias,
                x_values=x_background,
                feature_names=feature_names,
            )
            if member_x_target.size == 0 or member_x_background.size == 0:
                results["members"][member_alias] = {
                    "status": "skipped",
                    "reason": "empty_member_input",
                }
                continue

            attribution = self._engine.compute(
                model=member_model,
                x_background=member_x_background,
                x_target=member_x_target,
            )
            probabilities = np.asarray(member_model.predict_proba(member_x_target))
            member_scores = probabilities if probabilities.ndim == 1 else probabilities[:, -1]
            member_preds = (np.asarray(member_scores, dtype=float) >= 0.5).astype(int)

            member_payload: dict[str, Any] = {
                "status": "ok",
                "backend": attribution.backend,
                "feature_count": int(member_x_target.shape[1]),
                "global_feature_importance": [
                    item.to_dict()
                    for item in self._build_global_importance(
                        feature_names=member_features,
                        importances=attribution.global_importance,
                        top_k=top_k_features,
                    )
                ],
                "sample_explanations": [
                    item.to_dict()
                    for item in self._build_sample_explanations(
                        feature_names=member_features,
                        contributions=attribution.contributions,
                        y_score=np.asarray(member_scores, dtype=float),
                        y_pred=member_preds,
                        top_k_features=top_k_features,
                        top_k_samples=top_k_samples,
                    )
                ],
            }
            if attribution.diagnostics:
                member_payload["attribution_diagnostics"] = dict(attribution.diagnostics)
            if member_alias in member_weights:
                member_payload["weight"] = float(member_weights[member_alias])
            if member_alias in member_scores:
                member_payload["weight_score"] = float(member_scores[member_alias])
            results["members"][member_alias] = member_payload

        return results

    @staticmethod
    def _resolve_member_input(
        *,
        model: Any,
        alias: str,
        x_values: np.ndarray,
        feature_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        if hasattr(model, "_member_input"):
            try:
                member_x = np.asarray(model._member_input(alias, x_values))  # noqa: SLF001
                if member_x.ndim == 2 and member_x.shape[1] > 0:
                    index_map_raw = getattr(model, "_member_feature_indices", None)
                    if isinstance(index_map_raw, dict) and alias in index_map_raw:
                        indices = np.asarray(index_map_raw[alias], dtype=int)
                        member_features = [feature_names[int(idx)] for idx in indices]
                    else:
                        member_features = list(feature_names)
                    return member_x, member_features
            except Exception:
                pass
        return np.asarray(x_values), list(feature_names)

    @staticmethod
    def _resolve_member_weights(model: Any) -> dict[str, float]:
        if not hasattr(model, "effective_member_weights"):
            return {}
        try:
            raw = model.effective_member_weights()
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(alias): float(value)
            for alias, value in raw.items()
        }

    @staticmethod
    def _resolve_member_weight_scores(model: Any) -> dict[str, float]:
        if not hasattr(model, "member_weight_scores"):
            return {}
        try:
            raw = model.member_weight_scores()
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(alias): float(value)
            for alias, value in raw.items()
        }

    def _build_global_importance(
        self,
        *,
        feature_names: list[str],
        importances: np.ndarray,
        top_k: int,
    ) -> list[FeatureAttribution]:
        ranked = sorted(
            zip(feature_names, importances, strict=True),
            key=lambda value: float(abs(value[1])),
            reverse=True,
        )
        selected = ranked[:top_k]

        return [
            FeatureAttribution(
                feature=name,
                contribution=float(value),
                absolute_contribution=float(abs(value)),
                biological_label=self._mapper.label_for(name),
            )
            for name, value in selected
        ]

    def _build_sample_explanations(
        self,
        *,
        feature_names: list[str],
        contributions: np.ndarray,
        y_score: np.ndarray,
        y_pred: np.ndarray,
        top_k_features: int,
        top_k_samples: int,
    ) -> list[SampleExplanation]:
        sample_count = min(contributions.shape[0], top_k_samples)
        output: list[SampleExplanation] = []

        for row_index in range(sample_count):
            row_values = contributions[row_index]
            ranked_indices = np.argsort(np.abs(row_values))[::-1][:top_k_features]

            top_features: list[FeatureAttribution] = []
            top_labels: list[str] = []
            for feature_index in ranked_indices:
                feature_name = feature_names[int(feature_index)]
                label = self._mapper.label_for(feature_name)
                top_labels.append(label)
                contribution = float(row_values[int(feature_index)])
                top_features.append(
                    FeatureAttribution(
                        feature=feature_name,
                        contribution=contribution,
                        absolute_contribution=float(abs(contribution)),
                        biological_label=label,
                    )
                )

            output.append(
                SampleExplanation(
                    row_index=row_index,
                    score=float(y_score[row_index]),
                    predicted_label=int(y_pred[row_index]),
                    top_features=top_features,
                    narrative=self._mapper.narrative_for_top_features(top_labels=top_labels),
                )
            )

        return output

    def _build_false_positive_hotspots(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset: pd.DataFrame,
        group_columns: list[str] | None = None,
    ) -> list[dict[str, float | int | str]]:
        fp_config_raw = self.config.get("false_positive")
        fp_config = fp_config_raw if isinstance(fp_config_raw, dict) else {}
        if not bool(fp_config.get("enabled", True)):
            return []

        effective_group_columns = (
            list(group_columns)
            if isinstance(group_columns, list)
            else self._resolve_fp_group_columns(dataset)
        )

        top_k = int(fp_config.get("top_k_hotspots", 10))
        minimum_negative_count = int(fp_config.get("minimum_negative_count", 1))

        results: list[dict[str, float | int | str]] = []
        for column in effective_group_columns:
            if column not in dataset.columns:
                continue
            grouped = self._fp_analyzer.analyze(
                y_true=y_true,
                y_pred=y_pred,
                group_values=dataset[column].astype(str),
                group_name=column,
                top_k=top_k,
                minimum_negative_count=minimum_negative_count,
            )
            for item in grouped:
                enriched = dict(item)
                enriched["group_column"] = column
                results.append(enriched)

        results.sort(
            key=lambda value: float(value.get("false_positive_risk_ratio", 0.0)),
            reverse=True,
        )
        return results[:top_k]

    def _resolve_fp_group_columns(self, dataset: pd.DataFrame) -> list[str]:
        fp_config_raw = self.config.get("false_positive")
        fp_config = fp_config_raw if isinstance(fp_config_raw, dict) else {}
        group_columns_raw = fp_config.get("group_columns", self.config.get("group_columns", []))
        candidates = [str(item) for item in group_columns_raw] if isinstance(group_columns_raw, list) else []

        seen: set[str] = set()
        resolved: list[str] = []
        for column in candidates:
            if column in dataset.columns and column not in seen:
                resolved.append(column)
                seen.add(column)
        return resolved
