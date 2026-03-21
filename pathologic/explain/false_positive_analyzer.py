"""False-positive hotspot analysis helpers for explainability flows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathologic.engine.evaluator import Evaluator


class FalsePositiveAnalyzer:
    """Compute group-wise false-positive hotspots and relative risk style enrichment."""

    def analyze(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group_values: pd.Series,
        group_name: str,
        top_k: int,
        minimum_negative_count: int,
    ) -> list[dict[str, float | int | str]]:
        raw_hotspots = Evaluator._false_positive_hotspots(
            y_true=y_true,
            y_pred=y_pred,
            groups=group_values.astype(str),
            group_name=group_name,
            top_k=max(top_k * 3, top_k),
        )

        overall_negative = int((y_true == 0).sum())
        if overall_negative == 0:
            return []
        overall_false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
        overall_fp_rate = float(overall_false_positive / overall_negative)

        enriched: list[dict[str, float | int | str]] = []
        for item in raw_hotspots:
            negative_count = int(item.get("negative_count", 0))
            if negative_count < minimum_negative_count:
                continue

            fp_rate = float(item.get("false_positive_rate", 0.0))
            risk_ratio = float(fp_rate / overall_fp_rate) if overall_fp_rate > 0 else float("inf")

            enriched_item = dict(item)
            enriched_item["overall_false_positive_rate"] = overall_fp_rate
            enriched_item["false_positive_risk_ratio"] = risk_ratio
            enriched.append(enriched_item)

        enriched.sort(key=lambda value: float(value["false_positive_risk_ratio"]), reverse=True)
        return enriched[:top_k]
