"""Evaluation utilities for model metrics and grouped error analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationReport:
    """Serializable evaluation output."""

    metrics: dict[str, float]
    grouped_metrics: dict[str, dict[str, float]]
    false_positive_hotspots: list[dict[str, float | int | str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "grouped_metrics": {k: dict(v) for k, v in self.grouped_metrics.items()},
            "false_positive_hotspots": [dict(item) for item in self.false_positive_hotspots],
        }


class Evaluator:
    """Compute global and grouped classification metrics."""

    DEFAULT_METRICS = ("roc_auc", "auprc", "f1", "mcc", "precision", "recall")

    def __init__(self, metric_names: list[str] | None = None) -> None:
        self.metric_names = metric_names or list(self.DEFAULT_METRICS)

    def evaluate(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
        group_values: pd.Series | np.ndarray | None = None,
        group_name: str = "group",
        top_k_hotspots: int = 10,
    ) -> EvaluationReport:
        """Evaluate predictions and return a rich report."""
        y_true_arr = np.asarray(y_true).reshape(-1)
        y_pred_arr = np.asarray(y_pred).reshape(-1)
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")

        y_score_arr: np.ndarray | None = None
        if y_score is not None:
            y_score_arr = np.asarray(y_score).reshape(-1)
            if y_score_arr.shape[0] != y_true_arr.shape[0]:
                raise ValueError("y_score must have the same length as y_true.")

        global_metrics = self._compute_metrics(y_true_arr, y_pred_arr, y_score_arr)

        grouped: dict[str, dict[str, float]] = {}
        hotspots: list[dict[str, float | int | str]] = []
        if group_values is not None:
            groups = pd.Series(group_values).astype(str)
            grouped = self._compute_grouped_metrics(
                y_true=y_true_arr,
                y_pred=y_pred_arr,
                y_score=y_score_arr,
                groups=groups,
            )
            hotspots = self._false_positive_hotspots(
                y_true=y_true_arr,
                y_pred=y_pred_arr,
                groups=groups,
                group_name=group_name,
                top_k=top_k_hotspots,
            )

        return EvaluationReport(
            metrics=global_metrics,
            grouped_metrics=grouped,
            false_positive_hotspots=hotspots,
        )

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if "f1" in self.metric_names:
            metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        if "mcc" in self.metric_names:
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        if "precision" in self.metric_names:
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        if "recall" in self.metric_names:
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        if "specificity" in self.metric_names:
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        if "sensitivity" in self.metric_names:
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        if y_score is not None:
            if "roc_auc" in self.metric_names:
                metrics["roc_auc"] = self._safe_auc(y_true, y_score)
            if "auprc" in self.metric_names:
                metrics["auprc"] = self._safe_auprc(y_true, y_score)

        return metrics

    def _compute_grouped_metrics(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None,
        groups: pd.Series,
    ) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for value in groups.unique():
            mask = groups == value
            y_true_g = y_true[mask.to_numpy()]
            y_pred_g = y_pred[mask.to_numpy()]
            y_score_g = y_score[mask.to_numpy()] if y_score is not None else None
            output[str(value)] = self._compute_metrics(y_true_g, y_pred_g, y_score_g)
        return output

    @staticmethod
    def _false_positive_hotspots(
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: pd.Series,
        group_name: str,
        top_k: int,
    ) -> list[dict[str, float | int | str]]:
        result: list[dict[str, float | int | str]] = []
        frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
        for value, subset in frame.groupby("group"):
            negatives = int((subset["y_true"] == 0).sum())
            if negatives == 0:
                continue
            false_positives = int(((subset["y_true"] == 0) & (subset["y_pred"] == 1)).sum())
            fp_rate = float(false_positives / negatives)
            if false_positives > 0:
                result.append(
                    {
                        group_name: str(value),
                        "false_positive_count": false_positives,
                        "negative_count": negatives,
                        "false_positive_rate": fp_rate,
                    }
                )

        result.sort(key=lambda item: float(item["false_positive_rate"]), reverse=True)
        return result[:top_k]

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
