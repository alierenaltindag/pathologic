"""Unit tests for evaluator metric and hotspot reporting."""

from __future__ import annotations

import numpy as np

from pathologic.engine import Evaluator


def test_evaluator_computes_global_metrics_and_grouped_hotspots() -> None:
    y_true = np.array([1, 0, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1])
    groups = np.array(["G1", "G1", "G2", "G2", "G3", "G3"])

    evaluator = Evaluator(metric_names=["roc_auc", "auprc", "f1", "mcc", "precision", "recall"])
    report = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        group_values=groups,
        group_name="gene_id",
        top_k_hotspots=5,
    )

    assert "f1" in report.metrics
    assert "roc_auc" in report.metrics
    assert set(report.grouped_metrics.keys()) == {"G1", "G2", "G3"}
    assert len(report.false_positive_hotspots) > 0
    assert "gene_id" in report.false_positive_hotspots[0]


def test_evaluator_to_dict_is_serializable_shape() -> None:
    evaluator = Evaluator()
    report = evaluator.evaluate(
        y_true=np.array([0, 1, 0, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        y_score=np.array([0.1, 0.9, 0.8, 0.7]),
    )

    payload = report.to_dict()

    assert "metrics" in payload
    assert "grouped_metrics" in payload
    assert "false_positive_hotspots" in payload
