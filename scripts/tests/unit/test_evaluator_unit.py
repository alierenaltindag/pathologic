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


def test_panel_oof_f1_max_thresholds_returns_per_panel_best_threshold() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.array([0.10, 0.90, 0.30, 0.70, 0.20, 0.80])
    panels = np.array(["P1", "P1", "P1", "P2", "P2", "P2"])

    rows = Evaluator.panel_oof_f1_max_thresholds(
        y_true=y_true,
        y_score=y_score,
        panel_values=panels,
        min_samples=1,
        default_threshold=0.5,
    )

    assert [row["panel"] for row in rows] == ["P1", "P2"]
    p1 = next(row for row in rows if row["panel"] == "P1")
    p2 = next(row for row in rows if row["panel"] == "P2")

    assert p1["optimized"] == 1
    assert p2["optimized"] == 1
    assert float(p1["threshold"]) in {0.3, 0.9}
    assert float(p2["threshold"]) in {0.7, 0.8}
    assert float(p1["f1"]) == 1.0
    assert float(p2["f1"]) == 1.0


def test_panel_oof_f1_max_thresholds_falls_back_when_single_class_panel() -> None:
    y_true = np.array([1, 1, 1, 0, 1])
    y_score = np.array([0.95, 0.70, 0.20, 0.80, 0.60])
    panels = np.array(["P1", "P1", "P1", "P2", "P2"])

    rows = Evaluator.panel_oof_f1_max_thresholds(
        y_true=y_true,
        y_score=y_score,
        panel_values=panels,
        min_samples=1,
        default_threshold=0.5,
    )

    p1 = next(row for row in rows if row["panel"] == "P1")
    assert p1["optimized"] == 0
    assert float(p1["threshold"]) == 0.5
