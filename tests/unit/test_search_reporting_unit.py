from __future__ import annotations

import pytest

from pathologic.search.reporting import _build_split_manifest_warnings, _build_train_report_payload


def test_build_train_report_payload_includes_overfitting_metrics() -> None:
    leaderboard = [
        {
            "candidate": "xgboost",
            "kind": "single",
            "status": "ok",
            "runtime_seconds": 12.3,
            "test_metrics": {"f1": 0.78, "roc_auc": 0.86},
            "hpo": {
                "best_score": 0.91,
                "trials": [
                    {"params": {"max_depth": 4}, "score": 0.90, "fold_scores": [0.88, 0.91, 0.89]},
                    {"params": {"max_depth": 6}, "score": 0.91, "fold_scores": [0.92, 0.90, 0.91]},
                ],
            },
            "calibration": {
                "summary": {
                    "raw": {"status": "ok", "ece": 0.12, "brier_score": 0.20},
                    "platt": {"status": "ok", "ece": 0.09, "brier_score": 0.18},
                }
            },
            "holdout_bootstrap": {
                "status": "ok",
                "metrics": {
                    "f1": {"point_estimate": 0.78, "ci_low": 0.70, "ci_high": 0.84}
                },
            },
            "group_drift": {
                "status": "ok",
                "group_column": "gene_id",
                "group_count": 4,
                "metric_ranges": {
                    "f1": {"range": 0.12, "min_group": "g1", "max_group": "g2"}
                },
            },
            "selected_params_source": "hpo",
        }
    ]

    payload = _build_train_report_payload(
        objective="f1",
        leaderboard=leaderboard,
        best=leaderboard[0],
        objective_best=leaderboard[0],
        elapsed_seconds=30.0,
    )

    candidates = payload.get("candidates")
    assert isinstance(candidates, list) and candidates
    row = candidates[0]

    assert row["cv_objective_score"] == 0.91
    assert row["test_objective_score"] == 0.78
    assert row["generalization_gap"] == 0.13
    assert row["overfitting_risk_level"] == "high"
    assert row["overfitting_suspected"] is True
    assert row["calibration_ece_improvement"] == 0.03
    assert row["calibration_brier_improvement"] == pytest.approx(0.02)
    assert row["fold_distribution"]["status"] == "ok"
    assert row["learning_curve"]["status"] == "ok"
    reliability_sections = payload.get("reliability_sections")
    assert isinstance(reliability_sections, dict)
    assert reliability_sections["fold_distribution"]["available_candidates"] == 1
    assert reliability_sections["holdout_bootstrap"]["available_candidates"] == 1
    assert reliability_sections["group_drift"]["available_candidates"] == 1
    assert reliability_sections["learning_curve"]["available_candidates"] == 1
    split_warnings = payload.get("split_manifest_warnings")
    assert isinstance(split_warnings, dict)
    assert split_warnings.get("status") == "ok"


def test_build_train_report_payload_marks_unknown_overfitting_when_scores_missing() -> None:
    leaderboard = [
        {
            "candidate": "random_forest",
            "kind": "single",
            "status": "ok",
            "runtime_seconds": 5.0,
            "test_metrics": {},
            "hpo": {"status": "skipped"},
            "selected_params_source": "defaults",
        }
    ]

    payload = _build_train_report_payload(
        objective="f1",
        leaderboard=leaderboard,
        best=leaderboard[0],
        objective_best=leaderboard[0],
        elapsed_seconds=10.0,
    )

    row = payload["candidates"][0]
    assert row["cv_objective_score"] is None
    assert row["test_objective_score"] is None
    assert row["generalization_gap"] is None
    assert row["overfitting_risk_level"] == "unknown"
    assert row["overfitting_suspected"] is False


def test_build_split_manifest_warnings_reports_missing_and_nonzero_leakage_fields() -> None:
    warnings_payload = _build_split_manifest_warnings(
        split_summary={
            "train_val_shared_genes": 0,
            "train_test_shared_genes": 2,
            "train_size": 10,
            "val_size": 4,
            "test_size": 3,
        },
        outer_train_rows=10,
        outer_calibration_rows=4,
        outer_test_rows=3,
    )

    assert warnings_payload["status"] == "error"
    assert warnings_payload["has_errors"] is True
    items = warnings_payload["items"]
    assert isinstance(items, list)
    codes = {str(item.get("code")) for item in items if isinstance(item, dict)}
    assert "SPLIT_LEAKAGE_NONZERO" in codes
    assert "SPLIT_LEAKAGE_FIELD_MISSING" in codes


def test_build_split_manifest_warnings_reports_rowcount_mismatch() -> None:
    warnings_payload = _build_split_manifest_warnings(
        split_summary={
            "train_val_shared_genes": 0,
            "train_test_shared_genes": 0,
            "val_test_shared_genes": 0,
            "train_size": 9,
            "val_size": 4,
            "test_size": 3,
        },
        outer_train_rows=10,
        outer_calibration_rows=4,
        outer_test_rows=3,
    )

    assert warnings_payload["status"] == "warning"
    codes = {
        str(item.get("code"))
        for item in warnings_payload["items"]
        if isinstance(item, dict)
    }
    assert "SPLIT_ROWCOUNT_MISMATCH" in codes
