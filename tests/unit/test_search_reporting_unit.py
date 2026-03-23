from __future__ import annotations

from pathologic.search.reporting import _build_split_manifest_warnings, _build_train_report_payload


def test_build_train_report_payload_includes_overfitting_metrics() -> None:
    leaderboard = [
        {
            "candidate": "xgboost",
            "kind": "single",
            "status": "ok",
            "runtime_seconds": 12.3,
            "test_metrics": {"f1": 0.78, "roc_auc": 0.86},
            "hpo": {"best_score": 0.91},
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
