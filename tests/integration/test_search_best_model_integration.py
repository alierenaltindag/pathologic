from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_search_best_model_script_generates_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "model_search_out"
    cmd = [
        sys.executable,
        "scripts/search_best_model.py",
        "data.csv",
        "--output-dir",
        str(output_dir),
        "--budget-profile",
        "quick",
        "--tune-engine",
        "random",
        "--model-pool",
        "logreg,random_forest",
        "--max-candidates",
        "3",
        "--n-trials",
        "1",
        "--nas-candidates",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + "\n" + result.stdout

    run_dirs = sorted(output_dir.glob("search_*"))
    assert run_dirs, "Expected at least one run directory"
    run_dir = run_dirs[-1]

    leaderboard_path = run_dir / "leaderboard.json"
    best_path = run_dir / "best_model_summary.json"
    split_path = run_dir / "split_manifest.json"

    assert leaderboard_path.exists()
    assert best_path.exists()
    assert split_path.exists()

    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))

    assert leaderboard["objective"] == "f1"
    assert leaderboard["candidates_total"] == 3
    assert isinstance(best["winner"]["candidate"], str)

    rows = leaderboard.get("rows", [])
    assert isinstance(rows, list)
    assert rows, "Expected non-empty leaderboard rows"

    # Bu kosuda yalnizca non-neural modeller var; NAS policy geregi skip olmalidir.
    for row in rows:
        if not isinstance(row, dict):
            continue
        assert row.get("kind") in {"single", "hybrid_pair"}
        nas_payload = row.get("nas")
        assert isinstance(nas_payload, dict)
        assert nas_payload.get("status") == "skipped"
        assert nas_payload.get("reason") == "model_family_policy"

    first_row = rows[0]
    calibration = first_row.get("calibration")
    assert isinstance(calibration, dict)
    assert "methods" in calibration
    summary = calibration.get("summary")
    assert isinstance(summary, dict)
    for method_name in ("raw", "platt", "beta", "isotonic", "temperature"):
        assert method_name in summary
        method_summary = summary[method_name]
        assert isinstance(method_summary, dict)
        assert method_summary.get("status") in {"ok", "failed"}

    artifacts = calibration.get("artifacts", {})
    assert isinstance(artifacts, dict)
    for key in (
        "histogram_png",
        "reliability_png",
        "qq_plot_png",
        "reliability_bins_csv",
        "calibration_report_json",
        "calibration_report_html",
    ):
        assert key in artifacts
        assert Path(str(artifacts[key])).exists()

    panel_thresholds = first_row.get("panel_thresholds")
    assert isinstance(panel_thresholds, dict)
    assert panel_thresholds.get("status") in {"ok", "skipped", "failed"}
    if panel_thresholds.get("status") == "ok":
        panel_artifacts = panel_thresholds.get("artifacts", {})
        assert isinstance(panel_artifacts, dict)
        for key in (
            "panel_thresholds_report_json",
            "panel_thresholds_report_html",
        ):
            assert key in panel_artifacts
            assert Path(str(panel_artifacts[key])).exists()

    calibration_summary_path = run_dir / "calibration_summary.json"
    assert calibration_summary_path.exists()
    calibration_summary_html_path = run_dir / "calibration_summary.html"
    assert calibration_summary_html_path.exists()
    calibration_summary = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    assert "rows" in calibration_summary

    error_analysis_summary_path = run_dir / "error_analysis_summary.json"
    assert error_analysis_summary_path.exists()
    error_analysis_summary = json.loads(error_analysis_summary_path.read_text(encoding="utf-8"))
    assert isinstance(error_analysis_summary, dict)
    assert isinstance(error_analysis_summary.get("winner_candidate"), str)
    ea_rows = error_analysis_summary.get("rows")
    assert isinstance(ea_rows, list)
    assert ea_rows, "Expected non-empty error analysis summary rows"

    winner_candidate = best["winner"]["candidate"]
    summary_winner_rows = [row for row in ea_rows if row.get("is_winner") is True]
    assert len(summary_winner_rows) == 1
    assert summary_winner_rows[0].get("candidate") == winner_candidate

    leaderboard_candidates = {
        row.get("candidate")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("candidate"), str)
    }
    summary_candidates = {
        row.get("candidate")
        for row in ea_rows
        if isinstance(row, dict) and isinstance(row.get("candidate"), str)
    }
    assert summary_candidates.issubset(leaderboard_candidates)

    for row in ea_rows:
        assert row.get("status") in {"ok", "failed", "skipped"}
        assert isinstance(row.get("error_count"), int)
        assert isinstance(row.get("error_rate"), float)
        if row.get("status") == "ok":
            assert row.get("surrogate_status") in {"ok", "failed", "skipped"}
            assert row.get("clustering_status") in {"ok", "failed", "skipped"}

    explain_rows = [row for row in rows if isinstance(row, dict)]
    assert explain_rows, "Expected rows for explainability checks"
    for row in explain_rows:
        explainability = row.get("explainability")
        assert isinstance(explainability, dict)
        assert explainability.get("status") in {"ok", "failed", "skipped"}
        if explainability.get("status") != "ok":
            continue

        artifacts = explainability.get("artifacts")
        assert isinstance(artifacts, dict)
        for key in (
            "explainability_report_json",
            "explainability_summary_json",
            "explainability_report_html",
            "false_positive_hotspots_csv",
            "global_feature_importance_png",
            "false_positive_hotspots_png",
        ):
            assert key in artifacts
            assert Path(str(artifacts[key])).exists()

        compute_cost = row.get("compute_cost")
        assert isinstance(compute_cost, dict)
        assert compute_cost.get("status") in {"enabled", "skipped"}
        if compute_cost.get("status") != "enabled":
            continue

        artifacts = compute_cost.get("artifacts")
        assert isinstance(artifacts, dict)
        for key in ("compute_cost_report_json", "compute_cost_report_html"):
            assert key in artifacts
            assert Path(str(artifacts[key])).exists()

        training = compute_cost.get("training")
        assert isinstance(training, dict)
        assert isinstance(training.get("train_total_seconds"), float)

        inference = compute_cost.get("inference")
        assert isinstance(inference, dict)
        if inference.get("status") != "failed":
            assert isinstance(inference.get("single_sample_ms"), float)
            assert isinstance(inference.get("batch_total_ms"), float)

    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("kind") != "hybrid_pair":
            continue
        hybrid_config = row.get("hybrid_config")
        assert isinstance(hybrid_config, dict)
        assert isinstance(hybrid_config.get("strategy"), str)
        params = hybrid_config.get("params")
        assert isinstance(params, dict)
        assert "weighting_policy" in params

    if isinstance(best.get("winner"), dict) and best["winner"].get("kind") == "hybrid_pair":
        assert isinstance(best.get("winner_hybrid_config"), dict)

    summary = split["outer_split_summary"]
    for key in ("train_test_shared_genes", "train_val_shared_genes", "val_test_shared_genes"):
        if key in summary:
            assert int(summary[key]) == 0

    train_report_path = run_dir / "train_report.json"
    train_report_html_path = run_dir / "train_report.html"
    assert train_report_path.exists()
    assert train_report_html_path.exists()

    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    assert train_report.get("objective") == "f1"
    assert isinstance(train_report.get("candidates"), list)
    assert train_report["candidates"], "Expected candidate quick summary rows"
    assert isinstance(train_report.get("overfitting_policy"), dict)
    split_manifest_warnings = train_report.get("split_manifest_warnings")
    assert isinstance(split_manifest_warnings, dict)
    assert split_manifest_warnings.get("status") == "ok"
    assert split_manifest_warnings.get("warning_count") == 0

    winner_info = train_report.get("winner")
    assert isinstance(winner_info, dict)
    assert winner_info.get("candidate") == winner_candidate
    assert isinstance(winner_info.get("selected_params"), dict)
    assert isinstance(winner_info.get("test_metrics"), dict)
    assert "overfitting_risk_level" in winner_info
    assert "generalization_gap" in winner_info

    candidate_summary = train_report["candidates"][0]
    assert "cv_objective_score" in candidate_summary
    assert "test_objective_score" in candidate_summary
    assert "generalization_gap" in candidate_summary
    assert "generalization_gap_ratio" in candidate_summary
    assert "overfitting_risk_level" in candidate_summary
    assert "overfitting_suspected" in candidate_summary

    train_report_html = train_report_html_path.read_text(encoding="utf-8")
    assert "Split Manifest Warnings" in train_report_html


def test_search_best_model_hybrid_runs_nas_only_for_neural_member(tmp_path: Path) -> None:
    output_dir = tmp_path / "model_search_hybrid_nas_out"
    cmd = [
        sys.executable,
        "scripts/search_best_model.py",
        "data.csv",
        "--output-dir",
        str(output_dir),
        "--budget-profile",
        "quick",
        "--tune-engine",
        "random",
        "--model-pool",
        "tabnet,xgboost",
        "--max-candidates",
        "3",
        "--n-trials",
        "1",
        "--nas-candidates",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + "\n" + result.stdout

    run_dirs = sorted(output_dir.glob("search_*"))
    assert run_dirs, "Expected at least one run directory"
    run_dir = run_dirs[-1]

    leaderboard = json.loads((run_dir / "leaderboard.json").read_text(encoding="utf-8"))
    rows = leaderboard.get("rows", [])
    assert isinstance(rows, list) and rows

    hybrid_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("kind") == "hybrid_pair"
        and set(str(row.get("candidate", "")).split("+")) == {"tabnet", "xgboost"}
    ]
    assert hybrid_rows, "Expected tabnet+xgboost hybrid row"

    row = hybrid_rows[0]
    nas = row.get("nas")
    assert isinstance(nas, dict)
    assert nas.get("status") == "ok"
    nas_best_params = nas.get("best_params")
    assert isinstance(nas_best_params, dict)
    assert any(str(key).startswith("member__tabnet__") for key in nas_best_params)
    assert all(not str(key).startswith("member__xgboost__") for key in nas_best_params)

    assert isinstance(row.get("hpo_level1"), dict)
    assert isinstance(row.get("hpo_level2"), dict)
