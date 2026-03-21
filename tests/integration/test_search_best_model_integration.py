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
        "data/raw/data.csv",
        "--output-dir",
        str(output_dir),
        "--budget-profile",
        "quick",
        "--tune-engine",
        "random",
        "--models",
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
    first_row = rows[0]
    calibration = first_row.get("calibration")
    assert isinstance(calibration, dict)
    assert "methods" in calibration
    artifacts = calibration.get("artifacts", {})
    assert isinstance(artifacts, dict)
    for key in (
        "histogram_png",
        "reliability_png",
        "qq_plot_png",
        "reliability_bins_csv",
        "calibration_report_json",
    ):
        assert key in artifacts
        assert Path(str(artifacts[key])).exists()

    calibration_summary_path = run_dir / "calibration_summary.json"
    assert calibration_summary_path.exists()
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


def test_search_best_model_hybrid_hpo_includes_member_regularization_params(tmp_path: Path) -> None:
    output_dir = tmp_path / "model_search_reg_out"
    cmd = [
        sys.executable,
        "scripts/search_best_model.py",
        "data/raw/data.csv",
        "--output-dir",
        str(output_dir),
        "--budget-profile",
        "quick",
        "--tune-engine",
        "random",
        "--model-pool",
        "xgboost,catboost",
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
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    rows = leaderboard.get("rows", [])
    assert isinstance(rows, list) and rows

    hybrid_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("kind") == "hybrid_pair"
        and set(str(row.get("candidate", "")).split("+")) == {"catboost", "xgboost"}
    ]
    assert hybrid_rows, "Expected catboost+xgboost hybrid row"

    row = hybrid_rows[0]
    hpo = row.get("hpo")
    assert isinstance(hpo, dict)
    best_params = hpo.get("best_params")
    assert isinstance(best_params, dict)
    assert "member__xgboost__reg_alpha" in best_params
    assert "member__xgboost__reg_lambda" in best_params
    assert "member__catboost__l2_leaf_reg" in best_params
