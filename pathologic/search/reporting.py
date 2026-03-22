"""Leaderboard ranking and report serialization helpers for search workflows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pathologic.explain.visualizer import ExplainabilityVisualizer
from pathologic.search import artifacts as _search_artifacts


def _safe_metric(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("-inf")
    if math.isnan(parsed):
        return float("-inf")
    return parsed


def compute_calibration_rankings(
    *,
    leaderboard: list[dict[str, Any]],
    objective: str,
    objective_weight: float,
    ece_weight: float,
    brier_weight: float,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    calibration_summary_rows: list[dict[str, Any]] = []
    candidate_method_ranking: dict[str, list[dict[str, Any]]] = {}
    candidate_calibration_ranking: list[dict[str, Any]] = []

    for item in leaderboard:
        calibration_info = item.get("calibration")
        if not isinstance(calibration_info, dict):
            continue
        summary = calibration_info.get("summary")
        if not isinstance(summary, dict):
            continue
        candidate_name = str(item.get("candidate", "unknown"))
        method_rows_for_candidate: list[dict[str, Any]] = []
        for method_name, method_summary in summary.items():
            if not isinstance(method_summary, dict):
                continue
            if method_summary.get("status") != "ok":
                continue
            row = {
                "candidate": candidate_name,
                "method": method_name,
                "brier_score": float(method_summary["brier_score"]),
                "ece": float(method_summary["ece"]),
            }
            calibration_summary_rows.append(row)
            method_rows_for_candidate.append(row)

        if method_rows_for_candidate:
            sorted_methods = sorted(
                method_rows_for_candidate,
                key=lambda row: (row["ece"], row["brier_score"]),
            )
            candidate_method_ranking[candidate_name] = sorted_methods
            best_method = sorted_methods[0]
            objective_score = _safe_metric(item.get("test_metrics", {}).get(objective))
            calibration_penalty = (
                (ece_weight * float(best_method["ece"]))
                + (brier_weight * float(best_method["brier_score"]))
            )
            calibration_aware_score = (
                (objective_weight * float(objective_score)) - calibration_penalty
            )
            candidate_calibration_ranking.append(
                {
                    "candidate": candidate_name,
                    "objective_score": float(objective_score),
                    "best_calibration_method": best_method["method"],
                    "best_calibration_ece": float(best_method["ece"]),
                    "best_calibration_brier": float(best_method["brier_score"]),
                    "calibration_penalty": float(calibration_penalty),
                    "calibration_aware_score": float(calibration_aware_score),
                }
            )

    candidate_calibration_ranking = sorted(
        candidate_calibration_ranking,
        key=lambda row: (
            row["calibration_aware_score"],
            row["objective_score"],
            -row["best_calibration_ece"],
            -row["best_calibration_brier"],
        ),
        reverse=True,
    )
    return calibration_summary_rows, candidate_method_ranking, candidate_calibration_ranking


def select_calibration_aware_winner(
    *,
    ranked: list[dict[str, Any]],
    candidate_calibration_ranking: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    objective_best = ranked[0]
    ranked_by_name = {str(row.get("candidate")): row for row in ranked}
    calibration_aware_best = objective_best
    if candidate_calibration_ranking:
        top_candidate_name = str(candidate_calibration_ranking[0]["candidate"])
        calibration_aware_best = ranked_by_name.get(top_candidate_name, objective_best)
    return objective_best, calibration_aware_best


def write_run_reports(
    *,
    run_dir: Path,
    objective: str,
    budget_profile: str,
    seed: int,
    candidates_total: int,
    candidates_ok: int,
    leaderboard: list[dict[str, Any]],
    best: dict[str, Any],
    objective_best: dict[str, Any],
    elapsed_seconds: float,
    prep_stats: dict[str, Any],
    feature_count: int,
    split_summary: dict[str, Any],
    outer_train_rows: int,
    outer_calibration_rows: int,
    outer_test_rows: int,
    objective_weight: float,
    ece_weight: float,
    brier_weight: float,
    calibration_summary_rows: list[dict[str, Any]],
    candidate_method_ranking: dict[str, list[dict[str, Any]]],
    candidate_calibration_ranking: list[dict[str, Any]],
) -> tuple[Path, Path]:
    split_manifest = {
        "outer_split_summary": split_summary,
        "outer_train_rows": int(outer_train_rows),
        "outer_calibration_rows": int(outer_calibration_rows),
        "outer_test_rows": int(outer_test_rows),
    }
    leaderboard_payload = {
        "objective": objective,
        "budget_profile": budget_profile,
        "seed": int(seed),
        "candidates_total": int(candidates_total),
        "candidates_ok": int(candidates_ok),
        "rows": leaderboard,
    }
    best_summary = {
        "objective": objective,
        "winner": best,
        "winner_hybrid_config": best.get("hybrid_config"),
        "winner_selection_mode": "calibration_aware",
        "objective_only_winner": objective_best,
        "elapsed_seconds": float(elapsed_seconds),
        "prepared_stats": prep_stats,
        "feature_count": int(feature_count),
    }

    (run_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (run_dir / "best_model_summary.json").write_text(
        json.dumps(best_summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (run_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    calibration_summary_path = run_dir / "calibration_summary.json"
    calibration_summary = {
        "objective": objective,
        "winner_decision": {
            "selection_mode": "calibration_aware",
            "objective_only_winner": str(objective_best.get("candidate")),
            "calibration_aware_winner": str(best.get("candidate")),
            "changed_by_calibration": bool(
                str(objective_best.get("candidate")) != str(best.get("candidate"))
            ),
            "weights": {
                "objective": float(objective_weight),
                "ece": float(ece_weight),
                "brier": float(brier_weight),
            },
            "penalty_formula": (
                "calibration_aware_score = "
                "objective_weight*objective_score - ece_weight*ece - brier_weight*brier_score"
            ),
        },
        "candidate_method_ranking": candidate_method_ranking,
        "candidate_calibration_ranking": candidate_calibration_ranking,
        "rows": sorted(
            calibration_summary_rows,
            key=lambda row: (row["ece"], row["brier_score"]),
        ),
    }
    calibration_summary_path.write_text(
        json.dumps(calibration_summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    calibration_summary_html_path = run_dir / "calibration_summary.html"
    try:
        ExplainabilityVisualizer().render_calibration_summary_html(
            calibration_summary,
            str(calibration_summary_html_path),
        )
    except Exception:
        # HTML rendering is best-effort and must not block search completion.
        pass

    error_analysis_summary_path = run_dir / "error_analysis_summary.json"
    error_analysis_summary = _search_artifacts.build_error_analysis_run_summary(
        leaderboard_rows=leaderboard,
        winner_candidate=str(best.get("candidate", "")),
    )
    error_analysis_summary_path.write_text(
        json.dumps(error_analysis_summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return calibration_summary_path, error_analysis_summary_path
