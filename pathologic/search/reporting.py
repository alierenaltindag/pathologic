"""Leaderboard ranking and report serialization helpers for search workflows."""

from __future__ import annotations

from html import escape
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


def _build_train_report_payload(
    *,
    objective: str,
    leaderboard: list[dict[str, Any]],
    best: dict[str, Any],
    objective_best: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    for row in leaderboard:
        if not isinstance(row, dict):
            continue
        candidate_name = str(row.get("candidate", "unknown"))
        metrics = row.get("test_metrics") if isinstance(row.get("test_metrics"), dict) else {}

        calibration_payload = row.get("calibration") if isinstance(row.get("calibration"), dict) else {}
        calibration_summary = (
            calibration_payload.get("summary")
            if isinstance(calibration_payload.get("summary"), dict)
            else {}
        )
        raw_calibration = (
            calibration_summary.get("raw")
            if isinstance(calibration_summary.get("raw"), dict)
            else {}
        )
        raw_ece = raw_calibration.get("ece") if raw_calibration.get("status") == "ok" else None
        raw_brier = (
            raw_calibration.get("brier_score")
            if raw_calibration.get("status") == "ok"
            else None
        )

        best_calibration_method: str | None = None
        best_calibration_ece: float | None = None
        best_calibration_brier: float | None = None
        for method_name, method_summary in calibration_summary.items():
            if not isinstance(method_summary, dict) or method_summary.get("status") != "ok":
                continue
            method_ece = float(method_summary.get("ece", float("inf")))
            method_brier = float(method_summary.get("brier_score", float("inf")))
            if best_calibration_method is None:
                best_calibration_method = str(method_name)
                best_calibration_ece = method_ece
                best_calibration_brier = method_brier
                continue
            assert best_calibration_ece is not None
            assert best_calibration_brier is not None
            if (method_ece, method_brier) < (best_calibration_ece, best_calibration_brier):
                best_calibration_method = str(method_name)
                best_calibration_ece = method_ece
                best_calibration_brier = method_brier

        panel_thresholds = (
            row.get("panel_thresholds") if isinstance(row.get("panel_thresholds"), dict) else {}
        )
        panel_summary = (
            panel_thresholds.get("summary")
            if isinstance(panel_thresholds.get("summary"), dict)
            else {}
        )

        candidate_rows.append(
            {
                "candidate": candidate_name,
                "kind": str(row.get("kind", "unknown")),
                "status": str(row.get("status", "unknown")),
                "runtime_seconds": float(row.get("runtime_seconds", 0.0)),
                "objective_score": _safe_metric(metrics.get(objective)),
                "metrics": {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "auprc": metrics.get("auprc"),
                "mcc": metrics.get("mcc"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "raw_ece": raw_ece,
                "raw_brier": raw_brier,
                "best_calibration_method": best_calibration_method,
                "best_calibration_ece": best_calibration_ece,
                "best_calibration_brier": best_calibration_brier,
                "panel_threshold_status": str(panel_thresholds.get("status", "missing")),
                "panel_count": int(panel_summary.get("panel_count", 0)) if panel_summary else 0,
                "optimized_panel_count": int(panel_summary.get("optimized_panel_count", 0))
                if panel_summary
                else 0,
                "selected_params_source": str(row.get("selected_params_source", "unknown")),
            }
        )

    candidate_rows.sort(
        key=lambda item: (float(item.get("objective_score", float("-inf"))), -float(item.get("runtime_seconds", 0.0))),
        reverse=True,
    )

    winner_candidate = str(best.get("candidate", ""))
    winner_row = next(
        (
            row
            for row in leaderboard
            if isinstance(row, dict) and str(row.get("candidate", "")) == winner_candidate
        ),
        {},
    )

    return {
        "objective": objective,
        "elapsed_seconds": float(elapsed_seconds),
        "winner": {
            "candidate": winner_candidate,
            "selection_mode": "calibration_aware",
            "objective_only_winner": str(objective_best.get("candidate", "")),
            "kind": str(best.get("kind", winner_row.get("kind", "unknown"))),
            "selected_params_source": str(
                winner_row.get("selected_params_source", best.get("selected_params_source", "unknown"))
            ),
            "selected_params": winner_row.get("selected_params", best.get("selected_params", {})),
            "hybrid_config": winner_row.get("hybrid_config", best.get("hybrid_config", {})),
            "test_metrics": winner_row.get("test_metrics", best.get("test_metrics", {})),
            "hpo": winner_row.get("hpo"),
            "hpo_level1": winner_row.get("hpo_level1"),
            "hpo_level2": winner_row.get("hpo_level2"),
            "nas": winner_row.get("nas"),
            "calibration": winner_row.get("calibration"),
            "panel_thresholds": winner_row.get("panel_thresholds"),
        },
        "candidates": candidate_rows,
    }


def _render_train_report_html(*, report: dict[str, Any], output_path: Path) -> None:
    winner = report.get("winner") if isinstance(report.get("winner"), dict) else {}
    candidates = report.get("candidates") if isinstance(report.get("candidates"), list) else []
    objective = str(report.get("objective", "f1"))

    table_rows: list[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue

        def _fmt(value: Any) -> str:
            if value is None:
                return "-"
            if isinstance(value, (int, float)):
                return f"{float(value):.6f}"
            return escape(str(value))

        table_rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('candidate', 'unknown')))}</td>"
            f"<td>{escape(str(item.get('status', 'unknown')))}</td>"
            f"<td>{_fmt(item.get('f1'))}</td>"
            f"<td>{_fmt(item.get('roc_auc'))}</td>"
            f"<td>{_fmt(item.get('auprc'))}</td>"
            f"<td>{_fmt(item.get('mcc'))}</td>"
            f"<td>{_fmt(item.get('precision'))}</td>"
            f"<td>{_fmt(item.get('recall'))}</td>"
            f"<td>{_fmt(item.get('raw_ece'))}</td>"
            f"<td>{_fmt(item.get('raw_brier'))}</td>"
            f"<td>{_fmt(item.get('objective_score'))}</td>"
            f"<td>{_fmt(item.get('runtime_seconds'))}</td>"
            "</tr>"
        )

    if not table_rows:
        table_rows.append("<tr><td colspan='12'>No candidate rows available.</td></tr>")

    winner_json = escape(json.dumps(winner, ensure_ascii=True, indent=2))
    html = (
        "<html><head><meta charset='utf-8'><title>Train Report</title>"
        "<style>"
        "body{font-family:Segoe UI,Tahoma,sans-serif;line-height:1.5;color:#1f2937;max-width:1300px;margin:0 auto;padding:20px;background:#f3f6fb;}"
        "h1,h2{color:#0f2942;}"
        "table{width:100%;border-collapse:collapse;margin-top:8px;}"
        "th,td{border:1px solid #dbe4ee;padding:8px;text-align:left;background:#fff;font-size:12px;}"
        "th{background:#eaf1f8;font-weight:700;}"
        ".card{background:#fff;border:1px solid #dbe4ee;border-radius:8px;padding:14px;margin-bottom:14px;}"
        "pre{background:#0b1220;color:#dbeafe;padding:12px;border-radius:6px;overflow:auto;font-size:12px;}"
        "</style></head><body>"
        "<h1>PathoLogic Train Report</h1>"
        "<div class='card'>"
        f"<div><strong>Objective:</strong> {escape(objective)}</div>"
        f"<div><strong>Winner:</strong> {escape(str(winner.get('candidate', 'unknown')))}</div>"
        f"<div><strong>Selection mode:</strong> {escape(str(winner.get('selection_mode', 'calibration_aware')))}</div>"
        "</div>"
        "<div class='card'><h2>Candidate Quick Summary</h2>"
        "<table><thead><tr><th>Candidate</th><th>Status</th><th>F1</th><th>ROC-AUC</th><th>AUPRC</th><th>MCC</th><th>Precision</th><th>Recall</th><th>ECE(raw)</th><th>Brier(raw)</th><th>Objective</th><th>Runtime(s)</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></div>"
        "<div class='card'><h2>Winner Detailed Configuration</h2>"
        "<pre>"
        + winner_json
        + "</pre></div>"
        "</body></html>"
    )
    output_path.write_text(html, encoding="utf-8")


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
) -> tuple[Path, Path, Path]:
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

    train_report_path = run_dir / "train_report.json"
    train_report = _build_train_report_payload(
        objective=objective,
        leaderboard=leaderboard,
        best=best,
        objective_best=objective_best,
        elapsed_seconds=elapsed_seconds,
    )
    train_report_path.write_text(
        json.dumps(train_report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    train_report_html_path = run_dir / "train_report.html"
    _render_train_report_html(report=train_report, output_path=train_report_html_path)
    return calibration_summary_path, error_analysis_summary_path, train_report_path
