"""Leaderboard ranking and report serialization helpers for search workflows."""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any

from pathologic.explain.visualizer import ExplainabilityVisualizer
from pathologic.search import artifacts as _search_artifacts


_LEAKAGE_FIELDS: tuple[str, str, str] = (
    "train_val_shared_genes",
    "train_test_shared_genes",
    "val_test_shared_genes",
)


def _safe_metric(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("-inf")
    if math.isnan(parsed):
        return float("-inf")
    return parsed


def _extract_cv_objective_score(row: dict[str, Any]) -> float:
    hpo_payload = row.get("hpo") if isinstance(row.get("hpo"), dict) else {}
    if isinstance(hpo_payload, dict):
        score = _safe_metric(hpo_payload.get("best_score"))
        if score != float("-inf"):
            return float(score)

    for key in ("hpo_level2", "hpo_level1"):
        payload = row.get(key)
        if not isinstance(payload, dict):
            continue
        score = _safe_metric(payload.get("best_score"))
        if score != float("-inf"):
            return float(score)

    return float("-inf")


def _overfitting_summary(
    *,
    cv_score: float,
    test_score: float,
) -> dict[str, Any]:
    if cv_score == float("-inf") or test_score == float("-inf"):
        return {
            "cv_objective_score": None,
            "test_objective_score": None,
            "generalization_gap": None,
            "generalization_gap_ratio": None,
            "overfitting_risk_level": "unknown",
            "overfitting_suspected": False,
        }

    gap = float(cv_score - test_score)
    denom = max(abs(float(cv_score)), 1e-8)
    gap_ratio = float(gap / denom)

    if gap >= 0.08 or gap_ratio >= 0.12:
        risk_level = "high"
    elif gap >= 0.03 or gap_ratio >= 0.05:
        risk_level = "moderate"
    elif gap <= -0.05:
        risk_level = "underfit_or_shift"
    else:
        risk_level = "low"

    return {
        "cv_objective_score": float(cv_score),
        "test_objective_score": float(test_score),
        "generalization_gap": gap,
        "generalization_gap_ratio": gap_ratio,
        "overfitting_risk_level": risk_level,
        "overfitting_suspected": risk_level in {"moderate", "high"},
    }


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


def _build_split_manifest_warnings(
    *,
    split_summary: dict[str, Any],
    outer_train_rows: int,
    outer_calibration_rows: int,
    outer_test_rows: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    for field in _LEAKAGE_FIELDS:
        raw_value = split_summary.get(field)
        if raw_value is None:
            items.append(
                {
                    "code": "SPLIT_LEAKAGE_FIELD_MISSING",
                    "severity": "warning",
                    "field": field,
                    "actual": None,
                    "expected": 0,
                    "message": f"Missing leakage field '{field}' in outer_split_summary.",
                    "source": "split_manifest.outer_split_summary",
                }
            )
            continue

        try:
            parsed_value = int(raw_value)
        except (TypeError, ValueError):
            items.append(
                {
                    "code": "SPLIT_LEAKAGE_FIELD_INVALID_TYPE",
                    "severity": "warning",
                    "field": field,
                    "actual": raw_value,
                    "expected": "integer >= 0",
                    "message": f"Leakage field '{field}' is not an integer value.",
                    "source": "split_manifest.outer_split_summary",
                }
            )
            continue

        if parsed_value < 0:
            items.append(
                {
                    "code": "SPLIT_LEAKAGE_FIELD_NEGATIVE",
                    "severity": "warning",
                    "field": field,
                    "actual": parsed_value,
                    "expected": "integer >= 0",
                    "message": f"Leakage field '{field}' is negative.",
                    "source": "split_manifest.outer_split_summary",
                }
            )
            continue

        if parsed_value != 0:
            items.append(
                {
                    "code": "SPLIT_LEAKAGE_NONZERO",
                    "severity": "error",
                    "field": field,
                    "actual": parsed_value,
                    "expected": 0,
                    "message": f"Leakage detected: '{field}' must be zero.",
                    "source": "split_manifest.outer_split_summary",
                }
            )

    expected_total_rows = int(outer_train_rows) + int(outer_calibration_rows) + int(outer_test_rows)
    split_row_fields = ("train_size", "val_size", "test_size")
    if all(field in split_summary for field in split_row_fields):
        try:
            summary_total_rows = sum(int(split_summary[field]) for field in split_row_fields)
        except (TypeError, ValueError):
            items.append(
                {
                    "code": "SPLIT_ROWCOUNT_INVALID_TYPE",
                    "severity": "warning",
                    "field": "train_size/val_size/test_size",
                    "actual": [split_summary.get(field) for field in split_row_fields],
                    "expected": "integer triplet",
                    "message": "Split row counts are not all valid integers.",
                    "source": "split_manifest.outer_split_summary",
                }
            )
        else:
            if summary_total_rows != expected_total_rows:
                items.append(
                    {
                        "code": "SPLIT_ROWCOUNT_MISMATCH",
                        "severity": "warning",
                        "field": "train_size+val_size+test_size",
                        "actual": summary_total_rows,
                        "expected": expected_total_rows,
                        "message": "Split row counts do not match outer split partition totals.",
                        "source": "split_manifest",
                    }
                )

    has_errors = any(str(item.get("severity")) == "error" for item in items)
    status = "error" if has_errors else ("warning" if items else "ok")
    return {
        "status": status,
        "warning_count": len(items),
        "has_errors": has_errors,
        "checked_fields": list(_LEAKAGE_FIELDS),
        "items": items,
    }


def _build_train_report_payload(
    *,
    objective: str,
    leaderboard: list[dict[str, Any]],
    best: dict[str, Any],
    objective_best: dict[str, Any],
    elapsed_seconds: float,
    split_manifest_warnings: dict[str, Any] | None = None,
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

        compute_cost_payload = row.get("compute_cost") if isinstance(row.get("compute_cost"), dict) else {}
        compute_cost_training = (
            compute_cost_payload.get("training")
            if isinstance(compute_cost_payload.get("training"), dict)
            else {}
        )
        compute_cost_training_memory = (
            compute_cost_training.get("memory")
            if isinstance(compute_cost_training.get("memory"), dict)
            else {}
        )
        compute_cost_inference = (
            compute_cost_payload.get("inference")
            if isinstance(compute_cost_payload.get("inference"), dict)
            else {}
        )
        compute_cost_gpu = (
            compute_cost_payload.get("gpu_after_inference")
            if isinstance(compute_cost_payload.get("gpu_after_inference"), dict)
            else {}
        )
        compute_cost_artifacts = (
            compute_cost_payload.get("artifacts")
            if isinstance(compute_cost_payload.get("artifacts"), dict)
            else {}
        )

        cv_objective_score = _extract_cv_objective_score(row)
        test_objective_score = _safe_metric(metrics.get(objective))
        overfitting = _overfitting_summary(
            cv_score=cv_objective_score,
            test_score=test_objective_score,
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
                "compute_cost_status": str(compute_cost_payload.get("status", "missing")),
                "train_total_seconds": compute_cost_training.get("train_total_seconds"),
                "iteration_seconds": compute_cost_training.get("iteration_seconds"),
                "batch_size": compute_cost_training.get("batch_size"),
                "single_sample_ms": compute_cost_inference.get("single_sample_ms"),
                "batch_total_ms": compute_cost_inference.get("batch_total_ms"),
                "batch_per_sample_ms": compute_cost_inference.get("batch_per_sample_ms"),
                "peak_vram_mb": compute_cost_gpu.get("vram_peak_allocated_mb"),
                "train_memory_delta_mb": compute_cost_training_memory.get("process_rss_delta_mb"),
                "train_memory_peak_delta_mb": compute_cost_training_memory.get(
                    "process_rss_peak_delta_mb"
                ),
                "compute_cost_report_html": compute_cost_artifacts.get("compute_cost_report_html"),
                **overfitting,
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
        "split_manifest_warnings": split_manifest_warnings
        if isinstance(split_manifest_warnings, dict)
        else {
            "status": "ok",
            "warning_count": 0,
            "has_errors": False,
            "checked_fields": list(_LEAKAGE_FIELDS),
            "items": [],
        },
        "overfitting_policy": {
            "gap_thresholds": {
                "moderate": 0.03,
                "high": 0.08,
            },
            "gap_ratio_thresholds": {
                "moderate": 0.05,
                "high": 0.12,
            },
            "note": (
                "Heuristic generalization risk derived from CV objective vs test objective. "
                "Use with calibration and error-analysis artifacts for final diagnosis."
            ),
        },
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
            "cv_objective_score": next(
                (
                    row.get("cv_objective_score")
                    for row in candidate_rows
                    if str(row.get("candidate", "")) == winner_candidate
                ),
                None,
            ),
            "test_objective_score": next(
                (
                    row.get("test_objective_score")
                    for row in candidate_rows
                    if str(row.get("candidate", "")) == winner_candidate
                ),
                None,
            ),
            "generalization_gap": next(
                (
                    row.get("generalization_gap")
                    for row in candidate_rows
                    if str(row.get("candidate", "")) == winner_candidate
                ),
                None,
            ),
            "generalization_gap_ratio": next(
                (
                    row.get("generalization_gap_ratio")
                    for row in candidate_rows
                    if str(row.get("candidate", "")) == winner_candidate
                ),
                None,
            ),
            "overfitting_risk_level": next(
                (
                    row.get("overfitting_risk_level")
                    for row in candidate_rows
                    if str(row.get("candidate", "")) == winner_candidate
                ),
                "unknown",
            ),
            "overfitting_suspected": bool(
                next(
                    (
                        row.get("overfitting_suspected")
                        for row in candidate_rows
                        if str(row.get("candidate", "")) == winner_candidate
                    ),
                    False,
                )
            ),
        },
        "candidates": candidate_rows,
    }


def _render_train_report_html(*, report: dict[str, Any], output_path: Path) -> None:
    winner = report.get("winner") if isinstance(report.get("winner"), dict) else {}
    candidates = report.get("candidates") if isinstance(report.get("candidates"), list) else []
    objective = str(report.get("objective", "f1"))
    split_manifest_warnings = (
        report.get("split_manifest_warnings")
        if isinstance(report.get("split_manifest_warnings"), dict)
        else {}
    )
    warning_items = (
        split_manifest_warnings.get("items")
        if isinstance(split_manifest_warnings.get("items"), list)
        else []
    )

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
            f"<td>{_fmt(item.get('cv_objective_score'))}</td>"
            f"<td>{_fmt(item.get('test_objective_score'))}</td>"
            f"<td>{_fmt(item.get('generalization_gap'))}</td>"
            f"<td>{_fmt(item.get('generalization_gap_ratio'))}</td>"
            f"<td>{_fmt(item.get('overfitting_risk_level'))}</td>"
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
            f"<td>{_fmt(item.get('train_total_seconds'))}</td>"
            f"<td>{_fmt(item.get('single_sample_ms'))}</td>"
            f"<td>{_fmt(item.get('batch_total_ms'))}</td>"
            f"<td>{_fmt(item.get('peak_vram_mb'))}</td>"
            f"<td>{_fmt(item.get('train_memory_delta_mb'))}</td>"
            f"<td>{_fmt(item.get('train_memory_peak_delta_mb'))}</td>"
            f"<td>{_fmt(item.get('compute_cost_report_html'))}</td>"
            "</tr>"
        )

    if not table_rows:
        table_rows.append("<tr><td colspan='24'>No candidate rows available.</td></tr>")

    warning_rows: list[str] = []
    for item in warning_items:
        if not isinstance(item, dict):
            continue
        warning_rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('severity', 'warning')))}</td>"
            f"<td>{escape(str(item.get('code', 'unknown')))}</td>"
            f"<td>{escape(str(item.get('field', '-')))}</td>"
            f"<td>{escape(str(item.get('actual', '-')))}</td>"
            f"<td>{escape(str(item.get('expected', '-')))}</td>"
            f"<td>{escape(str(item.get('message', '-')))}</td>"
            "</tr>"
        )
    if not warning_rows:
        warning_rows.append("<tr><td colspan='6'>No split manifest warnings.</td></tr>")

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
        f"<div><strong>Split manifest warning status:</strong> {escape(str(split_manifest_warnings.get('status', 'ok')))}</div>"
        f"<div><strong>Split manifest warning count:</strong> {escape(str(split_manifest_warnings.get('warning_count', 0)))}</div>"
        "</div>"
        "<div class='card'><h2>Split Manifest Warnings</h2>"
        "<table><thead><tr><th>Severity</th><th>Code</th><th>Field</th><th>Actual</th><th>Expected</th><th>Message</th></tr></thead><tbody>"
        + "".join(warning_rows)
        + "</tbody></table></div>"
        "<div class='card'><h2>Candidate Quick Summary</h2>"
        "<table><thead><tr><th>Candidate</th><th>Status</th><th>CV Objective</th><th>Test Objective</th><th>Gap(CV-Test)</th><th>Gap Ratio</th><th>Overfit Risk</th><th>F1</th><th>ROC-AUC</th><th>AUPRC</th><th>MCC</th><th>Precision</th><th>Recall</th><th>ECE(raw)</th><th>Brier(raw)</th><th>Objective</th><th>Runtime(s)</th><th>Train(s)</th><th>SingleInf(ms)</th><th>BatchInf(ms)</th><th>PeakVRAM(MB)</th><th>TrainMemDelta(MB)</th><th>TrainMemPeakDelta(MB)</th><th>ComputeCostReport</th></tr></thead><tbody>"
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
    split_manifest_warnings = _build_split_manifest_warnings(
        split_summary=split_summary,
        outer_train_rows=int(outer_train_rows),
        outer_calibration_rows=int(outer_calibration_rows),
        outer_test_rows=int(outer_test_rows),
    )
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
        split_manifest_warnings=split_manifest_warnings,
    )
    train_report_path.write_text(
        json.dumps(train_report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    train_report_html_path = run_dir / "train_report.html"
    _render_train_report_html(report=train_report, output_path=train_report_html_path)
    return calibration_summary_path, error_analysis_summary_path, train_report_path
