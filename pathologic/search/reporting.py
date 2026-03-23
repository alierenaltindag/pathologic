"""Leaderboard ranking and report serialization helpers for search workflows."""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

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


def _extract_fold_distribution(row: dict[str, Any]) -> dict[str, Any]:
    hpo_payload = row.get("hpo") if isinstance(row.get("hpo"), dict) else {}
    trials = hpo_payload.get("trials") if isinstance(hpo_payload.get("trials"), list) else []
    collected: list[float] = []
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        fold_scores = trial.get("fold_scores") if isinstance(trial.get("fold_scores"), list) else []
        for value in fold_scores:
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                collected.append(float(value))

    if not collected:
        return {
            "status": "missing",
            "count": 0,
        }

    values = np.asarray(collected, dtype=float)
    return {
        "status": "ok",
        "count": int(values.shape[0]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
    }


def _extract_learning_curve(row: dict[str, Any]) -> dict[str, Any]:
    hpo_payload = row.get("hpo") if isinstance(row.get("hpo"), dict) else {}
    trials = hpo_payload.get("trials") if isinstance(hpo_payload.get("trials"), list) else []
    points: list[dict[str, Any]] = []
    best_so_far = float("-inf")
    for idx, trial in enumerate(trials, start=1):
        if not isinstance(trial, dict):
            continue
        score = trial.get("score")
        if not isinstance(score, (int, float)):
            continue
        score_float = float(score)
        best_so_far = max(best_so_far, score_float)
        points.append(
            {
                "trial_index": int(idx),
                "score": score_float,
                "best_so_far": float(best_so_far),
            }
        )

    if not points:
        return {
            "status": "missing",
            "points": [],
        }

    return {
        "status": "ok",
        "points": points,
        "total_points": int(len(points)),
        "improvement": float(points[-1]["best_so_far"] - points[0]["best_so_far"]),
    }


def _extract_calibration_joint_summary(calibration_summary: dict[str, Any]) -> dict[str, Any]:
    raw_payload = calibration_summary.get("raw") if isinstance(calibration_summary.get("raw"), dict) else {}
    raw_ece = raw_payload.get("ece") if raw_payload.get("status") == "ok" else None
    raw_brier = raw_payload.get("brier_score") if raw_payload.get("status") == "ok" else None

    best_method: str | None = None
    best_ece: float | None = None
    best_brier: float | None = None
    for method_name, method_summary in calibration_summary.items():
        if not isinstance(method_summary, dict):
            continue
        if method_summary.get("status") != "ok":
            continue
        method_ece = method_summary.get("ece")
        method_brier = method_summary.get("brier_score")
        if not isinstance(method_ece, (int, float)) or not isinstance(method_brier, (int, float)):
            continue
        method_ece_f = float(method_ece)
        method_brier_f = float(method_brier)
        if best_method is None or (method_ece_f, method_brier_f) < (best_ece, best_brier):
            best_method = str(method_name)
            best_ece = method_ece_f
            best_brier = method_brier_f

    return {
        "raw_ece": float(raw_ece) if isinstance(raw_ece, (int, float)) else None,
        "raw_brier": float(raw_brier) if isinstance(raw_brier, (int, float)) else None,
        "best_method": best_method,
        "best_ece": best_ece,
        "best_brier": best_brier,
        "ece_improvement": (
            float(raw_ece) - float(best_ece)
            if isinstance(raw_ece, (int, float)) and isinstance(best_ece, (int, float))
            else None
        ),
        "brier_improvement": (
            float(raw_brier) - float(best_brier)
            if isinstance(raw_brier, (int, float)) and isinstance(best_brier, (int, float))
            else None
        ),
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
                    "severity": "warning",
                    "field": field,
                    "actual": parsed_value,
                    "expected": 0,
                    "message": (
                        f"Same-gene overlap observed in '{field}'. "
                        "This is allowed by current split policy."
                    ),
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
    status = "warning" if items else "ok"
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
        calibration_joint = _extract_calibration_joint_summary(calibration_summary)
        raw_ece = calibration_joint.get("raw_ece")
        raw_brier = calibration_joint.get("raw_brier")
        best_calibration_method = calibration_joint.get("best_method")
        best_calibration_ece = calibration_joint.get("best_ece")
        best_calibration_brier = calibration_joint.get("best_brier")

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
        fold_distribution = _extract_fold_distribution(row)
        learning_curve = _extract_learning_curve(row)
        holdout_bootstrap = (
            row.get("holdout_bootstrap") if isinstance(row.get("holdout_bootstrap"), dict) else {}
        )
        group_drift = row.get("group_drift") if isinstance(row.get("group_drift"), dict) else {}

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
                "calibration_ece_improvement": calibration_joint.get("ece_improvement"),
                "calibration_brier_improvement": calibration_joint.get("brier_improvement"),
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
                "fold_distribution": fold_distribution,
                "holdout_bootstrap": holdout_bootstrap,
                "group_drift": group_drift,
                "learning_curve": learning_curve,
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
        "reliability_sections": {
            "fold_distribution": {
                "description": "Cross-validation fold score distribution aggregated from HPO trial fold scores.",
                "available_candidates": int(
                    sum(
                        1
                        for row in candidate_rows
                        if isinstance(row.get("fold_distribution"), dict)
                        and str(row["fold_distribution"].get("status")) == "ok"
                    )
                ),
            },
            "holdout_bootstrap": {
                "description": "No-retrain bootstrap confidence intervals computed on holdout predictions.",
                "available_candidates": int(
                    sum(
                        1
                        for row in candidate_rows
                        if isinstance(row.get("holdout_bootstrap"), dict)
                        and str(row["holdout_bootstrap"].get("status")) == "ok"
                    )
                ),
            },
            "group_drift": {
                "description": "Group-level performance spread over selected metadata column(s).",
                "available_candidates": int(
                    sum(
                        1
                        for row in candidate_rows
                        if isinstance(row.get("group_drift"), dict)
                        and str(row["group_drift"].get("status")) == "ok"
                    )
                ),
            },
            "learning_curve": {
                "description": "Trial-wise optimization curve (score and cumulative best) from HPO search.",
                "available_candidates": int(
                    sum(
                        1
                        for row in candidate_rows
                        if isinstance(row.get("learning_curve"), dict)
                        and str(row["learning_curve"].get("status")) == "ok"
                    )
                ),
            },
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
            f"<td>{_fmt(item.get('calibration_ece_improvement'))}</td>"
            f"<td>{_fmt(item.get('calibration_brier_improvement'))}</td>"
            f"<td>{_fmt(item.get('objective_score'))}</td>"
            f"<td>{_fmt(item.get('runtime_seconds'))}</td>"
            f"<td>{_fmt(item.get('train_total_seconds'))}</td>"
            f"<td>{_fmt(item.get('single_sample_ms'))}</td>"
            f"<td>{_fmt(item.get('batch_total_ms'))}</td>"
            f"<td>{_fmt(item.get('peak_vram_mb'))}</td>"
            f"<td>{_fmt(item.get('train_memory_delta_mb'))}</td>"
            f"<td>{_fmt(item.get('train_memory_peak_delta_mb'))}</td>"
            f"<td>{_fmt((item.get('fold_distribution') or {}).get('std'))}</td>"
            f"<td>{_fmt((item.get('group_drift') or {}).get('group_column'))}</td>"
            f"<td>{_fmt((item.get('group_drift') or {}).get('group_count'))}</td>"
            f"<td>{_fmt((item.get('learning_curve') or {}).get('total_points'))}</td>"
            f"<td>{_fmt(item.get('compute_cost_report_html'))}</td>"
            "</tr>"
        )

    if not table_rows:
        table_rows.append("<tr><td colspan='30'>No candidate rows available.</td></tr>")

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

    def _render_learning_curve_svg(points: list[dict[str, Any]]) -> str:
        if not points:
            return "<div>No learning-curve points available.</div>"
        values: list[float] = []
        for item in points:
            score = item.get("score") if isinstance(item, dict) else None
            if isinstance(score, (int, float)) and np.isfinite(float(score)):
                values.append(float(score))
        if len(values) < 2:
            return "<div>Not enough learning-curve points for visualization.</div>"

        width = 480.0
        height = 160.0
        padding = 16.0
        min_v = min(values)
        max_v = max(values)
        span = max(max_v - min_v, 1e-9)
        x_step = (width - (2.0 * padding)) / float(max(len(values) - 1, 1))

        polyline_points: list[str] = []
        for idx, value in enumerate(values):
            x = padding + (x_step * float(idx))
            y = padding + ((max_v - value) / span) * (height - (2.0 * padding))
            polyline_points.append(f"{x:.2f},{y:.2f}")

        return (
            "<svg width='100%' viewBox='0 0 480 160' role='img' aria-label='Learning curve sparkline'>"
            "<rect x='0' y='0' width='480' height='160' fill='#f8fafc' stroke='#dbe4ee'/>"
            "<polyline fill='none' stroke='#0f766e' stroke-width='2.5' points='"
            + " ".join(polyline_points)
            + "'/>"
            "</svg>"
        )

    def _calibration_delta_badge(*, label: str, value: Any) -> str:
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            return (
                "<span style='display:inline-block;padding:4px 8px;border-radius:999px;"
                "font-size:12px;background:#e5e7eb;color:#374151;border:1px solid #d1d5db;'>"
                + escape(f"{label}: n/a")
                + "</span>"
            )

        value_float = float(value)
        if value_float > 0:
            bg = "#dcfce7"
            fg = "#166534"
            border = "#86efac"
            symbol = "+"
        elif value_float < 0:
            bg = "#fee2e2"
            fg = "#991b1b"
            border = "#fecaca"
            symbol = ""
        else:
            bg = "#f3f4f6"
            fg = "#374151"
            border = "#d1d5db"
            symbol = ""

        return (
            "<span style='display:inline-block;padding:4px 8px;border-radius:999px;"
            "font-size:12px;font-weight:600;background:"
            + bg
            + ";color:"
            + fg
            + ";border:1px solid "
            + border
            + ";'>"
            + escape(f"{label}: {symbol}{value_float:.6f}")
            + "</span>"
        )

    winner_candidate_name = str(winner.get("candidate", ""))
    winner_candidate_row = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and str(item.get("candidate", "")) == winner_candidate_name
        ),
        {},
    )

    winner_bootstrap = (
        winner_candidate_row.get("holdout_bootstrap")
        if isinstance(winner_candidate_row.get("holdout_bootstrap"), dict)
        else {}
    )
    winner_bootstrap_metrics = (
        winner_bootstrap.get("metrics") if isinstance(winner_bootstrap.get("metrics"), dict) else {}
    )
    bootstrap_rows: list[str] = []
    for metric_name in ("f1", "roc_auc", "auprc", "mcc", "precision", "recall"):
        item = winner_bootstrap_metrics.get(metric_name)
        if not isinstance(item, dict):
            continue
        point_estimate = item.get("point_estimate")
        ci_low = item.get("ci_low")
        ci_high = item.get("ci_high")
        bootstrap_rows.append(
            "<tr>"
            f"<td>{escape(metric_name)}</td>"
            f"<td>{escape('-' if point_estimate is None else f'{float(point_estimate):.6f}')}</td>"
            f"<td>{escape('-' if ci_low is None else f'{float(ci_low):.6f}')}</td>"
            f"<td>{escape('-' if ci_high is None else f'{float(ci_high):.6f}')}</td>"
            "</tr>"
        )
    if not bootstrap_rows:
        bootstrap_rows.append("<tr><td colspan='4'>No bootstrap metrics available for winner.</td></tr>")

    winner_group_drift = (
        winner_candidate_row.get("group_drift")
        if isinstance(winner_candidate_row.get("group_drift"), dict)
        else {}
    )
    winner_group_ranges = (
        winner_group_drift.get("metric_ranges")
        if isinstance(winner_group_drift.get("metric_ranges"), dict)
        else {}
    )
    group_rows: list[str] = []
    for metric_name in ("f1", "roc_auc", "auprc", "mcc"):
        item = winner_group_ranges.get(metric_name)
        if not isinstance(item, dict):
            continue
        range_value = item.get("range")
        range_text = "-" if range_value is None else f"{float(range_value):.6f}"
        group_rows.append(
            "<tr>"
            f"<td>{escape(metric_name)}</td>"
            f"<td>{escape(str(item.get('min_group', '-')))}</td>"
            f"<td>{escape(str(item.get('max_group', '-')))}</td>"
            f"<td>{escape(range_text)}</td>"
            "</tr>"
        )
    if not group_rows:
        group_rows.append("<tr><td colspan='4'>No group-drift metric ranges for winner.</td></tr>")

    winner_learning_curve = (
        winner_candidate_row.get("learning_curve")
        if isinstance(winner_candidate_row.get("learning_curve"), dict)
        else {}
    )
    winner_learning_points = (
        winner_learning_curve.get("points") if isinstance(winner_learning_curve.get("points"), list) else []
    )
    learning_curve_svg = _render_learning_curve_svg(
        [item for item in winner_learning_points if isinstance(item, dict)]
    )
    ece_delta_badge = _calibration_delta_badge(
        label="ECE delta (raw-best)",
        value=winner_candidate_row.get("calibration_ece_improvement"),
    )
    brier_delta_badge = _calibration_delta_badge(
        label="Brier delta (raw-best)",
        value=winner_candidate_row.get("calibration_brier_improvement"),
    )

    winner_json = escape(json.dumps(winner, ensure_ascii=True, indent=2))
    reliability_sections = report.get("reliability_sections") if isinstance(report.get("reliability_sections"), dict) else {}
    reliability_json = escape(json.dumps(reliability_sections, ensure_ascii=True, indent=2))
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
        "<table><thead><tr><th>Candidate</th><th>Status</th><th>CV Objective</th><th>Test Objective</th><th>Gap(CV-Test)</th><th>Gap Ratio</th><th>Overfit Risk</th><th>F1</th><th>ROC-AUC</th><th>AUPRC</th><th>MCC</th><th>Precision</th><th>Recall</th><th>ECE(raw)</th><th>Brier(raw)</th><th>ECE Delta(raw-best)</th><th>Brier Delta(raw-best)</th><th>Objective</th><th>Runtime(s)</th><th>Train(s)</th><th>SingleInf(ms)</th><th>BatchInf(ms)</th><th>PeakVRAM(MB)</th><th>TrainMemDelta(MB)</th><th>TrainMemPeakDelta(MB)</th><th>Fold Std</th><th>Drift Group Column</th><th>Drift Group Count</th><th>Learning Points</th><th>ComputeCostReport</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></div>"
        "<div class='card'><h2>Reliability Visual Overview</h2>"
        f"<div><strong>Winner:</strong> {escape(winner_candidate_name or 'unknown')}</div>"
        "<h3>Calibration Delta Badges (Winner)</h3>"
        "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;'>"
        + ece_delta_badge
        + brier_delta_badge
        + "</div>"
        "<h3>Learning Curve (Winner)</h3>"
        + learning_curve_svg
        + "<h3>Holdout Bootstrap CI (Winner)</h3>"
        "<table><thead><tr><th>Metric</th><th>Point</th><th>CI Low</th><th>CI High</th></tr></thead><tbody>"
        + "".join(bootstrap_rows)
        + "</tbody></table>"
        "<h3>Group Drift Ranges (Winner)</h3>"
        "<table><thead><tr><th>Metric</th><th>Min Group</th><th>Max Group</th><th>Range</th></tr></thead><tbody>"
        + "".join(group_rows)
        + "</tbody></table></div>"
        "<div class='card'><h2>Reliability Sections</h2><pre>"
        + reliability_json
        + "</pre></div>"
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
