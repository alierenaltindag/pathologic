"""Error-analysis and calibration artifact helpers for search workflows."""

from __future__ import annotations

from html import escape
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pathologic import PathoLogic
from pathologic.engine import Evaluator
from pathologic.explain.error_analysis import MultiDimensionalErrorAnalyzer
from pathologic.explain.visualizer import ExplainabilityVisualizer
from pathologic.utils.calibration import (
    apply_beta_scaling,
    apply_isotonic_scaling,
    apply_platt_scaling,
    apply_temperature_scaling,
    calibration_report,
    save_probability_histogram,
    save_reliability_diagram,
)
from pathologic.utils.distribution_diagnostics import normality_report, save_qq_plot


def compute_holdout_bootstrap_artifacts(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = 400,
    confidence_level: float = 0.95,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        return {
            "status": "failed",
            "reason": "shape_mismatch",
            "sample_count": int(y_true_arr.shape[0]),
        }
    if y_true_arr.shape[0] == 0:
        return {
            "status": "failed",
            "reason": "empty_input",
            "sample_count": 0,
        }

    y_pred_arr = (y_score_arr >= float(threshold)).astype(int)
    point_metrics: dict[str, float | None] = {
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "roc_auc": None,
        "auprc": None,
    }
    if np.unique(y_true_arr).size >= 2:
        point_metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
        point_metrics["auprc"] = float(average_precision_score(y_true_arr, y_score_arr))

    rng = np.random.default_rng(int(seed))
    alpha = 1.0 - float(confidence_level)
    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - (alpha / 2.0))

    metric_samples: dict[str, list[float]] = {
        "f1": [],
        "mcc": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
        "auprc": [],
    }

    sample_count = int(y_true_arr.shape[0])
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, sample_count, size=sample_count)
        y_true_b = y_true_arr[idx]
        y_score_b = y_score_arr[idx]
        y_pred_b = (y_score_b >= float(threshold)).astype(int)

        metric_samples["f1"].append(float(f1_score(y_true_b, y_pred_b, zero_division=0)))
        metric_samples["mcc"].append(float(matthews_corrcoef(y_true_b, y_pred_b)))
        metric_samples["precision"].append(
            float(precision_score(y_true_b, y_pred_b, zero_division=0))
        )
        metric_samples["recall"].append(float(recall_score(y_true_b, y_pred_b, zero_division=0)))

        if np.unique(y_true_b).size >= 2:
            metric_samples["roc_auc"].append(float(roc_auc_score(y_true_b, y_score_b)))
            metric_samples["auprc"].append(float(average_precision_score(y_true_b, y_score_b)))

    metric_ci: dict[str, Any] = {}
    for metric_name, samples in metric_samples.items():
        point_estimate = point_metrics.get(metric_name)
        if not samples:
            metric_ci[metric_name] = {
                "point_estimate": point_estimate,
                "ci_low": None,
                "ci_high": None,
                "successful_resamples": 0,
            }
            continue
        values = np.asarray(samples, dtype=float)
        metric_ci[metric_name] = {
            "point_estimate": point_estimate,
            "ci_low": float(np.percentile(values, low_q)),
            "ci_high": float(np.percentile(values, high_q)),
            "successful_resamples": int(values.shape[0]),
        }

    return {
        "status": "ok",
        "sample_count": sample_count,
        "n_resamples": int(n_resamples),
        "confidence_level": float(confidence_level),
        "threshold": float(threshold),
        "metrics": metric_ci,
    }


def candidate_slug(name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return token or "candidate"


def extract_prediction_payload(
    *,
    model: PathoLogic,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> dict[str, Any]:
    if model._preprocessor is None or model._trained_model is None:  # noqa: SLF001
        raise RuntimeError("Model must be trained before calibration score extraction.")

    transformed = model._preprocessor.transform(dataset)  # noqa: SLF001
    model_feature_columns = getattr(model, "_feature_columns", None)
    resolved_feature_columns = (
        list(model_feature_columns) if model_feature_columns else list(feature_columns)
    )
    x = transformed[resolved_feature_columns].to_numpy(dtype=float)
    y = transformed[label_column].to_numpy(dtype=int)

    probabilities = np.asarray(model._trained_model.predict_proba(x))  # noqa: SLF001
    if probabilities.ndim == 1:
        scores = probabilities
    else:
        scores = probabilities[:, -1]
    y_score = np.asarray(scores, dtype=float).reshape(-1)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "transformed": transformed,
        "y_true": y,
        "y_score": y_score,
        "y_pred": y_pred,
    }


def extract_scores_from_model(
    *,
    model: PathoLogic,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    payload = extract_prediction_payload(
        model=model,
        dataset=dataset,
        feature_columns=feature_columns,
        label_column=label_column,
    )
    return payload["y_true"], payload["y_score"]


def compute_candidate_error_analysis_artifacts(
    *,
    model: PathoLogic,
    dataset: pd.DataFrame,
    run_dir: Path,
    candidate_name: str,
    feature_columns: list[str],
    detailed: bool,
) -> dict[str, Any]:
    payload = extract_prediction_payload(
        model=model,
        dataset=dataset,
        feature_columns=feature_columns,
        label_column="label",
    )

    analyzer = MultiDimensionalErrorAnalyzer(random_state=42)
    error_dir = run_dir / "error_analysis" / candidate_slug(candidate_name)
    result = analyzer.analyze_candidate(
        candidate_name=candidate_name,
        y_true=np.asarray(payload["y_true"], dtype=int),
        y_pred=np.asarray(payload["y_pred"], dtype=int),
        y_score=np.asarray(payload["y_score"], dtype=float),
        dataset=payload["transformed"],
        output_dir=error_dir,
        detailed=detailed,
    )
    
    # Render standalone HTML report
    visualizer = ExplainabilityVisualizer()
    report_path = error_dir / "error_analysis_report.html"
    visualizer.render_error_report_html(result.to_dict(), str(report_path))
    
    return result.to_dict()


def build_error_analysis_run_summary(
    *,
    leaderboard_rows: list[dict[str, Any]],
    winner_candidate: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in leaderboard_rows:
        candidate_name = str(item.get("candidate", "unknown"))
        payload = item.get("error_analysis")
        if not isinstance(payload, dict):
            rows.append(
                {
                    "candidate": candidate_name,
                    "status": "missing",
                }
            )
            continue

        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        surrogate = summary.get("surrogate_tree") if isinstance(summary.get("surrogate_tree"), dict) else {}
        clustering = summary.get("clustering") if isinstance(summary.get("clustering"), dict) else {}

        row: dict[str, Any] = {
            "candidate": candidate_name,
            "is_winner": bool(candidate_name == winner_candidate),
            "status": str(payload.get("status", "unknown")),
            "error_count": int(summary.get("error_count", 0)) if summary else 0,
            "error_rate": float(summary.get("error_rate", 0.0)) if summary else 0.0,
            "surrogate_status": str(surrogate.get("status", "unknown")),
            "clustering_status": str(clustering.get("status", "unknown")),
        }
        rows.append(row)

    return {
        "winner_candidate": winner_candidate,
        "rows": rows,
    }


def compute_candidate_calibration_artifacts(
    *,
    run_dir: Path,
    candidate_name: str,
    y_calibration: np.ndarray,
    score_calibration: np.ndarray,
    y_test: np.ndarray,
    score_test: np.ndarray,
    bins: int,
) -> dict[str, Any]:
    calibration_dir = run_dir / "calibration" / candidate_slug(candidate_name)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    method_scores: dict[str, np.ndarray] = {
        "raw": np.asarray(score_test, dtype=float),
    }
    method_reports: dict[str, dict[str, Any]] = {
        "raw": calibration_report(y_test, method_scores["raw"], n_bins=bins),
    }
    method_normality: dict[str, dict[str, Any]] = {
        "raw": normality_report(method_scores["raw"]),
    }

    method_builders: dict[str, Any] = {
        "platt": apply_platt_scaling,
        "beta": apply_beta_scaling,
        "isotonic": apply_isotonic_scaling,
        "temperature": apply_temperature_scaling,
    }
    for method_name, method_builder in method_builders.items():
        try:
            method_scores[method_name] = method_builder(
                score_calibration,
                y_calibration,
                score_test,
            )
            method_reports[method_name] = calibration_report(
                y_test,
                method_scores[method_name],
                n_bins=bins,
            )
            method_normality[method_name] = normality_report(method_scores[method_name])
        except Exception as exc:
            method_reports[method_name] = {"status": "failed", "reason": str(exc)}
            method_normality[method_name] = {"status": "failed", "reason": str(exc)}

    histogram_path = calibration_dir / "probability_histogram.png"
    reliability_path = calibration_dir / "reliability_diagram.png"
    qq_plot_path = calibration_dir / "qq_plot.png"
    save_probability_histogram(method_scores=method_scores, output_path=histogram_path)
    save_reliability_diagram(
        y_true=y_test,
        method_scores=method_scores,
        output_path=reliability_path,
        n_bins=bins,
    )
    save_qq_plot(method_scores=method_scores, output_path=qq_plot_path)

    bin_rows: list[dict[str, Any]] = []
    for method_name, report in method_reports.items():
        report_bins = report.get("bins") if isinstance(report, dict) else None
        if not isinstance(report_bins, list):
            continue
        for item in report_bins:
            if isinstance(item, dict):
                row = dict(item)
                row["method"] = method_name
                bin_rows.append(row)
    bins_csv_path = calibration_dir / "reliability_bins.csv"
    pd.DataFrame(bin_rows).to_csv(bins_csv_path, index=False)

    # Keep calibration_report.json compact by excluding per-bin arrays.
    method_reports_public: dict[str, dict[str, Any]] = {}
    for method_name, report in method_reports.items():
        if not isinstance(report, dict):
            continue
        method_reports_public[method_name] = {
            key: value
            for key, value in report.items()
            if key != "bins"
        }

    methods_summary: dict[str, Any] = {}
    for method_name, report in method_reports.items():
        if not isinstance(report, dict):
            continue
        if "brier_score" in report and "ece" in report:
            methods_summary[method_name] = {
                "status": "ok",
                "brier_score": float(report["brier_score"]),
                "ece": float(report["ece"]),
                "samples": int(report.get("samples", 0)),
            }
        else:
            methods_summary[method_name] = {
                "status": "failed",
                "reason": str(report.get("reason", "unknown")),
            }

    calibration_payload = {
        "candidate": candidate_name,
        "methods": method_reports_public,
        "normality_tests": method_normality,
        "summary": methods_summary,
        "artifacts": {
            "histogram_png": str(histogram_path),
            "reliability_png": str(reliability_path),
            "qq_plot_png": str(qq_plot_path),
            "reliability_bins_csv": str(bins_csv_path),
        },
    }
    calibration_json_path = calibration_dir / "calibration_report.json"
    calibration_json_path.write_text(
        json.dumps(calibration_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    calibration_payload["artifacts"]["calibration_report_json"] = str(calibration_json_path)

    # Render standalone candidate-level calibration HTML with detailed bins and diagnostics.
    visualizer = ExplainabilityVisualizer()
    calibration_html_path = calibration_dir / "calibration_report.html"
    calibration_payload_for_html = {
        "candidate": candidate_name,
        "methods": method_reports,
        "normality_tests": method_normality,
        "summary": methods_summary,
        "artifacts": calibration_payload["artifacts"],
    }
    visualizer.render_calibration_report_html(
        calibration_payload_for_html,
        str(calibration_html_path),
    )
    calibration_payload["artifacts"]["calibration_report_html"] = str(calibration_html_path)
    return calibration_payload


def compute_candidate_panel_threshold_artifacts(
    *,
    model: PathoLogic,
    dataset: pd.DataFrame,
    run_dir: Path,
    candidate_name: str,
    feature_columns: list[str],
    panel_column: str = "Veri_Kaynagi_Paneli",
    label_column: str = "label",
    min_samples: int = 1,
    default_threshold: float = 0.5,
) -> dict[str, Any]:
    payload = extract_prediction_payload(
        model=model,
        dataset=dataset,
        feature_columns=feature_columns,
        label_column=label_column,
    )
    transformed = payload["transformed"]

    if panel_column not in transformed.columns:
        return {
            "candidate": candidate_name,
            "status": "skipped",
            "reason": f"missing_panel_column:{panel_column}",
            "panel_column": panel_column,
            "rows": [],
            "artifacts": {},
        }

    panel_rows = Evaluator.panel_oof_f1_max_thresholds(
        y_true=np.asarray(payload["y_true"], dtype=int),
        y_score=np.asarray(payload["y_score"], dtype=float),
        panel_values=transformed[panel_column].astype(str).to_numpy(),
        min_samples=int(min_samples),
        default_threshold=float(default_threshold),
    )

    panel_dir = run_dir / "panel_thresholds" / candidate_slug(candidate_name)
    panel_dir.mkdir(parents=True, exist_ok=True)

    panel_payload: dict[str, Any] = {
        "candidate": candidate_name,
        "status": "ok",
        "panel_column": panel_column,
        "rows": panel_rows,
        "summary": {
            "panel_count": int(len(panel_rows)),
            "optimized_panel_count": int(sum(int(row.get("optimized", 0)) for row in panel_rows)),
            "default_threshold": float(default_threshold),
            "min_samples": int(min_samples),
        },
        "artifacts": {},
    }

    panel_json_path = panel_dir / "panel_thresholds_report.json"
    panel_json_path.write_text(
        json.dumps(panel_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    panel_payload["artifacts"]["panel_thresholds_report_json"] = str(panel_json_path)

    panel_html_path = panel_dir / "panel_thresholds_report.html"
    ExplainabilityVisualizer().render_panel_threshold_report_html(
        panel_payload,
        str(panel_html_path),
    )
    panel_payload["artifacts"]["panel_thresholds_report_html"] = str(panel_html_path)

    return panel_payload


def _render_compute_cost_html(*, payload: dict[str, Any], output_path: Path) -> None:
    def _flatten(prefix: str, value: Any, output: list[tuple[str, str]]) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                _flatten(child_prefix, nested, output)
            return
        if isinstance(value, list):
            output.append((prefix, json.dumps(value, ensure_ascii=True)))
            return
        output.append((prefix, str(value)))

    def _kv_rows(values: dict[str, Any]) -> str:
        flattened: list[tuple[str, str]] = []
        _flatten("", values, flattened)
        rows: list[str] = []
        for key, rendered in flattened:
            rows.append(
                "<tr>"
                f"<td>{escape(str(key))}</td>"
                f"<td>{escape(str(rendered))}</td>"
                "</tr>"
            )
        return "".join(rows)

    system_info = payload.get("system") if isinstance(payload.get("system"), dict) else {}
    frameworks = payload.get("frameworks") if isinstance(payload.get("frameworks"), dict) else {}
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    inference = payload.get("inference") if isinstance(payload.get("inference"), dict) else {}
    reproducibility = (
        payload.get("reproducibility") if isinstance(payload.get("reproducibility"), dict) else {}
    )

    html = (
        "<html><head><meta charset='utf-8'><title>Candidate Compute Cost</title>"
        "<style>"
        "body{font-family:Segoe UI,Tahoma,sans-serif;line-height:1.5;color:#1f2937;max-width:1200px;margin:0 auto;padding:20px;background:#f3f6fb;}"
        "h1,h2{color:#0f2942;}"
        "table{width:100%;border-collapse:collapse;margin-top:8px;}"
        "th,td{border:1px solid #dbe4ee;padding:8px;text-align:left;background:#fff;font-size:12px;vertical-align:top;}"
        "th{background:#eaf1f8;font-weight:700;}"
        ".card{background:#fff;border:1px solid #dbe4ee;border-radius:8px;padding:14px;margin-bottom:14px;}"
        "</style></head><body>"
        "<h1>Candidate Compute Cost Report</h1>"
        "<div class='card'>"
        f"<div><strong>Status:</strong> {escape(str(payload.get('status', 'unknown')))}</div>"
        f"<div><strong>Candidate:</strong> {escape(str(payload.get('candidate', 'unknown')))}</div>"
        "</div>"
        "<div class='card'><h2>System</h2><table><tbody>"
        + _kv_rows(system_info)
        + "</tbody></table></div>"
        "<div class='card'><h2>Framework Versions</h2><table><tbody>"
        + _kv_rows(frameworks)
        + "</tbody></table></div>"
        "<div class='card'><h2>Training Cost</h2><table><tbody>"
        + _kv_rows(training)
        + "</tbody></table></div>"
        "<div class='card'><h2>Inference Cost</h2><table><tbody>"
        + _kv_rows(inference)
        + "</tbody></table></div>"
        "<div class='card'><h2>Reproducibility</h2><table><tbody>"
        + _kv_rows(reproducibility)
        + "</tbody></table></div>"
        "</body></html>"
    )
    output_path.write_text(html, encoding="utf-8")


def compute_candidate_compute_cost_artifacts(
    *,
    run_dir: Path,
    candidate_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    compute_dir = run_dir / "compute_cost" / candidate_slug(candidate_name)
    compute_dir.mkdir(parents=True, exist_ok=True)

    compute_payload = dict(payload)
    compute_payload["candidate"] = candidate_name

    json_path = compute_dir / "compute_cost_report.json"
    html_path = compute_dir / "compute_cost_report.html"

    json_path.write_text(
        json.dumps(compute_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    _render_compute_cost_html(payload=compute_payload, output_path=html_path)

    artifacts = compute_payload.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    artifacts["compute_cost_report_json"] = str(json_path)
    artifacts["compute_cost_report_html"] = str(html_path)
    compute_payload["artifacts"] = artifacts
    return compute_payload
