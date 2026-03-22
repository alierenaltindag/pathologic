"""Error-analysis and calibration artifact helpers for search workflows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
