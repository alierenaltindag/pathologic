"""Explainability artifact helpers for search workflows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pathologic import PathoLogic


def _candidate_slug(name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return token or "candidate"


def build_global_importance_label(item: dict[str, Any]) -> str:
    feature_name = str(item.get("feature", "unknown_feature"))
    biological_label = str(item.get("biological_label", "")).strip()
    if not biological_label:
        return feature_name

    if biological_label.lower() in {"unknown", feature_name.lower()}:
        return feature_name

    return f"{feature_name} ({biological_label})"


def build_hotspot_label(item: dict[str, Any]) -> str:
    group_column = str(item.get("group_column", "")).strip()
    if group_column and group_column in item:
        value = item.get(group_column)
        if value is not None and str(value).strip() != "":
            return f"{group_column}: {value}"

    excluded_keys = {
        "false_positive_count",
        "negative_count",
        "false_positive_rate",
        "overall_false_positive_rate",
        "false_positive_risk_ratio",
        "group_column",
    }
    for key, value in item.items():
        if key in excluded_keys:
            continue
        if value is not None and str(value).strip() != "":
            return f"{key}: {value}"

    return "unknown"


def save_global_importance_plot(
    *,
    global_importance: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [build_global_importance_label(item) for item in global_importance]
    values = [float(item.get("absolute_contribution", 0.0)) for item in global_importance]

    if not labels:
        labels = ["no_data"]
        values = [0.0]

    y_positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(3.0, len(labels) * 0.45)))
    ax.barh(y_positions, values, color="#2f80ed")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("absolute_contribution")
    ax.set_title("Global Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_fp_hotspots_plot(
    *,
    hotspots: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [build_hotspot_label(item) for item in hotspots]
    values = [float(item.get("false_positive_risk_ratio", 0.0)) for item in hotspots]

    if not labels:
        labels = ["no_data"]
        values = [0.0]

    y_positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(3.0, len(labels) * 0.45)))
    ax.barh(y_positions, values, color="#eb5757")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("false_positive_risk_ratio")
    ax.set_title("False Positive Hotspots")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_candidate_explainability_artifacts(
    *,
    model: PathoLogic,
    test_csv: str,
    run_dir: Path,
    candidate_name: str,
    top_k_features: int,
    top_k_samples: int,
    background_size: int,
    fp_top_k: int,
    fp_min_negative_count: int,
) -> dict[str, Any]:
    explain_dir = run_dir / "explainability" / _candidate_slug(candidate_name)
    explain_dir.mkdir(parents=True, exist_ok=True)

    html_path = explain_dir / "explainability_report.html"
    full_report = model.explain(
        test_csv,
        top_k_features=int(top_k_features),
        top_k_samples=int(top_k_samples),
        background_size=int(background_size),
        false_positive={
            "enabled": True,
            "top_k_hotspots": int(fp_top_k),
            "minimum_negative_count": int(fp_min_negative_count),
        },
        visual_report={
            "enabled": True,
            "save_path": str(html_path),
        },
    )

    if not isinstance(full_report, dict):
        raise RuntimeError("Explainability report must be a mapping payload.")

    full_report_path = explain_dir / "explainability_report.json"
    full_report_path.write_text(
        json.dumps(full_report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    global_importance = full_report.get("global_feature_importance", [])
    if not isinstance(global_importance, list):
        global_importance = []

    sample_explanations = full_report.get("sample_explanations", [])
    if not isinstance(sample_explanations, list):
        sample_explanations = []

    hotspots = full_report.get("false_positive_hotspots", [])
    if not isinstance(hotspots, list):
        hotspots = []

    hotspots_csv_path = explain_dir / "false_positive_hotspots.csv"
    pd.DataFrame(hotspots).to_csv(hotspots_csv_path, index=False)

    global_plot_path = explain_dir / "global_feature_importance.png"
    hotspots_plot_path = explain_dir / "false_positive_hotspots.png"
    save_global_importance_plot(global_importance=global_importance, output_path=global_plot_path)
    save_fp_hotspots_plot(hotspots=hotspots, output_path=hotspots_plot_path)

    summary_payload = {
        "status": "ok",
        "backend": str(full_report.get("backend", "unknown")),
        "metadata": dict(full_report.get("metadata", {}))
        if isinstance(full_report.get("metadata"), dict)
        else {},
        "global_top_features": global_importance[: max(1, int(top_k_features))],
        "false_positive_hotspots_top": hotspots[: max(1, int(fp_top_k))],
        "sample_explanations_count": int(len(sample_explanations)),
        "artifacts": {
            "explainability_report_json": str(full_report_path),
            "explainability_report_html": str(html_path),
            "false_positive_hotspots_csv": str(hotspots_csv_path),
            "global_feature_importance_png": str(global_plot_path),
            "false_positive_hotspots_png": str(hotspots_plot_path),
        },
    }

    summary_path = explain_dir / "explainability_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    summary_payload["artifacts"]["explainability_summary_json"] = str(summary_path)
    return summary_payload
