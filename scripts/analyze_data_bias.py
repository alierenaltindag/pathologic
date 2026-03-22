"""Generate an HTML data bias report from a labeled CSV dataset.

This script inspects class imbalance and group-level bias patterns and renders
an HTML report with embedded plots. A different CSV can be selected via CLI.
"""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _detect_label_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    candidates: list[str] = []
    if preferred and preferred.strip():
        candidates.append(preferred.strip())
    candidates.extend(["label", "Label", "target", "Target", "class", "Class"])

    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        "Could not detect label column. Use --label-column to specify one explicitly."
    )


def _validate_binary_label(series: pd.Series, label_column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    clean = values.dropna()
    unique = set(clean.astype(int).unique().tolist())
    if not unique.issubset({0, 1}) or len(unique) < 2:
        raise ValueError(
            f"Label column '{label_column}' must contain binary values 0/1 with both classes present."
        )
    return values.astype(int)


def _parse_group_columns(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _to_base64_png(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _plot_class_distribution(y: pd.Series) -> str:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Class 0", "Class 1"]
    values = [int(counts.loc[0]), int(counts.loc[1])]
    bars = ax.bar(labels, values, color=["#4E79A7", "#E15759"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    for bar in bars:
        height = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")
    return _to_base64_png(fig)


def _compute_imbalance_stats(y: pd.Series) -> dict[str, Any]:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    negatives = int(counts.loc[0])
    positives = int(counts.loc[1])
    total = int(len(y))

    minority = min(negatives, positives)
    majority = max(negatives, positives)
    imbalance_ratio = float(majority / max(minority, 1))

    return {
        "total_samples": total,
        "class_0_count": negatives,
        "class_1_count": positives,
        "class_0_ratio": float(negatives / max(total, 1)),
        "class_1_ratio": float(positives / max(total, 1)),
        "imbalance_ratio": imbalance_ratio,
        "minority_class": int(0 if negatives <= positives else 1),
    }


def _compute_group_bias(
    df: pd.DataFrame,
    *,
    label_column: str,
    group_columns: list[str],
    min_group_size: int,
    top_k_groups: int,
) -> pd.DataFrame:
    y = df[label_column].astype(int)
    global_rate = float(y.mean())
    rows: list[dict[str, Any]] = []

    for column in group_columns:
        if column not in df.columns:
            continue

        series = df[column].fillna("__missing__").astype(str)
        grouped = (
            pd.DataFrame({"group_value": series, "label": y})
            .groupby("group_value", dropna=False)
            .agg(sample_count=("label", "size"), positives=("label", "sum"))
            .reset_index()
        )

        grouped = grouped[grouped["sample_count"] >= int(min_group_size)]
        if grouped.empty:
            continue

        grouped["negative_count"] = grouped["sample_count"] - grouped["positives"]
        grouped["positive_rate"] = grouped["positives"] / grouped["sample_count"]
        grouped["global_positive_rate"] = global_rate
        grouped["abs_rate_gap"] = (grouped["positive_rate"] - global_rate).abs()
        grouped["risk_ratio"] = grouped["positive_rate"] / max(global_rate, 1e-9)

        for _, item in grouped.iterrows():
            rows.append(
                {
                    "group_column": column,
                    "group_value": str(item["group_value"]),
                    "sample_count": int(item["sample_count"]),
                    "positive_count": int(item["positives"]),
                    "negative_count": int(item["negative_count"]),
                    "positive_rate": float(item["positive_rate"]),
                    "global_positive_rate": float(item["global_positive_rate"]),
                    "abs_rate_gap": float(item["abs_rate_gap"]),
                    "risk_ratio": float(item["risk_ratio"]),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "group_column",
                "group_value",
                "sample_count",
                "positive_count",
                "negative_count",
                "positive_rate",
                "global_positive_rate",
                "abs_rate_gap",
                "risk_ratio",
            ]
        )

    output = pd.DataFrame(rows).sort_values(
        by=["abs_rate_gap", "sample_count"],
        ascending=[False, False],
    )
    return output.head(int(top_k_groups)).reset_index(drop=True)


def _compute_missingness_by_class(df: pd.DataFrame, *, label_column: str, top_k: int = 20) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col != label_column]
    if not feature_cols:
        return pd.DataFrame(columns=["feature", "missing_rate_class0", "missing_rate_class1", "abs_gap"])

    class0 = df[df[label_column] == 0]
    class1 = df[df[label_column] == 1]

    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        miss0 = float(class0[col].isna().mean()) if len(class0) else 0.0
        miss1 = float(class1[col].isna().mean()) if len(class1) else 0.0
        rows.append(
            {
                "feature": col,
                "missing_rate_class0": miss0,
                "missing_rate_class1": miss1,
                "abs_gap": abs(miss1 - miss0),
            }
        )

    out = pd.DataFrame(rows).sort_values(by="abs_gap", ascending=False)
    return out.head(int(top_k)).reset_index(drop=True)


def _plot_group_bias(group_bias: pd.DataFrame) -> str | None:
    if group_bias.empty:
        return None

    labels = [f"{row.group_column}:{row.group_value}" for row in group_bias.itertuples()]
    values = group_bias["abs_rate_gap"].to_numpy(dtype=float)
    y_pos = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(10, max(4, len(values) * 0.4)))
    ax.barh(y_pos, values, color="#F28E2B")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Absolute Positive-Rate Gap")
    ax.set_title("Top Group Bias Signals")
    return _to_base64_png(fig)


def _plot_missingness_gap(missingness: pd.DataFrame) -> str | None:
    if missingness.empty:
        return None

    labels = missingness["feature"].astype(str).tolist()
    values = missingness["abs_gap"].to_numpy(dtype=float)
    y_pos = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(10, max(4, len(values) * 0.35)))
    ax.barh(y_pos, values, color="#59A14F")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Missingness Gap |class1 - class0|")
    ax.set_title("Top Missingness Bias Signals")
    return _to_base64_png(fig)


def _render_table(df: pd.DataFrame, *, max_rows: int = 50) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    shown = df.head(max_rows).copy()
    return shown.to_html(index=False, border=0, classes="table")


def generate_data_bias_report(
    *,
    csv_path: str,
    output_html: str,
    label_column: str | None,
    group_columns: list[str],
    min_group_size: int,
    top_k_groups: int,
) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    resolved_label = _detect_label_column(df, preferred=label_column)
    df = df.copy()
    df[resolved_label] = _validate_binary_label(df[resolved_label], resolved_label)

    imbalance_stats = _compute_imbalance_stats(df[resolved_label])
    group_bias = _compute_group_bias(
        df,
        label_column=resolved_label,
        group_columns=group_columns,
        min_group_size=min_group_size,
        top_k_groups=top_k_groups,
    )
    missingness = _compute_missingness_by_class(df, label_column=resolved_label)

    class_plot_b64 = _plot_class_distribution(df[resolved_label])
    group_plot_b64 = _plot_group_bias(group_bias)
    missing_plot_b64 = _plot_missingness_gap(missingness)

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = """
    <html>
    <head>
      <meta charset='utf-8'>
      <title>PathoLogic Data Bias Report</title>
      <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 24px; background: #f7fafc; color: #1f2937; }
        h1, h2 { color: #0f2942; }
        .card { background: #fff; border: 1px solid #dbe4ee; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
        .kpi { background: #eef4fb; border-radius: 8px; padding: 10px; }
        .kpi .k { font-size: 11px; text-transform: uppercase; color: #486581; }
        .kpi .v { font-size: 19px; font-weight: 700; color: #0f2942; }
        .table { width: 100%; border-collapse: collapse; }
        .table th, .table td { border: 1px solid #dbe4ee; padding: 6px; text-align: left; font-size: 13px; }
        .table th { background: #eaf1f8; }
        img { max-width: 100%; border: 1px solid #dbe4ee; border-radius: 6px; }
      </style>
    </head>
    <body>
    """

    html += (
        f"<h1>PathoLogic Data Bias Report</h1>"
        f"<p><strong>Dataset:</strong> {Path(csv_path).name} | <strong>Rows:</strong> {len(df)} | "
        f"<strong>Label Column:</strong> {resolved_label}</p>"
    )

    html += "<div class='card'><h2>Class Imbalance</h2><div class='grid'>"
    for key in [
        "total_samples",
        "class_0_count",
        "class_1_count",
        "class_0_ratio",
        "class_1_ratio",
        "imbalance_ratio",
        "minority_class",
    ]:
        value = imbalance_stats[key]
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        html += f"<div class='kpi'><div class='k'>{key}</div><div class='v'>{shown}</div></div>"
    html += "</div>"
    html += f"<div style='margin-top:12px'><img src='data:image/png;base64,{class_plot_b64}' alt='class_distribution'></div>"
    html += "</div>"

    html += "<div class='card'><h2>Group Bias (Positive-Rate Gap)</h2>"
    if group_plot_b64:
        html += f"<img src='data:image/png;base64,{group_plot_b64}' alt='group_bias'>"
    html += _render_table(group_bias)
    html += "</div>"

    html += "<div class='card'><h2>Missingness Bias by Class</h2>"
    if missing_plot_b64:
        html += f"<img src='data:image/png;base64,{missing_plot_b64}' alt='missingness_bias'>"
    html += _render_table(missingness)
    html += "</div>"

    html += "</body></html>"
    output_path.write_text(html, encoding="utf-8")

    return {
        "csv_path": str(csv_path),
        "output_html": str(output_path),
        "label_column": resolved_label,
        "rows": int(len(df)),
        "imbalance": imbalance_stats,
        "group_bias_rows": int(len(group_bias)),
        "missingness_rows": int(len(missingness)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate class-imbalance and data-bias HTML diagnostics from CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data.csv",
        help="Input CSV file path (default: data.csv).",
    )
    parser.add_argument(
        "--output-html",
        default="results/data_bias_report.html",
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional label column name; auto-detected when omitted.",
    )
    parser.add_argument(
        "--group-columns",
        default="Veri_Kaynagi_Paneli,Gene(s),Ref_AA,Alt_AA",
        help="Comma-separated categorical columns for group-bias analysis.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=20,
        help="Minimum group sample size included in bias table.",
    )
    parser.add_argument(
        "--top-k-groups",
        type=int,
        default=30,
        help="Maximum number of high-gap groups shown in report.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    payload = generate_data_bias_report(
        csv_path=args.csv_path,
        output_html=args.output_html,
        label_column=args.label_column,
        group_columns=_parse_group_columns(args.group_columns),
        min_group_size=int(args.min_group_size),
        top_k_groups=int(args.top_k_groups),
    )
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
