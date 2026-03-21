"""Benchmark report helpers for summarizing hardware bottlenecks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _extract_timing(row: Mapping[str, Any], key: str) -> float | None:
    section = row.get(key)
    if not isinstance(section, Mapping):
        return None
    value = section.get("avg_seconds")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _hardware_bottleneck_summary(report: Mapping[str, Any]) -> str:
    """Return a compact textual summary of train/predict bottleneck ratios.

    The report is expected to include a ``models`` sequence where each item has:
    - ``model`` name
    - ``train.avg_seconds``
    - ``predict.avg_seconds``
    """
    models = report.get("models")
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
        return "No benchmark models found."

    valid_rows: list[dict[str, float | str]] = []
    for row in models:
        if not isinstance(row, Mapping):
            continue
        model = row.get("model")
        train_avg = _extract_timing(row, "train")
        predict_avg = _extract_timing(row, "predict")
        if isinstance(model, str) and train_avg is not None and predict_avg is not None:
            valid_rows.append(
                {
                    "model": model,
                    "train_avg": train_avg,
                    "predict_avg": predict_avg,
                }
            )

    if not valid_rows:
        return "No valid benchmark timings found."

    slowest_train = max(valid_rows, key=lambda item: float(item["train_avg"]))
    fastest_train = min(valid_rows, key=lambda item: float(item["train_avg"]))
    slowest_predict = max(valid_rows, key=lambda item: float(item["predict_avg"]))
    fastest_predict = min(valid_rows, key=lambda item: float(item["predict_avg"]))

    fastest_train_value = max(float(fastest_train["train_avg"]), 1e-12)
    fastest_predict_value = max(float(fastest_predict["predict_avg"]), 1e-12)

    training_ratio = float(slowest_train["train_avg"]) / fastest_train_value
    inference_ratio = float(slowest_predict["predict_avg"]) / fastest_predict_value

    return "\n".join(
        [
            f"Slowest training model: {slowest_train['model']}",
            f"Fastest training model: {fastest_train['model']}",
            f"Training bottleneck ratio: {training_ratio:.2f}x",
            f"Inference bottleneck ratio: {inference_ratio:.2f}x",
        ]
    )
