"""Distribution diagnostics for model probability outputs."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _to_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def normality_report(values: np.ndarray) -> dict[str, Any]:
    """Return Shapiro-Wilk and Anderson-Darling normality diagnostics."""
    x = _to_1d(values)
    if x.shape[0] < 3:
        return {
            "n": int(x.shape[0]),
            "shapiro": {"status": "skipped", "reason": "requires_at_least_3_samples"},
            "anderson": {"status": "skipped", "reason": "requires_at_least_3_samples"},
        }

    x_shapiro = x[: min(5000, x.shape[0])]
    shapiro_stat, shapiro_p = stats.shapiro(x_shapiro)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        anderson_result = stats.anderson(x, dist="norm")

    return {
        "n": int(x.shape[0]),
        "shapiro": {
            "status": "ok",
            "tested_n": int(x_shapiro.shape[0]),
            "statistic": float(shapiro_stat),
            "p_value": float(shapiro_p),
        },
        "anderson": {
            "status": "ok",
            "statistic": float(anderson_result.statistic),
            "critical_values": [float(v) for v in anderson_result.critical_values],
            "significance_levels": [float(v) for v in anderson_result.significance_level],
        },
    }


def save_qq_plot(*, method_scores: Mapping[str, np.ndarray], output_path: Path) -> None:
    """Save QQ plots of method score distributions against normal distribution."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_names = list(method_scores.keys())
    cols = 2
    rows = int(np.ceil(len(method_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes_array = np.atleast_1d(axes).reshape(-1)

    for idx, method in enumerate(method_names):
        ax = axes_array[idx]
        values = _to_1d(method_scores[method])
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(f"QQ Plot - {method}")

    for idx in range(len(method_names), len(axes_array)):
        axes_array[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
