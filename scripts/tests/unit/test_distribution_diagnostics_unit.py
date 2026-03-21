from __future__ import annotations

import numpy as np

from pathologic.utils.distribution_diagnostics import normality_report


def test_normality_report_contains_shapiro_and_anderson() -> None:
    values = np.array([0.1, 0.2, 0.25, 0.3, 0.7, 0.9], dtype=float)

    report = normality_report(values)

    assert report["shapiro"]["status"] == "ok"
    assert report["anderson"]["status"] == "ok"
    assert "statistic" in report["shapiro"]
    assert "statistic" in report["anderson"]


def test_normality_report_skips_small_samples() -> None:
    values = np.array([0.4, 0.6], dtype=float)

    report = normality_report(values)

    assert report["shapiro"]["status"] == "skipped"
    assert report["anderson"]["status"] == "skipped"
