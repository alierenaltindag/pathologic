from __future__ import annotations

import numpy as np
import pytest

from pathologic.utils.calibration import (
    apply_beta_scaling,
    apply_platt_scaling,
    calibration_report,
    expected_calibration_error,
)


def test_expected_calibration_error_is_zero_for_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    ece, bins = expected_calibration_error(y_true, y_prob, n_bins=4)

    assert ece == pytest.approx(0.0, abs=1e-5)
    assert len(bins) == 4


def test_calibration_report_contains_brier_and_ece() -> None:
    y_true = np.array([0, 1, 1, 0, 1], dtype=int)
    y_prob = np.array([0.1, 0.8, 0.6, 0.3, 0.7], dtype=float)

    report = calibration_report(y_true, y_prob, n_bins=5)

    assert 0.0 <= float(report["brier_score"]) <= 1.0
    assert 0.0 <= float(report["ece"]) <= 1.0
    assert isinstance(report["bins"], list)


def test_platt_scaling_returns_valid_probabilities() -> None:
    y_cal = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    s_cal = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=float)
    s_test = np.array([0.15, 0.5, 0.95], dtype=float)

    calibrated = apply_platt_scaling(s_cal, y_cal, s_test)

    assert calibrated.shape == (3,)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_beta_scaling_returns_valid_probabilities() -> None:
    pytest.importorskip("betacal")

    y_cal = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    s_cal = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=float)
    s_test = np.array([0.15, 0.5, 0.95], dtype=float)

    calibrated = apply_beta_scaling(s_cal, y_cal, s_test)

    assert calibrated.shape == (3,)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
