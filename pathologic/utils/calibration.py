"""Calibration utilities for binary probability outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _clip_probabilities(values: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    clipped = np.asarray(values, dtype=float).reshape(-1)
    return np.clip(clipped, eps, 1.0 - eps)


def _validate_binary_labels(y_true: np.ndarray) -> None:
    labels = np.unique(np.asarray(y_true, dtype=int))
    if labels.size < 2:
        raise ValueError("Calibration requires at least two classes in calibration labels.")


def apply_platt_scaling(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    target_scores: np.ndarray,
) -> np.ndarray:
    """Fit Platt scaling on calibration scores and transform target scores."""
    y_cal = np.asarray(calibration_labels, dtype=int).reshape(-1)
    _validate_binary_labels(y_cal)

    x_cal = _clip_probabilities(np.asarray(calibration_scores, dtype=float)).reshape(-1, 1)
    x_target = _clip_probabilities(np.asarray(target_scores, dtype=float)).reshape(-1, 1)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_cal, y_cal)
    return _clip_probabilities(model.predict_proba(x_target)[:, -1])


def apply_beta_scaling(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    target_scores: np.ndarray,
) -> np.ndarray:
    """Fit Beta scaling on calibration scores and transform target scores."""
    try:
        from betacal import BetaCalibration
    except Exception as exc:  # pragma: no cover - import failure path tested via integration
        raise RuntimeError(
            "Beta scaling requires 'betacal'. Install dependency and retry."
        ) from exc

    y_cal = np.asarray(calibration_labels, dtype=int).reshape(-1)
    _validate_binary_labels(y_cal)

    x_cal = _clip_probabilities(np.asarray(calibration_scores, dtype=float))
    x_target = _clip_probabilities(np.asarray(target_scores, dtype=float))

    calibrator = BetaCalibration()
    calibrator.fit(x_cal, y_cal)
    transformed = calibrator.predict(x_target)
    return _clip_probabilities(np.asarray(transformed, dtype=float))


def apply_isotonic_scaling(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    target_scores: np.ndarray,
) -> np.ndarray:
    """Fit isotonic regression on calibration scores and transform target scores."""
    y_cal = np.asarray(calibration_labels, dtype=int).reshape(-1)
    _validate_binary_labels(y_cal)

    x_cal = _clip_probabilities(np.asarray(calibration_scores, dtype=float))
    x_target = _clip_probabilities(np.asarray(target_scores, dtype=float))

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(x_cal, y_cal)
    return _clip_probabilities(np.asarray(calibrator.predict(x_target), dtype=float))


def _probabilities_to_logits(values: np.ndarray) -> np.ndarray:
    probs = _clip_probabilities(np.asarray(values, dtype=float))
    return np.log(probs / (1.0 - probs))


def _logits_to_probabilities(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=float)
    return _clip_probabilities(1.0 / (1.0 + np.exp(-logits)))


def apply_temperature_scaling(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    target_scores: np.ndarray,
) -> np.ndarray:
    """Fit temperature scaling on calibration scores and transform target scores."""
    y_cal = np.asarray(calibration_labels, dtype=int).reshape(-1)
    _validate_binary_labels(y_cal)

    logits_cal = _probabilities_to_logits(np.asarray(calibration_scores, dtype=float))
    logits_target = _probabilities_to_logits(np.asarray(target_scores, dtype=float))

    temperatures = np.geomspace(0.05, 5.0, num=101)
    best_temperature: float | None = None
    best_loss = float("inf")

    for temperature in temperatures:
        scaled = logits_cal / float(temperature)
        calibrated_probs = _logits_to_probabilities(scaled)
        loss = float(log_loss(y_cal, calibrated_probs, labels=[0, 1]))
        if not np.isfinite(loss):
            continue
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)

    if best_temperature is None:
        raise RuntimeError("Temperature scaling failed to find a valid temperature.")

    transformed = _logits_to_probabilities(logits_target / best_temperature)
    return _clip_probabilities(np.asarray(transformed, dtype=float))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute ECE with uniform bins and return bin-level stats."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = _clip_probabilities(np.asarray(y_prob, dtype=float))
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and y_prob must have equal length.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = int(y.shape[0])
    ece = 0.0
    bins: list[dict[str, Any]] = []

    for idx in range(n_bins):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_index": idx,
                    "left": left,
                    "right": right,
                    "count": 0,
                    "avg_predicted": None,
                    "avg_observed": None,
                    "abs_gap": None,
                }
            )
            continue

        avg_pred = float(np.mean(p[mask]))
        avg_obs = float(np.mean(y[mask]))
        abs_gap = abs(avg_obs - avg_pred)
        ece += (count / n) * abs_gap
        bins.append(
            {
                "bin_index": idx,
                "left": left,
                "right": right,
                "count": count,
                "avg_predicted": avg_pred,
                "avg_observed": avg_obs,
                "abs_gap": float(abs_gap),
            }
        )

    return float(ece), bins


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, Any]:
    """Build calibration report with Brier, ECE and distribution summary."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = _clip_probabilities(np.asarray(y_prob, dtype=float))
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and y_prob must have equal length.")

    ece, bins = expected_calibration_error(y, p, n_bins=n_bins)
    brier = float(brier_score_loss(y, p))

    return {
        "samples": int(y.shape[0]),
        "brier_score": brier,
        "ece": float(ece),
        "probability_distribution": {
            "mean": float(np.mean(p)),
            "std": float(np.std(p)),
            "min": float(np.min(p)),
            "max": float(np.max(p)),
            "p05": float(np.percentile(p, 5)),
            "p25": float(np.percentile(p, 25)),
            "p50": float(np.percentile(p, 50)),
            "p75": float(np.percentile(p, 75)),
            "p95": float(np.percentile(p, 95)),
        },
        "bins": bins,
    }


def save_probability_histogram(
    *,
    method_scores: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save overlaid histogram for probability distributions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, values in method_scores.items():
        clipped = _clip_probabilities(np.asarray(values, dtype=float))
        ax.hist(
            clipped,
            bins=20,
            range=(0.0, 1.0),
            alpha=0.4,
            density=True,
            label=method,
        )
    ax.set_title("Probability Distribution")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def save_reliability_diagram(
    *,
    y_true: np.ndarray,
    method_scores: Mapping[str, np.ndarray],
    output_path: Path,
    n_bins: int,
) -> None:
    """Save reliability diagram for one or more probability vectors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0, label="perfect")

    y = np.asarray(y_true, dtype=int).reshape(-1)
    for method, values in method_scores.items():
        probs = _clip_probabilities(np.asarray(values, dtype=float))
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.2, label=method)

    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
