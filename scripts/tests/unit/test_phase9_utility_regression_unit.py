"""Phase 9 unit regression tests for critical utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pathologic.core_helpers import resolve_finetune_config, resolve_split_config
from pathologic.utils.benchmark import benchmark_callable
from pathologic.utils.hardware import detect_preferred_device
from scripts.benchmark_report import _hardware_bottleneck_summary


def test_phase9_benchmark_callable_returns_timing_payload() -> None:
    calls = {"count": 0}

    def _target() -> int:
        calls["count"] += 1
        return 7

    output, payload = benchmark_callable(name="unit_target", func=_target, runs=4)

    assert output == 7
    assert payload.name == "unit_target"
    assert payload.runs == 4
    assert payload.total_seconds >= 0.0
    assert payload.avg_seconds >= 0.0
    assert calls["count"] == 4


def test_phase9_benchmark_callable_rejects_non_positive_runs() -> None:
    with pytest.raises(ValueError, match="runs must be greater than 0"):
        benchmark_callable(name="invalid", func=lambda: 1, runs=0)


def test_phase9_detect_preferred_device_prefers_mps_when_cuda_missing(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert detect_preferred_device() == "mps"


def test_phase9_resolve_split_config_precedence() -> None:
    resolved = resolve_split_config(
        defaults={"split": {"mode": "cross_validation", "n_splits": 3}},
        train_config={"split": {"mode": "holdout", "test_size": 0.3}},
        tune_config={"split": {"mode": "cross_validation", "n_splits": 5}},
    )

    assert resolved["mode"] == "cross_validation"
    assert resolved["n_splits"] == 5
    assert resolved["test_size"] == 0.3


def test_phase9_resolve_finetune_config_returns_mapping() -> None:
    resolved = resolve_finetune_config(
        defaults={"finetune": {"epochs": 12, "freeze_layers": "backbone_last2"}}
    )

    assert resolved["epochs"] == 12
    assert resolved["freeze_layers"] == "backbone_last2"


def test_phase9_benchmark_bottleneck_summary_includes_ratios() -> None:
    summary = _hardware_bottleneck_summary(
        {
            "models": [
                {
                    "model": "logreg",
                    "train": {"avg_seconds": 0.02},
                    "predict": {"avg_seconds": 0.01},
                },
                {
                    "model": "mlp",
                    "train": {"avg_seconds": 0.60},
                    "predict": {"avg_seconds": 0.05},
                },
            ]
        }
    )

    assert "Slowest training model: mlp" in summary
    assert "Fastest training model: logreg" in summary
    assert "Training bottleneck ratio: 30.00x" in summary
    assert "Inference bottleneck ratio: 5.00x" in summary
