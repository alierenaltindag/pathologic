from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pathologic.search.artifacts import compute_candidate_compute_cost_artifacts
from pathologic.utils.compute_cost import benchmark_inference_latency


class _DummyPreprocessor:
    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset


class _DummyTrainedModel:
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        arr = x.to_numpy(dtype=float)
        logits = arr.sum(axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


class _NumpyOnlyTrainedModel:
    def predict_proba(self, x: object) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise TypeError("expects numpy array")
        logits = x.sum(axis=1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


class _DummyModel:
    def __init__(self) -> None:
        self._preprocessor = _DummyPreprocessor()
        self._trained_model = _DummyTrainedModel()
        self._feature_columns = ["feature__a", "feature__b"]


class _NumpyOnlyModel:
    def __init__(self) -> None:
        self._preprocessor = _DummyPreprocessor()
        self._trained_model = _NumpyOnlyTrainedModel()
        self._feature_columns = ["feature__a", "feature__b"]


def test_compute_cost_artifacts_write_json_and_html(tmp_path: Path) -> None:
    payload = {
        "status": "enabled",
        "system": {"os": {"system": "Windows"}},
        "frameworks": {"numpy": "2.x"},
        "training": {"train_total_seconds": 1.2},
        "inference": {"single_sample_ms": 0.3},
        "reproducibility": {"seed": 42},
    }

    result = compute_candidate_compute_cost_artifacts(
        run_dir=tmp_path,
        candidate_name="lightgbm",
        payload=payload,
    )

    artifacts = result.get("artifacts")
    assert isinstance(artifacts, dict)
    assert Path(str(artifacts["compute_cost_report_json"])).exists()
    assert Path(str(artifacts["compute_cost_report_html"])).exists()


def test_benchmark_inference_latency_returns_positive_metrics() -> None:
    dataset = pd.DataFrame(
        {
            "feature__a": [0.1, 0.2, 0.3, 0.4],
            "feature__b": [0.5, 0.6, 0.7, 0.8],
            "label": [0, 1, 0, 1],
        }
    )
    model = _DummyModel()

    result = benchmark_inference_latency(
        model=model,
        dataset=dataset,
        feature_columns=["feature__a", "feature__b"],
        label_column="label",
        single_runs=3,
        batch_runs=3,
        warmup_runs=1,
        batch_size=3,
    )

    assert result.single_sample_ms > 0.0
    assert result.batch_total_ms > 0.0
    assert result.batch_per_sample_ms > 0.0
    assert result.batch_size == 3
    assert result.full_dataset_size == 4


def test_benchmark_inference_latency_supports_numpy_only_predictor() -> None:
    dataset = pd.DataFrame(
        {
            "feature__a": [0.1, 0.2, 0.3, 0.4],
            "feature__b": [0.5, 0.6, 0.7, 0.8],
            "label": [0, 1, 0, 1],
        }
    )
    model = _NumpyOnlyModel()

    result = benchmark_inference_latency(
        model=model,
        dataset=dataset,
        feature_columns=["feature__a", "feature__b"],
        label_column="label",
        single_runs=2,
        batch_runs=2,
        warmup_runs=1,
        batch_size=2,
    )

    assert result.single_sample_ms > 0.0
    assert result.batch_total_ms > 0.0
    assert result.full_dataset_size == 4
