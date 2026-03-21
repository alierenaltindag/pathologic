from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pathologic.explain.error_analysis import MultiDimensionalErrorAnalyzer


def _sample_error_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_id": [
                "G1",
                "G1",
                "G2",
                "G2",
                "G3",
                "G3",
                "G4",
                "G4",
                "G5",
                "G5",
                "G6",
                "G6",
            ],
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "feature__REVEL_Score": [0.9, 0.2, 0.7, 0.4, 0.8, 0.1, 0.75, 0.35, 0.6, 0.3, 0.85, 0.15],
            "feature__cadd.phred": [25.0, 8.0, 23.0, 10.0, 26.0, 7.0, 24.0, 9.0, 20.0, 11.0, 27.0, 6.0],
            "feature__gnomAD_AF": [0.0, 0.03, 0.001, 0.04, 0.0, 0.05, 0.002, 0.02, 0.005, 0.03, 0.0, 0.04],
            "feature__Hyd_Delta": [1.2, -0.4, 0.8, -0.3, 1.0, -0.6, 0.9, -0.2, 0.7, -0.5, 1.1, -0.7],
            "feature__MW_Delta": [15.0, -8.0, 12.0, -6.0, 14.0, -9.0, 13.0, -5.0, 11.0, -7.0, 16.0, -10.0],
            "feature__AA_Position": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            "Veri_Kaynagi_Paneli": [
                "P1",
                "P1",
                "P2",
                "P2",
                "P3",
                "P3",
                "P1",
                "P1",
                "P2",
                "P2",
                "P3",
                "P3",
            ],
        }
    )


def test_error_analysis_excludes_aa_position_and_generates_summary(tmp_path: Path) -> None:
    dataset = _sample_error_dataset()
    y_true = dataset["label"].to_numpy(dtype=int)
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int)
    y_score = np.array([0.8, 0.7, 0.75, 0.2, 0.3, 0.1, 0.77, 0.66, 0.72, 0.22, 0.25, 0.15])

    analyzer = MultiDimensionalErrorAnalyzer(random_state=42)
    result = analyzer.analyze_candidate(
        candidate_name="xgboost",
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        dataset=dataset,
        output_dir=tmp_path,
        detailed=False,
    )

    assert result.status == "ok"
    assert "feature__AA_Position" not in result.summary["numeric_features_used"]
    assert "error_analysis_summary_json" in result.artifacts


def test_error_analysis_returns_cluster_summary_when_errors_exist(tmp_path: Path) -> None:
    dataset = _sample_error_dataset()
    y_true = dataset["label"].to_numpy(dtype=int)
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int)
    y_score = np.linspace(0.15, 0.85, num=len(dataset))

    analyzer = MultiDimensionalErrorAnalyzer(random_state=7)
    result = analyzer.analyze_candidate(
        candidate_name="catboost",
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        dataset=dataset,
        output_dir=tmp_path,
        detailed=False,
    )

    clustering = result.summary.get("clustering", {})
    assert result.status == "ok"
    assert isinstance(clustering, dict)
    assert clustering.get("status") in {"ok", "skipped"}
    if clustering.get("status") == "ok":
        assert "kmeans_profiles" in clustering
        assert "dbscan_profiles" in clustering
        assert "kmeans_cluster_profiles_csv" in result.artifacts
