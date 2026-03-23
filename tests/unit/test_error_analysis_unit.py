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
            "feature__gnomAD_log": [-8.0, -1.52, -3.0, -1.39, -8.0, -1.3, -2.7, -1.7, -2.3, -1.52, -8.0, -1.39],
            "feature__gnomAD_is_zero": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
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
    assert "feature__gnomAD_log" in result.summary["numeric_features_used"]
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


def test_error_analysis_includes_panel_performance_counts(tmp_path: Path) -> None:
    dataset = _sample_error_dataset()
    y_true = dataset["label"].to_numpy(dtype=int)
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int)
    y_score = np.linspace(0.2, 0.9, num=len(dataset))

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

    panel_summary = result.summary.get("panel_performance")
    assert isinstance(panel_summary, dict)
    assert panel_summary.get("status") == "ok"
    assert panel_summary.get("panel_column") == "Veri_Kaynagi_Paneli"
    assert panel_summary.get("panel_count") == 3
    assert panel_summary.get("total_samples") == len(dataset)
    assert panel_summary.get("total_correct_predictions") == 8

    rows = panel_summary.get("rows")
    assert isinstance(rows, list)
    by_panel = {str(item["panel"]): item for item in rows if isinstance(item, dict)}
    assert by_panel["P1"]["total_samples"] == 4
    assert by_panel["P1"]["correct_predictions"] == 2
    assert by_panel["P2"]["total_samples"] == 4
    assert by_panel["P2"]["correct_predictions"] == 4
    assert by_panel["P3"]["total_samples"] == 4
    assert by_panel["P3"]["correct_predictions"] == 2

    panel_csv = result.artifacts.get("panel_performance_csv")
    assert isinstance(panel_csv, str)
    assert Path(panel_csv).exists()

def test_pattern_concentration_analysis_logic() -> None:
    dataset = pd.DataFrame({
        "feature__gnomAD_log": [-5.0, -2.3, -0.3, -6.5],
        "feature__REVEL_Score": [0.9, 0.1, 0.7, 0.2],
        "feature__cadd.phred": [10.0, 30.0, 5.0, 35.0],
        "feature__Charge_Change": [1, -1, 0, 0],
        "feature__Polarity_Change": [1, -1, 0, 1],
        "feature__Hyd_Delta": [1.5, -2, 0.5, 0],
        "label": [1, 0, 1, 0]
    })
    
    # 4 errors: 2 FPs, 2 FNs
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1]) # All wrong
    y_score = np.array([0.2, 0.8, 0.2, 0.8])
    
    analyzer = MultiDimensionalErrorAnalyzer(random_state=42)
    error_frame = analyzer._build_error_frame(
        y_true=y_true, y_pred=y_pred, y_score=y_score,
        dataset=dataset, numeric_columns=analyzer._resolve_numeric_columns(dataset)
    )
    
    patterns = analyzer._analyze_pattern_concentration(error_frame)
    
    # Check relative frequency bins for transformed AF values
    assert "population_frequency" in patterns
    freqs = patterns["population_frequency"]
    assert "Low (Relative AF)" in freqs
    assert "Mid (Relative AF)" in freqs
    assert "High (Relative AF)" in freqs
    total_from_bins = (
        freqs["Low (Relative AF)"]["total"]
        + freqs["Mid (Relative AF)"]["total"]
        + freqs["High (Relative AF)"]["total"]
        + freqs.get("Missing/Invalid", {}).get("total", 0)
    )
    assert int(total_from_bins) == int(len(error_frame))
    
    # Check conflicts
    # Row 0: REVEL 0.9, CADD 10 -> revel_high_cadd_low
    # Row 1: REVEL 0.1, CADD 30 -> revel_low_cadd_high
    # Row 2: REVEL 0.7, CADD 5 -> revel_high_cadd_low
    # Row 3: REVEL 0.2, CADD 35 -> revel_low_cadd_high
    assert patterns["insilico_conflicts"]["revel_high_cadd_low"]["total"] == 2
    assert patterns["insilico_conflicts"]["revel_low_cadd_high"]["total"] == 2
    
    # Check biochemical
    bio = patterns["biochemical_patterns"]
    assert bio["charge"]["Gain of Positive"]["total"] == 1
    assert bio["charge"]["Loss of Positive"]["total"] == 1
    assert bio["polarity"]["Increase"]["total"] == 2
    assert bio["hydropathy"]["More Hydrophobic"]["total"] == 1
    assert bio["hydropathy"]["More Hydrophilic"]["total"] == 1


def test_population_frequency_handles_scaled_inputs_without_dropping_rows() -> None:
    dataset = pd.DataFrame(
        {
            "feature__gnomAD_log": [-1.3, -0.8, -0.2, 0.1, 0.5, 1.2],
            "feature__REVEL_Score": [0.2, 0.4, 0.1, 0.8, 0.3, 0.7],
            "feature__cadd.phred": [12.0, 14.0, 8.0, 22.0, 18.0, 25.0],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    y_pred = np.array([1, 0, 1, 0, 1, 0], dtype=int)
    y_score = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1], dtype=float)

    analyzer = MultiDimensionalErrorAnalyzer(random_state=42)
    error_frame = analyzer._build_error_frame(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        dataset=dataset,
        numeric_columns=analyzer._resolve_numeric_columns(dataset),
    )
    patterns = analyzer._analyze_pattern_concentration(error_frame)

    freq = patterns["population_frequency"]
    total_from_bins = sum(int(stats["total"]) for stats in freq.values())
    assert total_from_bins == int(len(error_frame))
    assert "Low (Relative AF)" in freq
    assert "Mid (Relative AF)" in freq
    assert "High (Relative AF)" in freq
