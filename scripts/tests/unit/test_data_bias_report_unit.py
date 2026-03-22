from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analyze_data_bias import (
    _compute_group_bias,
    _compute_imbalance_stats,
    _detect_label_column,
    _parse_group_columns,
    generate_data_bias_report,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Target": [1, 1, 1, 0, 0, 0, 1, 0],
            "Gene(s)": ["G1", "G1", "G2", "G2", "G2", "G3", "G3", "G3"],
            "Veri_Kaynagi_Paneli": ["P1", "P1", "P1", "P2", "P2", "P3", "P3", "P3"],
            "feature_x": [0.1, 0.2, None, 0.3, None, 0.4, 0.5, None],
        }
    )


def test_detect_label_column_prefers_known_names() -> None:
    df = _sample_df()
    assert _detect_label_column(df) == "Target"


def test_parse_group_columns_handles_spaces() -> None:
    cols = _parse_group_columns("Gene(s), Veri_Kaynagi_Paneli ,Ref_AA")
    assert cols == ["Gene(s)", "Veri_Kaynagi_Paneli", "Ref_AA"]


def test_compute_imbalance_stats_has_expected_keys() -> None:
    df = _sample_df()
    stats = _compute_imbalance_stats(df["Target"])
    assert stats["total_samples"] == 8
    assert stats["class_0_count"] == 4
    assert stats["class_1_count"] == 4
    assert stats["imbalance_ratio"] == 1.0


def test_compute_group_bias_returns_gap_metrics() -> None:
    df = _sample_df()
    out = _compute_group_bias(
        df,
        label_column="Target",
        group_columns=["Gene(s)", "Veri_Kaynagi_Paneli"],
        min_group_size=1,
        top_k_groups=10,
    )
    assert not out.empty
    assert "abs_rate_gap" in out.columns
    assert "risk_ratio" in out.columns


def test_generate_data_bias_report_writes_html(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    html_path = tmp_path / "report.html"
    _sample_df().to_csv(csv_path, index=False)

    payload = generate_data_bias_report(
        csv_path=str(csv_path),
        output_html=str(html_path),
        label_column="Target",
        group_columns=["Gene(s)", "Veri_Kaynagi_Paneli"],
        min_group_size=1,
        top_k_groups=10,
    )

    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "PathoLogic Data Bias Report" in content
    assert payload["label_column"] == "Target"
    assert payload["rows"] == 8
