"""Shared pytest fixtures for PathoLogic tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def fake_dataset_path(tmp_path: Path) -> str:
    """Provide a fake CSV path for skeleton API tests."""
    path = tmp_path / "dummy_variants.csv"
    path.write_text(
        (
            "variant_id,gene_id,label,feat_a,feat_b\n"
            "1,GENE1,1,0.1,1.0\n"
            "2,GENE1,0,0.2,1.1\n"
            "3,GENE2,1,0.3,0.9\n"
            "4,GENE2,0,0.4,1.2\n"
            "5,GENE3,1,0.5,1.3\n"
            "6,GENE3,0,0.6,0.8\n"
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def variant_frame() -> pd.DataFrame:
    """Representative dataset containing groups, imbalance, and missing values."""
    return pd.DataFrame(
        {
            "variant_id": [
                "v1",
                "v2",
                "v3",
                "v4",
                "v5",
                "v6",
                "v7",
                "v8",
                "v9",
            ],
            "gene_id": ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4", "G5"],
            "label": [1, 0, 1, 0, 1, 0, 0, 0, 1],
            "feat_a": [0.1, None, 0.3, 0.5, 0.2, 0.6, 0.9, 0.8, 0.4],
            "feat_b": [1.0, 1.1, None, 0.9, 1.3, 1.2, 0.7, 0.6, 1.5],
        }
    )


@pytest.fixture
def variant_csv_path(tmp_path: Path, variant_frame: pd.DataFrame) -> str:
    """Persist a representative variant frame as CSV for loader integration tests."""
    file_path = tmp_path / "variants.csv"
    variant_frame.to_csv(file_path, index=False)
    return str(file_path)
