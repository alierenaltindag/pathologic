"""Integration tests for train-to-validation artifact consistency."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathologic.data.loader import build_folds, load_dataset
from pathologic.data.preprocessor import FoldPreprocessor


@pytest.mark.integration
def test_train_val_preprocess_artifact_consistency(tmp_path: Path, variant_csv_path: str) -> None:
    df = load_dataset(variant_csv_path)
    folds = build_folds(df, n_splits=3, stratified=True, random_state=42)
    train_idx, val_idx = folds[0]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        gene_column="gene_id",
        impute_strategy="median",
        scaler="standard",
        per_gene=True,
    )
    processor.fit(train_df)

    artifact_path = tmp_path / "phase2_preprocessor.pkl"
    processor.save_artifacts(str(artifact_path))

    loaded = FoldPreprocessor.load_artifacts(str(artifact_path))

    first = processor.transform(val_df)
    second = loaded.transform(val_df)

    assert first[["revel_score", "cadd_phred"]].equals(second[["revel_score", "cadd_phred"]])


@pytest.mark.integration
def test_transform_rejects_missing_feature_columns(variant_csv_path: str) -> None:
    df = load_dataset(variant_csv_path)
    processor = FoldPreprocessor(numeric_features=["revel_score", "cadd_phred"], per_gene=False)
    processor.fit(df)

    broken = df.drop(columns=["cadd_phred"])
    with pytest.raises(ValueError, match="Missing numeric feature columns"):
        processor.transform(broken)


@pytest.mark.integration
def test_missing_indicator_columns_survive_artifact_roundtrip(
    tmp_path: Path,
    variant_csv_path: str,
) -> None:
    df = load_dataset(variant_csv_path)
    train_df = df.iloc[:8].reset_index(drop=True)
    val_df = df.iloc[8:].reset_index(drop=True)

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        gene_column="gene_id",
        impute_strategy="median",
        scaler="standard",
        per_gene=False,
        add_missing_indicators=True,
    )
    processor.fit(train_df)

    artifact_path = tmp_path / "phase2_preprocessor_missing_flags.pkl"
    processor.save_artifacts(str(artifact_path))
    loaded = FoldPreprocessor.load_artifacts(str(artifact_path))

    first = processor.transform(val_df)
    second = loaded.transform(val_df)

    assert "revel_score__is_missing" in first.columns
    assert "cadd_phred__is_missing" in first.columns
    assert first[["revel_score__is_missing", "cadd_phred__is_missing"]].equals(
        second[["revel_score__is_missing", "cadd_phred__is_missing"]]
    )


