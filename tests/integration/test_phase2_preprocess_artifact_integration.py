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
        numeric_features=["feat_a", "feat_b"],
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

    assert first[["feat_a", "feat_b"]].equals(second[["feat_a", "feat_b"]])


@pytest.mark.integration
def test_transform_rejects_missing_feature_columns(variant_csv_path: str) -> None:
    df = load_dataset(variant_csv_path)
    processor = FoldPreprocessor(numeric_features=["feat_a", "feat_b"], per_gene=False)
    processor.fit(df)

    broken = df.drop(columns=["feat_b"])
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
        numeric_features=["feat_a", "feat_b"],
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

    assert "feat_a__is_missing" in first.columns
    assert "feat_b__is_missing" in first.columns
    assert first[["feat_a__is_missing", "feat_b__is_missing"]].equals(
        second[["feat_a__is_missing", "feat_b__is_missing"]]
    )
