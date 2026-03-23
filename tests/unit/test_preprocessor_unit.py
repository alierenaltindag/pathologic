"""Unit tests for fold-aware preprocessing behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pathologic.data.preprocessor import FoldPreprocessor


def test_preprocessor_fit_transform_without_leakage(variant_frame: pd.DataFrame) -> None:
    train_df = variant_frame.iloc[:6].copy()
    val_df = variant_frame.iloc[6:].copy()

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        gene_column="gene_id",
        impute_strategy="median",
        scaler="standard",
        per_gene=False,
    )
    train_processed = processor.fit_transform(train_df)
    val_processed = processor.transform(val_df)

    assert train_processed[["revel_score", "cadd_phred"]].isna().sum().sum() == 0
    assert val_processed[["revel_score", "cadd_phred"]].isna().sum().sum() == 0

    # Train fold should be centered by train-only statistics.
    assert abs(float(train_processed["revel_score"].mean())) < 1e-9


def test_per_gene_preprocessor_supports_unseen_genes(variant_frame: pd.DataFrame) -> None:
    train_df = variant_frame.iloc[:7].copy()
    val_df = variant_frame.iloc[7:].copy()
    val_df.loc[val_df.index[0], "gene_id"] = "G_UNSEEN"

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        gene_column="gene_id",
        scaler="minmax",
        per_gene=True,
    )
    processor.fit(train_df)
    transformed = processor.transform(val_df)

    assert transformed[["revel_score", "cadd_phred"]].isna().sum().sum() == 0


def test_per_gene_subset_normalization_with_unseen_gene(variant_frame: pd.DataFrame) -> None:
    train_df = variant_frame.iloc[:8].copy()
    val_df = variant_frame.iloc[8:].copy()

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        gene_column="gene_id",
        scaler="standard",
        per_gene=True,
        per_gene_features=["revel_score"],
    )
    processor.fit(train_df)
    transformed = processor.transform(val_df)

    assert transformed[["revel_score", "cadd_phred"]].isna().sum().sum() == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"per_gene_features": ["feat_x"]},
            "per_gene_features must be a subset of numeric_features",
        ),
        (
            {"scaler_features": ["feat_x"]},
            "scaler_features must be a subset of numeric_features",
        ),
        (
            {"missing_indicator_features": ["feat_x"]},
            "missing_indicator_features must be a subset of numeric_features",
        ),
    ],
)
def test_invalid_subset_features_raise_value_error(
    kwargs: dict[str, list[str]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FoldPreprocessor(
            numeric_features=["revel_score", "cadd_phred"],
            per_gene=True,
            **kwargs,
        )


def test_artifact_save_load_roundtrip(tmp_path: Path, variant_frame: pd.DataFrame) -> None:
    train_df = variant_frame.iloc[:6].copy()
    val_df = variant_frame.iloc[6:].copy()

    processor = FoldPreprocessor(numeric_features=["revel_score", "cadd_phred"], per_gene=True)
    processor.fit(train_df)

    artifact = tmp_path / "preprocessor.pkl"
    processor.save_artifacts(str(artifact))

    loaded = FoldPreprocessor.load_artifacts(str(artifact))
    original_out = processor.transform(val_df)
    loaded_out = loaded.transform(val_df)

    assert original_out[["revel_score", "cadd_phred"]].equals(loaded_out[["revel_score", "cadd_phred"]])


def test_impute_none_keeps_missing_when_no_scaling(variant_frame: pd.DataFrame) -> None:
    train_df = variant_frame.iloc[:6].copy()

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        impute_strategy="none",
        scaler="standard",
        per_gene=False,
        scaler_features=[],
    )
    transformed = processor.fit_transform(train_df)

    assert transformed["revel_score"].isna().sum() > 0
    assert transformed["cadd_phred"].isna().sum() > 0


def test_missing_value_policy_none_skips_imputation(variant_frame: pd.DataFrame) -> None:
    train_df = variant_frame.iloc[:6].copy()

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        missing_value_policy="none",
        impute_strategy="median",
        scaler="standard",
        per_gene=False,
        scaler_features=[],
    )
    transformed = processor.fit_transform(train_df)

    assert transformed["revel_score"].isna().sum() > 0
    assert transformed["cadd_phred"].isna().sum() > 0


def test_missing_value_policy_none_disables_scaling_and_per_gene_automatically(
    variant_frame: pd.DataFrame,
) -> None:
    train_df = variant_frame.iloc[:6].copy()

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        missing_value_policy="none",
        impute_strategy="median",
        scaler="standard",
        per_gene=True,
    )

    transformed = processor.fit_transform(train_df)

    assert transformed["revel_score"].isna().sum() > 0
    assert transformed["cadd_phred"].isna().sum() > 0


def test_impute_none_with_scaling_and_missing_raises(variant_frame: pd.DataFrame) -> None:
    train_df = variant_frame.iloc[:6].copy()

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        impute_strategy="none",
        scaler="standard",
        per_gene=False,
    )

    with pytest.raises(ValueError, match="Missing values detected"):
        processor.fit(train_df)


def test_missing_indicators_added_and_persisted(
    tmp_path: Path,
    variant_frame: pd.DataFrame,
) -> None:
    train_df = variant_frame.iloc[:6].copy()
    val_df = variant_frame.iloc[6:].copy()

    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        impute_strategy="median",
        add_missing_indicators=True,
    )
    processor.fit(train_df)
    transformed = processor.transform(val_df)

    assert "revel_score__is_missing" in transformed.columns
    assert "cadd_phred__is_missing" in transformed.columns

    artifact = tmp_path / "preprocessor_missing_flags.pkl"
    processor.save_artifacts(str(artifact))
    loaded = FoldPreprocessor.load_artifacts(str(artifact))
    loaded_transformed = loaded.transform(val_df)

    assert transformed[["revel_score__is_missing", "cadd_phred__is_missing"]].equals(
        loaded_transformed[["revel_score__is_missing", "cadd_phred__is_missing"]]
    )


def test_drop_rows_policy_removes_rows_with_missing_numeric_values(
    variant_frame: pd.DataFrame,
) -> None:
    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        missing_value_policy="drop_rows",
        impute_strategy="none",
        scaler="standard",
        per_gene=False,
    )

    transformed = processor.fit_transform(variant_frame)

    assert len(transformed) < len(variant_frame)
    assert transformed[["revel_score", "cadd_phred"]].isna().sum().sum() == 0


def test_missing_indicator_feature_allow_list_is_respected(
    variant_frame: pd.DataFrame,
) -> None:
    processor = FoldPreprocessor(
        numeric_features=["revel_score", "cadd_phred"],
        add_missing_indicators=True,
        missing_indicator_features=["revel_score"],
    )

    transformed = processor.fit_transform(variant_frame)

    assert "revel_score__is_missing" in transformed.columns
    assert "cadd_phred__is_missing" not in transformed.columns


def test_invalid_missing_value_policy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing_value_policy must be one of"):
        FoldPreprocessor(
            numeric_features=["revel_score", "cadd_phred"],
            missing_value_policy="invalid_policy",  # type: ignore[arg-type]
        )


