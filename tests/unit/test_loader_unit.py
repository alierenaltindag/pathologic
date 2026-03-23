"""Unit tests for data loading, schema validation, and fold construction."""

from __future__ import annotations

import pandas as pd
import pytest

from pathologic.data.loader import (
    build_folds,
    build_holdout_split,
    summarize_folds,
    summarize_holdout_split,
    validate_schema,
)


def test_validate_schema_raises_on_missing_required_columns(variant_frame: pd.DataFrame) -> None:
    broken = variant_frame.drop(columns=["gene_id"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(
            broken,
            label_column="label",
            gene_column="gene_id",
            required_feature_columns=["revel_score", "cadd_phred"],
        )


def test_validate_schema_accepts_required_columns(variant_frame: pd.DataFrame) -> None:
    validate_schema(
        variant_frame,
        label_column="label",
        gene_column="gene_id",
        required_feature_columns=["revel_score", "cadd_phred"],
    )


def test_build_folds_allows_gene_overlap_by_default(variant_frame: pd.DataFrame) -> None:
    folds = build_folds(
        variant_frame,
        label_column="label",
        gene_column="gene_id",
        n_splits=3,
        stratified=True,
        random_state=42,
    )

    overlap_counts: list[int] = []
    for train_idx, val_idx in folds:
        train_genes = set(variant_frame.iloc[train_idx]["gene_id"])
        val_genes = set(variant_frame.iloc[val_idx]["gene_id"])
        overlap_counts.append(len(train_genes.intersection(val_genes)))
    assert any(count > 0 for count in overlap_counts)


def test_build_folds_prevents_gene_overlap_when_disabled(variant_frame: pd.DataFrame) -> None:
    folds = build_folds(
        variant_frame,
        label_column="label",
        gene_column="gene_id",
        n_splits=3,
        stratified=True,
        allow_same_gene_overlap=False,
        random_state=42,
    )

    for train_idx, val_idx in folds:
        train_genes = set(variant_frame.iloc[train_idx]["gene_id"])
        val_genes = set(variant_frame.iloc[val_idx]["gene_id"])
        assert train_genes.isdisjoint(val_genes)


def test_summarize_folds_reports_distribution(variant_frame: pd.DataFrame) -> None:
    folds = build_folds(
        variant_frame,
        label_column="label",
        gene_column="gene_id",
        n_splits=3,
        stratified=True,
        random_state=42,
    )
    summary = summarize_folds(variant_frame, folds, label_column="label", gene_column="gene_id")

    assert len(summary) == 3
    assert all(int(item["shared_genes"]) >= 0 for item in summary)
    assert all(0.0 <= float(item["train_positive_rate"]) <= 1.0 for item in summary)
    assert all(0.0 <= float(item["val_positive_rate"]) <= 1.0 for item in summary)


def test_build_holdout_split_prevents_gene_overlap(variant_frame: pd.DataFrame) -> None:
    split_indices = build_holdout_split(
        variant_frame,
        label_column="label",
        gene_column="gene_id",
        test_size=0.2,
        val_size=0.2,
        stratified=True,
        allow_same_gene_overlap=False,
        random_state=42,
    )

    summary = summarize_holdout_split(
        variant_frame,
        split_indices,
        label_column="label",
        gene_column="gene_id",
    )
    assert int(summary["train_val_shared_genes"]) == 0
    assert int(summary["train_test_shared_genes"]) == 0
    assert int(summary["val_test_shared_genes"]) == 0
    assert int(summary["train_size"]) + int(summary["val_size"]) + int(summary["test_size"]) == len(
        variant_frame
    )


def test_build_holdout_split_rejects_invalid_ratios(variant_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"test_size \+ val_size must be < 1"):
        build_holdout_split(
            variant_frame,
            label_column="label",
            gene_column="gene_id",
            test_size=0.6,
            val_size=0.4,
            random_state=42,
        )


def test_build_folds_warns_when_gene_column_missing(variant_frame: pd.DataFrame) -> None:
    gene_missing = variant_frame.drop(columns=["gene_id"])

    with pytest.warns(UserWarning, match="falling back to non-grouped folds"):
        folds = build_folds(
            gene_missing,
            label_column="label",
            gene_column="gene_id",
            n_splits=3,
            stratified=True,
            random_state=42,
        )

    assert len(folds) == 3


def test_build_holdout_warns_when_gene_column_missing(variant_frame: pd.DataFrame) -> None:
    gene_missing = variant_frame.drop(columns=["gene_id"])

    with pytest.warns(UserWarning, match="falling back to non-grouped holdout split"):
        split_indices = build_holdout_split(
            gene_missing,
            label_column="label",
            gene_column="gene_id",
            test_size=0.2,
            val_size=0.2,
            stratified=True,
            random_state=42,
        )

    assert set(split_indices) == {"train", "val", "test"}

