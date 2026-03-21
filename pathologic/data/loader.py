"""Dataset loading, schema validation, and leakage-safe fold construction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV or Parquet dataset with actionable validation errors."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    elif suffix == ".parquet":
        try:
            frame = pd.read_parquet(dataset_path)
        except ImportError as exc:
            raise ValueError(
                "Parquet loading requires 'pyarrow' or 'fastparquet'. "
                "Install one of them to read .parquet files."
            ) from exc
    else:
        raise ValueError("Unsupported dataset format. Use .csv or .parquet files.")

    if frame.empty:
        raise ValueError("Loaded dataset is empty.")
        
    # MAC_OPTIMIZATION / PERFORMANCE: Downcast float64 to float32 to halve memory usage 
    # and increase computation speed (cache hits) without losing practical precision.
    float64_cols = frame.select_dtypes(include=['float64']).columns
    if len(float64_cols) > 0:
        frame[float64_cols] = frame[float64_cols].astype(np.float32)
        
    return frame


def validate_schema(
    df: pd.DataFrame,
    *,
    label_column: str = "label",
    gene_column: str = "gene_id",
    require_gene_column: bool = True,
    required_feature_columns: Sequence[str] | None = None,
) -> None:
    """Validate required schema columns and non-null constraints."""
    required = [label_column]
    if require_gene_column:
        required.append(gene_column)
    if required_feature_columns:
        required.extend(required_feature_columns)

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if df[label_column].isna().any():
        raise ValueError(f"Label column '{label_column}' contains null values.")
    if gene_column in df.columns and df[gene_column].isna().any():
        raise ValueError(f"Gene column '{gene_column}' contains null values.")


def _can_stratify(y: pd.Series, n_splits: int) -> bool:
    counts = y.value_counts()
    if counts.empty:
        return False
    return int(counts.min()) >= n_splits and y.nunique() > 1


def build_folds(
    df: pd.DataFrame,
    *,
    label_column: str = "label",
    gene_column: str = "gene_id",
    n_splits: int = 5,
    stratified: bool = True,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build folds with group leakage prevention when gene IDs are available."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    y = df[label_column]
    has_gene = gene_column in df.columns

    splitter: GroupKFold | StratifiedGroupKFold | StratifiedKFold | KFold
    if has_gene:
        groups = df[gene_column]
        if stratified and _can_stratify(y, n_splits):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            return list(splitter.split(df, y, groups=groups))

        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(df, y, groups=groups))

    if stratified and _can_stratify(y, n_splits):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(df, y))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(df))


def summarize_folds(
    df: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    label_column: str = "label",
    gene_column: str = "gene_id",
) -> list[dict[str, float | int]]:
    """Summarize split quality and class distribution per fold."""
    summaries: list[dict[str, float | int]] = []
    has_gene = gene_column in df.columns
    for fold_index, (train_idx, val_idx) in enumerate(folds):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_positive_rate = float(train_df[label_column].mean())
        val_positive_rate = float(val_df[label_column].mean())

        summary: dict[str, float | int] = {
            "fold": fold_index,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "train_positive_rate": train_positive_rate,
            "val_positive_rate": val_positive_rate,
        }
        if has_gene:
            shared_genes = set(train_df[gene_column]).intersection(set(val_df[gene_column]))
            summary["train_unique_genes"] = int(train_df[gene_column].nunique())
            summary["val_unique_genes"] = int(val_df[gene_column].nunique())
            summary["shared_genes"] = int(len(shared_genes))
        summaries.append(summary)

    return summaries


def build_holdout_split(
    df: pd.DataFrame,
    *,
    label_column: str = "label",
    gene_column: str = "gene_id",
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratified: bool = True,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Build train/validation/test split with group leakage prevention when possible."""
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("test_size must satisfy 0 < test_size < 1")
    if val_size <= 0.0 or val_size >= 1.0:
        raise ValueError("val_size must satisfy 0 < val_size < 1")
    if (test_size + val_size) >= 1.0:
        raise ValueError("test_size + val_size must be < 1")

    y = df[label_column]
    indices = np.arange(len(df))
    has_gene = gene_column in df.columns

    if has_gene:
        groups = df[gene_column]
        first_split = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_val_pos, test_pos = next(first_split.split(indices, y, groups=groups))
        train_val_idx = indices[train_val_pos]
        test_idx = indices[test_pos]

        relative_val_size = val_size / (1.0 - test_size)
        second_split = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_val_size,
            random_state=random_state,
        )
        second_groups = groups.iloc[train_val_idx]
        train_pos, val_pos = next(
            second_split.split(train_val_idx, y.iloc[train_val_idx], groups=second_groups)
        )
        train_idx = train_val_idx[train_pos]
        val_idx = train_val_idx[val_pos]
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    if stratified and _can_stratify(y, n_splits=2):
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        relative_val_size = val_size / (1.0 - test_size)
        y_train_val = y.iloc[train_val_idx]
        stratify_train_val = y_train_val if _can_stratify(y_train_val, n_splits=2) else None
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=stratify_train_val,
        )
        return {
            "train": np.asarray(train_idx, dtype=int),
            "val": np.asarray(val_idx, dtype=int),
            "test": np.asarray(test_idx, dtype=int),
        }

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=random_state,
        shuffle=True,
    )
    return {
        "train": np.asarray(train_idx, dtype=int),
        "val": np.asarray(val_idx, dtype=int),
        "test": np.asarray(test_idx, dtype=int),
    }


def summarize_holdout_split(
    df: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
    *,
    label_column: str = "label",
    gene_column: str = "gene_id",
) -> dict[str, float | int | str]:
    """Summarize holdout split quality and leakage indicators."""
    train_df = df.iloc[split_indices["train"]]
    val_df = df.iloc[split_indices["val"]]
    test_df = df.iloc[split_indices["test"]]

    summary: dict[str, float | int | str] = {
        "split_mode": "holdout",
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "train_positive_rate": float(train_df[label_column].mean()),
        "val_positive_rate": float(val_df[label_column].mean()),
        "test_positive_rate": float(test_df[label_column].mean()),
    }

    if gene_column in df.columns:
        train_genes = set(train_df[gene_column])
        val_genes = set(val_df[gene_column])
        test_genes = set(test_df[gene_column])
        summary["train_val_shared_genes"] = int(len(train_genes.intersection(val_genes)))
        summary["train_test_shared_genes"] = int(len(train_genes.intersection(test_genes)))
        summary["val_test_shared_genes"] = int(len(val_genes.intersection(test_genes)))

    return summary
