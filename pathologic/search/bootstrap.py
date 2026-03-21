"""Run bootstrap helpers for search orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd

from pathologic.data.loader import build_holdout_split, load_dataset, summarize_holdout_split
from pathologic.search import candidate as _search_candidate
from pathologic.search import data as _search_data
from pathologic.search import hpo_nas as _search_hpo_nas
from pathologic.search.logging import build_run_logger
from pathologic.search.spec import BudgetProfile
from pathologic.search.utils import parse_model_pool


@dataclass(frozen=True)
class SearchRunBootstrapContext:
    started: float
    budget: BudgetProfile
    quiet_inner_search: bool
    search_defaults: dict[str, Any]
    cv_splits: int
    run_dir: Path
    run_logger: logging.Logger
    log_path: Path
    prepared_csv: str
    feature_columns: list[str]
    prep_stats: dict[str, Any]
    split_summary: dict[str, Any]
    outer_base_train_df: pd.DataFrame
    outer_calibration_df: pd.DataFrame
    outer_test_df: pd.DataFrame
    outer_train_csv: Path
    outer_calibration_csv: Path
    outer_test_csv: Path
    candidates: list[Any]
    x_train_nas: np.ndarray
    y_train_nas: np.ndarray
    x_val_nas: np.ndarray
    y_val_nas: np.ndarray


def _ensure_no_holdout_leakage(summary: dict[str, Any]) -> None:
    for key in (
        "train_val_shared_genes",
        "train_test_shared_genes",
        "val_test_shared_genes",
    ):
        if key in summary and int(summary[key]) != 0:
            raise RuntimeError(f"Leakage detected: {key}={summary[key]}")


def bootstrap_search_run(args: argparse.Namespace, *, budget: BudgetProfile) -> SearchRunBootstrapContext:
    search_defaults = _search_data.resolve_search_defaults_from_defaults()
    cv_splits = int(args.cv_splits) if args.cv_splits is not None else budget.cv_splits
    if cv_splits < 2:
        raise ValueError("cv_splits must be >= 2 for cross-validation.")
    quiet_inner_search = not bool(args.verbose_inner_search)

    started = monotonic()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"search_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_logger, log_path = build_run_logger(run_dir)

    excluded_columns = _search_data.resolve_excluded_columns_from_defaults()
    error_analysis_columns = _search_data.resolve_error_analysis_columns_from_defaults()

    prepared_path = run_dir / "prepared_dataset.csv"
    prepared_csv, feature_columns, prep_stats = _search_data.prepare_dataset_for_pathologic(
        args.data_csv,
        str(prepared_path),
        excluded_columns=excluded_columns,
        error_analysis_columns=error_analysis_columns,
    )

    prepared_df = load_dataset(prepared_csv)
    split_indices = build_holdout_split(
        prepared_df,
        label_column="label",
        gene_column="gene_id",
        test_size=float(args.outer_test_size),
        val_size=float(args.outer_val_size),
        stratified=True,
        random_state=int(args.seed),
    )
    split_summary = summarize_holdout_split(
        prepared_df,
        split_indices,
        label_column="label",
        gene_column="gene_id",
    )
    _ensure_no_holdout_leakage(split_summary)

    outer_base_train_idx = split_indices["train"]
    outer_calibration_idx = split_indices["val"]
    outer_test_idx = split_indices["test"]
    outer_base_train_df = prepared_df.iloc[outer_base_train_idx].reset_index(drop=True)
    outer_calibration_df = prepared_df.iloc[outer_calibration_idx].reset_index(drop=True)
    outer_test_df = prepared_df.iloc[outer_test_idx].reset_index(drop=True)

    outer_train_csv = run_dir / "outer_train.csv"
    outer_calibration_csv = run_dir / "outer_calibration.csv"
    outer_test_csv = run_dir / "outer_test.csv"
    outer_base_train_df.to_csv(outer_train_csv, index=False)
    outer_calibration_df.to_csv(outer_calibration_csv, index=False)
    outer_test_df.to_csv(outer_test_csv, index=False)

    include_models = parse_model_pool(args.model_pool)
    if include_models is None:
        include_models = parse_model_pool(args.models)
    regularization_models = parse_model_pool(getattr(args, "regularization_models", None))
    exclude_models = (
        [item.strip() for item in args.exclude_models.split(",") if item.strip()]
        if args.exclude_models
        else None
    )

    candidates = _search_candidate.build_candidate_specs(
        include_models=include_models,
        exclude_models=exclude_models,
        include_hybrids=not args.disable_hybrids,
        max_candidates=args.max_candidates,
        hybrid_tune_strategy_and_params=bool(args.hybrid_tune_strategy_and_params),
        hybrid_tuning_search_space=(
            _search_candidate.build_hybrid_strategy_tuning_search_space(
                budget_profile=str(args.budget_profile),
                search_defaults=search_defaults,
            )
            if bool(args.hybrid_tune_strategy_and_params)
            else None
        ),
        regularization_profile=str(getattr(args, "regularization_profile", "auto")),
        regularization_models=regularization_models,
    )
    if not candidates:
        raise RuntimeError("No candidates to evaluate after include/exclude filters.")

    x_train_nas, y_train_nas, x_val_nas, y_val_nas = _search_hpo_nas.build_nas_arrays(
        train_df=outer_base_train_df,
        feature_columns=feature_columns,
        seed=int(args.seed),
    )

    return SearchRunBootstrapContext(
        started=started,
        budget=budget,
        quiet_inner_search=quiet_inner_search,
        search_defaults=search_defaults,
        cv_splits=cv_splits,
        run_dir=run_dir,
        run_logger=run_logger,
        log_path=log_path,
        prepared_csv=prepared_csv,
        feature_columns=feature_columns,
        prep_stats=prep_stats,
        split_summary=split_summary,
        outer_base_train_df=outer_base_train_df,
        outer_calibration_df=outer_calibration_df,
        outer_test_df=outer_test_df,
        outer_train_csv=outer_train_csv,
        outer_calibration_csv=outer_calibration_csv,
        outer_test_csv=outer_test_csv,
        candidates=candidates,
        x_train_nas=x_train_nas,
        y_train_nas=y_train_nas,
        x_val_nas=x_val_nas,
        y_val_nas=y_val_nas,
    )
