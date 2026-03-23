"""Candidate-loop orchestration helpers for search workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd

from pathologic import PathoLogic
from pathologic.search import evaluation as _search_evaluation
from pathologic.search import hpo_nas as _search_hpo_nas
from pathologic.search import progress as _search_progress
from pathologic.search.logging import emit
from pathologic.search.spec import BudgetProfile, CandidateSpec
from pathologic.utils.progress import is_progress_enabled, step_progress


def run_candidate_search_loop(
    *,
    args: argparse.Namespace,
    candidates: list[CandidateSpec],
    budget: BudgetProfile,
    quiet_inner_search: bool,
    outer_train_csv: Path,
    outer_test_csv: Path,
    outer_test_df: pd.DataFrame,
    outer_calibration_df: pd.DataFrame,
    run_dir: Path,
    feature_columns: list[str],
    cv_splits: int,
    x_train_nas: np.ndarray,
    y_train_nas: np.ndarray,
    x_val_nas: np.ndarray,
    y_val_nas: np.ndarray,
    run_logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, PathoLogic]]:
    leaderboard: list[dict[str, Any]] = []
    successful_models: dict[str, PathoLogic] = {}

    show_candidate_progress = is_progress_enabled()
    with step_progress(
        total=len(candidates),
        desc="candidate search",
        enabled=show_candidate_progress,
        leave=True,
    ) as candidate_bar:
        for index, candidate in enumerate(candidates, start=1):
            step_start = monotonic()

            stage_order = _search_hpo_nas.candidate_stage_order(candidate)
            with step_progress(
                total=len(stage_order),
                desc=f"{candidate.name} stages",
                enabled=show_candidate_progress,
                leave=True,
            ) as stage_bar:
                tracker = _search_progress.CandidateProgressTracker(
                    index=index,
                    total_candidates=len(candidates),
                    candidate_name=candidate.name,
                    show_candidate_progress=show_candidate_progress,
                    step_start=step_start,
                    stage_order=stage_order,
                    stage_bar=stage_bar,
                    candidate_bar=candidate_bar,
                    run_logger=run_logger,
                )
                try:
                    if not show_candidate_progress:
                        emit(
                            f"[{index}/{len(candidates)}] Evaluating {candidate.name}",
                            color="cyan",
                            bold=True,
                            run_logger=run_logger,
                        )

                    row, model = _search_evaluation.evaluate_candidate(
                        candidate=candidate,
                        args=args,
                        budget=budget,
                        quiet_inner_search=quiet_inner_search,
                        outer_train_csv=outer_train_csv,
                        outer_test_csv=outer_test_csv,
                        outer_test_df=outer_test_df,
                        outer_calibration_df=outer_calibration_df,
                        run_dir=run_dir,
                        feature_columns=feature_columns,
                        cv_splits=cv_splits,
                        x_train_nas=x_train_nas,
                        y_train_nas=y_train_nas,
                        x_val_nas=x_val_nas,
                        y_val_nas=y_val_nas,
                        stage_update=tracker.update,
                        run_logger=run_logger,
                        step_start=step_start,
                    )
                    if model is not None:
                        successful_models[candidate.name] = model
                        tracker.update("candidate", state="done")
                finally:
                    tracker.close()

            leaderboard.append(row)
            candidate_bar.update(1)
            candidate_bar.set_postfix(
                model=candidate.name,
                step_s=f"{float(row.get('runtime_seconds', 0.0)):.1f}",
            )

    return leaderboard, successful_models
