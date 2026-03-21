"""Candidate evaluation helpers for search workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import monotonic
from typing import Any, Callable

import numpy as np
import pandas as pd

from pathologic import PathoLogic
from pathologic.search import artifacts as _search_artifacts
from pathologic.search import candidate as _search_candidate
from pathologic.search import explainability as _search_explainability
from pathologic.search import hpo_nas as _search_hpo_nas
from pathologic.search.logging import inner_search_runtime
from pathologic.search.spec import BudgetProfile, CandidateSpec
from pathologic.search.utils import parse_model_pool


StageUpdate = Callable[[str], None]


def evaluate_candidate(
    *,
    candidate: CandidateSpec,
    args: argparse.Namespace,
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
    stage_update: Callable[..., None],
    run_logger: logging.Logger,
    step_start: float,
) -> tuple[dict[str, Any], PathoLogic | None]:
    row: dict[str, Any] = {
        "candidate": candidate.name,
        "kind": candidate.kind,
        "members": list(candidate.members),
        "status": "ok",
    }

    if candidate.kind == "hybrid_pair":
        row["hybrid_config"] = _search_candidate.resolve_hybrid_config_for_report(
            candidate=candidate,
            args=args,
            selected_params=None,
        )

    try:
        model = _search_candidate.model_for_candidate_with_hybrid_config(candidate, args)
        model.defaults.setdefault("data", {})["required_features"] = list(feature_columns)
        model.defaults.setdefault("data", {})["label_column"] = "label"
        model.defaults.setdefault("data", {})["gene_column"] = "gene_id"

        hpo_result: dict[str, Any]
        try:
            hpo_total_trials = (
                int(args.n_trials) if args.n_trials is not None else int(budget.n_trials)
            )
            hpo_trial_state = {"done": 0}

            def _on_hpo_trial_complete(
                _trial: dict[str, Any],
                _state: dict[str, int] = hpo_trial_state,
                _total_trials: int = hpo_total_trials,
            ) -> None:
                _state["done"] += 1
                trial_score = _trial.get("score")
                score_text = (
                    f" score={float(trial_score):.4f}"
                    if isinstance(trial_score, (int, float))
                    else ""
                )
                stage_update(
                    "hpo",
                    state="progress",
                    detail=f"{_state['done']}/{_total_trials} trials{score_text}",
                )

            stage_update("hpo", state="start")
            with inner_search_runtime(
                quiet=quiet_inner_search,
                show_inner_progress=True,
                suppress_stdout=True,
                suppress_stderr=False,
            ):
                hpo_result = _search_hpo_nas.run_hpo(
                    model=model,
                    train_csv=str(outer_train_csv),
                    objective=args.objective,
                    tune_engine=args.tune_engine,
                    budget=budget,
                    cv_splits=cv_splits,
                    n_trials_override=args.n_trials,
                    on_trial_complete=_on_hpo_trial_complete,
                )
            stage_update("hpo", state="done", detail=str(hpo_result.get("status", "ok")))
        except Exception as exc:
            hpo_result = {"status": "failed", "reason": str(exc)}
            stage_update("hpo", state="failed", detail=str(exc))

        search_space = candidate.tuning_search_space
        if not search_space:
            cfg_space = model._resolve_model_config().get("tuning_search_space")  # noqa: SLF001
            if isinstance(cfg_space, dict):
                search_space = {
                    str(k): dict(v)
                    for k, v in cfg_space.items()
                    if isinstance(v, dict)
                }

        nas_result: dict[str, Any]
        try:
            regularization_models = parse_model_pool(
                getattr(args, "regularization_models", None)
            )
            optimize_regularization_in_nas = bool(
                getattr(args, "optimize_regularization_in_nas", False)
            )
            nas_search_space = (
                search_space
                if optimize_regularization_in_nas
                else _search_candidate.strip_regularization_search_space(
                    search_space=search_space,
                    members=tuple(candidate.members),
                    regularization_models=regularization_models,
                )
            )

            nas_total_candidates = (
                int(args.nas_candidates)
                if args.nas_candidates is not None
                else int(budget.nas_candidates)
            )
            nas_candidate_state = {"done": 0}

            def _on_nas_candidate_complete(
                _trial: dict[str, Any],
                _state: dict[str, int] = nas_candidate_state,
                _total_candidates: int = nas_total_candidates,
            ) -> None:
                _state["done"] += 1
                trial_score = _trial.get("score")
                score_text = (
                    f" score={float(trial_score):.4f}"
                    if isinstance(trial_score, (int, float))
                    else ""
                )
                stage_update(
                    "nas",
                    state="progress",
                    detail=(
                        f"{_state['done']}/{_total_candidates} candidates"
                        f"{score_text}"
                    ),
                )

            stage_update("nas", state="start")
            with inner_search_runtime(
                quiet=quiet_inner_search,
                show_inner_progress=True,
                suppress_stdout=True,
                suppress_stderr=False,
            ):
                nas_result = _search_hpo_nas.run_nas(
                    candidate=candidate,
                    seed=int(args.seed),
                    nas_strategy=args.nas_strategy,
                    budget=budget,
                    nas_candidates_override=args.nas_candidates,
                    search_space=nas_search_space,
                    base_model_params=(
                        dict(hpo_result.get("best_params", {}))
                        if isinstance(hpo_result.get("best_params"), dict)
                        else {}
                    ),
                    x_train=x_train_nas,
                    y_train=y_train_nas,
                    x_val=x_val_nas,
                    y_val=y_val_nas,
                    on_candidate_complete=_on_nas_candidate_complete,
                )
            stage_update("nas", state="done", detail=str(nas_result.get("status", "ok")))
        except Exception as exc:
            nas_result = {"status": "failed", "reason": str(exc)}
            stage_update("nas", state="failed", detail=str(exc))

        selected_params, selected_source = _search_hpo_nas.select_best_params(
            hpo_result,
            nas_result,
        )

        train_kwargs: dict[str, Any] = {}
        if selected_params:
            train_kwargs["model_params"] = selected_params

        stage_update("train", state="start")
        with inner_search_runtime(
            quiet=quiet_inner_search,
            show_inner_progress=False,
            suppress_stdout=True,
            suppress_stderr=True,
        ):
            model.train(str(outer_train_csv), **train_kwargs)
        stage_update("train", state="done")

        stage_update("evaluate", state="start")
        with inner_search_runtime(
            quiet=quiet_inner_search,
            show_inner_progress=False,
            suppress_stdout=True,
            suppress_stderr=True,
        ):
            eval_report = model.evaluate(str(outer_test_csv))
        stage_update("evaluate", state="done")
        metrics = eval_report.get("metrics", {}) if isinstance(eval_report, dict) else {}

        if not bool(args.disable_explainability):
            stage_update("explainability", state="start")
            try:
                with inner_search_runtime(
                    quiet=quiet_inner_search,
                    show_inner_progress=False,
                    suppress_stdout=True,
                    suppress_stderr=True,
                ):
                    explain_payload = _search_explainability.compute_candidate_explainability_artifacts(
                        model=model,
                        test_csv=str(outer_test_csv),
                        run_dir=run_dir,
                        candidate_name=candidate.name,
                        top_k_features=int(args.explain_top_k_features),
                        top_k_samples=int(args.explain_top_k_samples),
                        background_size=int(args.explain_background_size),
                        fp_top_k=int(args.explain_fp_top_k),
                        fp_min_negative_count=int(args.explain_fp_min_negative_count),
                    )
                stage_update("explainability", state="done")
            except Exception as exc:
                explain_payload = {
                    "status": "failed",
                    "reason": str(exc),
                }
                stage_update("explainability", state="failed", detail=str(exc))
        else:
            explain_payload = {
                "status": "skipped",
                "reason": "disabled_by_flag",
            }
            stage_update("explainability", state="done", detail="skipped")

        stage_update("calibration_fit", state="start")
        y_calibration, score_calibration = _search_artifacts.extract_scores_from_model(
            model=model,
            dataset=outer_calibration_df,
            feature_columns=feature_columns,
            label_column="label",
        )
        stage_update("calibration_fit", state="done")

        stage_update("calibration_eval", state="start")
        y_test, score_test = _search_artifacts.extract_scores_from_model(
            model=model,
            dataset=outer_test_df,
            feature_columns=feature_columns,
            label_column="label",
        )
        calibration_payload = _search_artifacts.compute_candidate_calibration_artifacts(
            run_dir=run_dir,
            candidate_name=candidate.name,
            y_calibration=y_calibration,
            score_calibration=score_calibration,
            y_test=y_test,
            score_test=score_test,
            bins=int(args.calibration_bins),
        )
        stage_update("calibration_eval", state="done")

        if not bool(args.disable_error_analysis):
            try:
                error_analysis_payload = _search_artifacts.compute_candidate_error_analysis_artifacts(
                    model=model,
                    dataset=outer_test_df,
                    run_dir=run_dir,
                    candidate_name=candidate.name,
                    feature_columns=feature_columns,
                    detailed=bool(args.error_analysis_mode == "full"),
                )
            except Exception as exc:
                error_analysis_payload = {
                    "status": "failed",
                    "reason": str(exc),
                }
        else:
            error_analysis_payload = {
                "status": "skipped",
                "reason": "disabled_by_flag",
            }

        row["hpo"] = hpo_result
        row["nas"] = nas_result
        row["selected_params_source"] = selected_source
        row["selected_params"] = selected_params
        if candidate.kind == "hybrid_pair":
            row["hybrid_config"] = _search_candidate.resolve_hybrid_config_for_report(
                candidate=candidate,
                args=args,
                selected_params=selected_params,
            )
        row["test_metrics"] = metrics
        row["explainability"] = explain_payload
        row["calibration"] = calibration_payload
        row["error_analysis"] = error_analysis_payload
        row["runtime_seconds"] = float(monotonic() - step_start)
        run_logger.info(
            "candidate=%s status=ok source=%s f1=%s runtime_seconds=%.4f",
            candidate.name,
            selected_source,
            metrics.get("f1"),
            row["runtime_seconds"],
        )
        return row, model
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        row["runtime_seconds"] = float(monotonic() - step_start)
        stage_update("candidate", state="failed", detail=str(exc))
        run_logger.warning(
            "candidate=%s status=failed error=%s runtime_seconds=%.4f",
            candidate.name,
            str(exc),
            row["runtime_seconds"],
        )
        return row, None