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
from pathologic.search.logging import emit, inner_search_runtime
from pathologic.search.spec import BudgetProfile, CandidateSpec
from pathologic.search.utils import parse_model_pool
from pathologic.utils.compute_cost import (
    benchmark_inference_latency,
    collect_framework_versions,
    collect_gpu_memory_snapshot,
    collect_reproducibility_settings,
    collect_system_info,
    create_process_memory_monitor,
    extract_iteration_metadata,
    resolve_batch_size,
    reset_gpu_peak_memory_stats,
)
from pathologic.utils.hardware import detect_preferred_device


StageUpdate = Callable[[str], None]

_GPU_CAPABLE_ALIASES = {"xgboost", "lightgbm", "catboost", "tabnet"}
_CUDA_REQUIRED_ALIASES = {"xgboost", "catboost", "tabnet"}


def _emit_gpu_capability_warnings(
    *,
    candidate: CandidateSpec,
    model: PathoLogic,
    run_logger: logging.Logger,
) -> None:
    device = str(getattr(model, "device", "cpu")).strip().lower()
    members = [str(alias).strip().lower() for alias in candidate.members]
    gpu_capable = [alias for alias in members if alias in _GPU_CAPABLE_ALIASES]
    cpu_only = [alias for alias in members if alias not in _GPU_CAPABLE_ALIASES]

    if device != "cuda":
        cuda_required = [alias for alias in members if alias in _CUDA_REQUIRED_ALIASES]
        if cuda_required:
            emit(
                (
                    f"[gpu-warning] candidate={candidate.name} device={device}. "
                    "CUDA-dependent backends may fall back to CPU/MPS: "
                    + ", ".join(cuda_required)
                    + "."
                ),
                color="yellow",
                run_logger=run_logger,
            )
        if "lightgbm" in members:
            emit(
                (
                    f"[gpu] candidate={candidate.name} lightgbm can still use OpenCL GPU "
                    "with device='gpu' even when torch CUDA detector is unavailable."
                ),
                color="cyan",
                run_logger=run_logger,
            )

    if cpu_only:
        emit(
            (
                f"[gpu-warning] candidate={candidate.name} CPU-only model(s): "
                + ", ".join(cpu_only)
                + ". GPU acceleration is not available for these aliases in this codebase."
            ),
            color="yellow",
            run_logger=run_logger,
        )

    if gpu_capable:
        emit(
            (
                f"[gpu] candidate={candidate.name} GPU-capable model(s): "
                + ", ".join(gpu_capable)
                + "."
            ),
            color="cyan",
            run_logger=run_logger,
        )


def _emit_gpu_runtime_backend_warnings(
    *,
    candidate: CandidateSpec,
    model: PathoLogic,
    run_logger: logging.Logger,
) -> None:
    trained_model = getattr(model, "_trained_model", None)
    estimator = getattr(trained_model, "estimator", None)
    if estimator is None:
        return

    members = [str(alias).strip().lower() for alias in candidate.members]

    if "xgboost" in members:
        try:
            xgb_params = estimator.get_xgb_params()
            if str(xgb_params.get("device", "")).strip().lower() != "cuda":
                emit(
                    (
                        f"[gpu-warning] candidate={candidate.name} xgboost backend "
                        "fell back to CPU."
                    ),
                    color="yellow",
                    run_logger=run_logger,
                )
        except Exception:
            pass

    if "lightgbm" in members:
        try:
            lgb_params = estimator.get_params()
            if str(lgb_params.get("device", "")).strip().lower() != "gpu":
                emit(
                    (
                        f"[gpu-warning] candidate={candidate.name} lightgbm backend "
                        "is not running with GPU (likely CPU fallback/build limitation)."
                    ),
                    color="yellow",
                    run_logger=run_logger,
                )
        except Exception:
            pass

    if "catboost" in members:
        try:
            task_type = str(estimator.get_param("task_type") or "").strip().upper()
            if task_type != "GPU":
                emit(
                    (
                        f"[gpu-warning] candidate={candidate.name} catboost backend "
                        "is not running on GPU."
                    ),
                    color="yellow",
                    run_logger=run_logger,
                )
        except Exception:
            pass


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
    compute_cost_enabled = not bool(getattr(args, "disable_compute_cost", False))
    compute_cost_single_runs = int(getattr(args, "compute_cost_single_runs", 20))
    compute_cost_batch_runs = int(getattr(args, "compute_cost_batch_runs", 10))
    compute_cost_warmup_runs = int(getattr(args, "compute_cost_warmup_runs", 2))
    compute_cost_batch_size = int(getattr(args, "compute_cost_batch_size", 256))

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
        compute_cost_payload: dict[str, Any] = {
            "status": "enabled" if compute_cost_enabled else "skipped",
            "config": {
                "single_runs": compute_cost_single_runs,
                "batch_runs": compute_cost_batch_runs,
                "warmup_runs": compute_cost_warmup_runs,
                "batch_size": compute_cost_batch_size,
            },
        }
        if compute_cost_enabled:
            compute_cost_payload["system"] = collect_system_info()
            compute_cost_payload["frameworks"] = collect_framework_versions()
            compute_cost_payload["gpu_before_train"] = collect_gpu_memory_snapshot()
            compute_cost_payload["reproducibility"] = collect_reproducibility_settings(
                seed=int(args.seed),
                model=model,
            )
            reset_gpu_peak_memory_stats()
            train_memory_monitor = create_process_memory_monitor(sample_interval_seconds=0.05)
        else:
            train_memory_monitor = None

        if detect_preferred_device() != "cuda":
            emit(
                (
                    "[gpu-warning] System CUDA backend is unavailable. "
                    "CUDA-dependent models may run on CPU/MPS backends. "
                    "LightGBM can still use OpenCL GPU with device='gpu'."
                ),
                color="yellow",
                run_logger=run_logger,
            )
        _emit_gpu_capability_warnings(candidate=candidate, model=model, run_logger=run_logger)

        model.defaults.setdefault("data", {})["required_features"] = list(feature_columns)
        model.defaults.setdefault("data", {})["label_column"] = "label"
        model.defaults.setdefault("data", {})["gene_column"] = "gene_id"

        search_space = candidate.tuning_search_space
        if not search_space:
            cfg_space = model._resolve_model_config().get("tuning_search_space")  # noqa: SLF001
            if isinstance(cfg_space, dict):
                search_space = {
                    str(k): dict(v)
                    for k, v in cfg_space.items()
                    if isinstance(v, dict)
                }

        run_nas_for_candidate = _search_hpo_nas.should_run_nas_for_candidate(candidate)
        level1_space: dict[str, dict[str, Any]] = {}
        level2_space: dict[str, dict[str, Any]] = {}
        if candidate.kind == "hybrid_pair" and search_space:
            level1_space, level2_space = _search_hpo_nas.split_hybrid_hpo_search_space(
                search_space
            )

        def _run_hpo_stage(
            *,
            stage_name: str,
            search_space_override: dict[str, dict[str, Any]] | None = None,
            base_model_params_override: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
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
                        stage_name,
                        state="progress",
                        detail=f"{_state['done']}/{_total_trials} trials{score_text}",
                    )

                stage_update(stage_name, state="start")
                with inner_search_runtime(
                    quiet=quiet_inner_search,
                    show_inner_progress=True,
                    suppress_stdout=True,
                    suppress_stderr=False,
                ):
                    hpo_result_local = _search_hpo_nas.run_hpo(
                        model=model,
                        train_csv=str(outer_train_csv),
                        objective=args.objective,
                        tune_engine=args.tune_engine,
                        budget=budget,
                        cv_splits=cv_splits,
                        n_trials_override=args.n_trials,
                        on_trial_complete=_on_hpo_trial_complete,
                        search_space_override=search_space_override,
                        base_model_params_override=base_model_params_override,
                    )
                if "status" not in hpo_result_local:
                    hpo_result_local["status"] = "ok"
                stage_update(
                    stage_name,
                    state="done",
                    detail=str(hpo_result_local.get("status", "ok")),
                )
                return hpo_result_local
            except Exception as exc:
                stage_update(stage_name, state="failed", detail=str(exc))
                return {"status": "failed", "reason": str(exc)}

        def _run_nas_stage(*, base_model_params: dict[str, Any]) -> dict[str, Any]:
            if not run_nas_for_candidate:
                stage_update("nas", state="start")
                skipped = _search_hpo_nas.skipped_nas_result(reason="model_family_policy")
                stage_update("nas", state="done", detail=str(skipped.get("status", "skipped")))
                return skipped

            try:
                regularization_models = parse_model_pool(
                    getattr(args, "regularization_models", None)
                )
                optimize_regularization_in_nas = bool(
                    getattr(args, "optimize_regularization_in_nas", False)
                )
                if candidate.kind == "hybrid_pair":
                    nas_search_space = _search_hpo_nas.build_hybrid_neural_nas_search_space(
                        candidate=candidate,
                        search_space=search_space,
                    )
                    if not nas_search_space:
                        stage_update("nas", state="start")
                        skipped = _search_hpo_nas.skipped_nas_result(
                            reason="model_family_policy_non_neural_hybrid"
                        )
                        stage_update(
                            "nas",
                            state="done",
                            detail=str(skipped.get("status", "skipped")),
                        )
                        return skipped

                    if not optimize_regularization_in_nas:
                        nas_search_space = _search_candidate.strip_regularization_search_space(
                            search_space=nas_search_space,
                            members=_search_hpo_nas.hybrid_neural_member_aliases(candidate),
                            regularization_models=regularization_models,
                        )
                else:
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
                    nas_result_local = _search_hpo_nas.run_nas(
                        candidate=candidate,
                        seed=int(args.seed),
                        nas_strategy=args.nas_strategy,
                        budget=budget,
                        nas_candidates_override=args.nas_candidates,
                        search_space=nas_search_space,
                        base_model_params=base_model_params,
                        x_train=x_train_nas,
                        y_train=y_train_nas,
                        x_val=x_val_nas,
                        y_val=y_val_nas,
                        on_candidate_complete=_on_nas_candidate_complete,
                    )
                stage_update("nas", state="done", detail=str(nas_result_local.get("status", "ok")))
                return nas_result_local
            except Exception as exc:
                stage_update("nas", state="failed", detail=str(exc))
                return {"status": "failed", "reason": str(exc)}

        hpo_level1_result: dict[str, Any] | None = None
        hpo_level2_result: dict[str, Any] | None = None

        if candidate.kind == "hybrid_pair":
            nas_result = _run_nas_stage(base_model_params={})
            nas_best_params = (
                dict(nas_result["best_params"])
                if isinstance(nas_result.get("best_params"), dict)
                else {}
            )
            hpo_level1_result = _run_hpo_stage(
                stage_name="hpo_level1",
                search_space_override=level1_space,
                base_model_params_override=nas_best_params,
            )
            level1_best_params = (
                dict(hpo_level1_result["best_params"])
                if isinstance(hpo_level1_result.get("best_params"), dict)
                else {}
            )
            hpo_level2_result = _run_hpo_stage(
                stage_name="hpo_level2",
                search_space_override=level2_space,
                base_model_params_override=level1_best_params,
            )
            hpo_result = _search_hpo_nas.merge_hpo_level_results(
                level1_result=hpo_level1_result,
                level2_result=hpo_level2_result,
            )
        elif run_nas_for_candidate:
            nas_result = _run_nas_stage(base_model_params={})
            hpo_result = _run_hpo_stage(stage_name="hpo")
        else:
            hpo_result = _run_hpo_stage(stage_name="hpo")
            nas_result = _run_nas_stage(base_model_params={})

        if run_nas_for_candidate:
            if isinstance(hpo_result.get("best_params"), dict):
                selected_params = dict(hpo_result["best_params"])
                selected_source = "hpo_after_nas"
            elif isinstance(nas_result.get("best_params"), dict):
                selected_params = dict(nas_result["best_params"])
                selected_source = "nas"
            else:
                selected_params = {}
                selected_source = "defaults"
        else:
            if candidate.kind == "hybrid_pair":
                if isinstance(hpo_result.get("best_params"), dict):
                    selected_params = dict(hpo_result["best_params"])
                    hpo_selected_source = str(
                        hpo_result.get("selected_params_source", "hpo_level1")
                    )
                    if isinstance(nas_result.get("best_params"), dict):
                        if hpo_selected_source == "hpo_level2_after_level1":
                            selected_source = "hpo_level2_after_level1_after_nas"
                        elif hpo_selected_source == "hpo_level1":
                            selected_source = "hpo_level1_after_nas"
                        else:
                            selected_source = hpo_selected_source
                    else:
                        selected_source = hpo_selected_source
                elif isinstance(nas_result.get("best_params"), dict):
                    selected_params = dict(nas_result["best_params"])
                    selected_source = "nas_hybrid_member_only"
                else:
                    selected_params = {}
                    selected_source = "defaults"
            else:
                selected_params, selected_source = _search_hpo_nas.select_best_params(
                    hpo_result,
                    nas_result,
                )

        train_kwargs: dict[str, Any] = {}
        if selected_params:
            train_kwargs["model_params"] = selected_params

        stage_update("train", state="start")
        train_started = monotonic()
        if train_memory_monitor is not None:
            train_memory_monitor.start()
        with inner_search_runtime(
            quiet=quiet_inner_search,
            show_inner_progress=False,
            suppress_stdout=True,
            suppress_stderr=True,
        ):
            model.train(str(outer_train_csv), **train_kwargs)
        train_seconds = float(monotonic() - train_started)
        train_memory_profile = (
            train_memory_monitor.stop() if train_memory_monitor is not None else None
        )
        _emit_gpu_runtime_backend_warnings(candidate=candidate, model=model, run_logger=run_logger)
        stage_update("train", state="done")

        if compute_cost_enabled:
            compute_cost_payload["training"] = extract_iteration_metadata(
                model=model,
                train_seconds=train_seconds,
            )
            compute_cost_payload["training"]["batch_size"] = resolve_batch_size(
                model=model,
                selected_params=selected_params,
            )
            if isinstance(train_memory_profile, dict):
                compute_cost_payload["training"]["memory"] = train_memory_profile
            compute_cost_payload["gpu_after_train"] = collect_gpu_memory_snapshot()

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

        if bool(getattr(args, "disable_panel_thresholds", False)):
            panel_thresholds_payload = {
                "candidate": candidate.name,
                "status": "skipped",
                "reason": "disabled_by_flag",
                "panel_column": str(getattr(args, "panel_threshold_column", "Veri_Kaynagi_Paneli")),
                "rows": [],
                "artifacts": {},
            }
        else:
            try:
                panel_thresholds_payload = _search_artifacts.compute_candidate_panel_threshold_artifacts(
                    model=model,
                    dataset=outer_calibration_df,
                    run_dir=run_dir,
                    candidate_name=candidate.name,
                    feature_columns=feature_columns,
                    panel_column=str(
                        getattr(args, "panel_threshold_column", "Veri_Kaynagi_Paneli")
                    ),
                    label_column="label",
                    min_samples=int(getattr(args, "panel_threshold_min_samples", 1)),
                    default_threshold=float(getattr(args, "panel_threshold_default", 0.5)),
                )
            except Exception as exc:
                panel_thresholds_payload = {
                    "candidate": candidate.name,
                    "status": "failed",
                    "reason": str(exc),
                    "panel_column": str(
                        getattr(args, "panel_threshold_column", "Veri_Kaynagi_Paneli")
                    ),
                    "rows": [],
                    "artifacts": {},
                }

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

        if compute_cost_enabled:
            try:
                inference_latency = benchmark_inference_latency(
                    model=model,
                    dataset=outer_test_df,
                    feature_columns=feature_columns,
                    label_column="label",
                    single_runs=compute_cost_single_runs,
                    batch_runs=compute_cost_batch_runs,
                    warmup_runs=compute_cost_warmup_runs,
                    batch_size=compute_cost_batch_size,
                )
                compute_cost_payload["inference"] = {
                    "single_sample_ms": float(inference_latency.single_sample_ms),
                    "batch_total_ms": float(inference_latency.batch_total_ms),
                    "batch_per_sample_ms": float(inference_latency.batch_per_sample_ms),
                    "batch_size": int(inference_latency.batch_size),
                    "full_dataset_ms": float(inference_latency.full_dataset_ms),
                    "full_dataset_size": int(inference_latency.full_dataset_size),
                }
                compute_cost_payload["gpu_after_inference"] = collect_gpu_memory_snapshot()
            except Exception as exc:
                compute_cost_payload["inference"] = {
                    "status": "failed",
                    "reason": str(exc),
                }

            compute_cost_payload = _search_artifacts.compute_candidate_compute_cost_artifacts(
                run_dir=run_dir,
                candidate_name=candidate.name,
                payload=compute_cost_payload,
            )
        else:
            compute_cost_payload = {
                "status": "skipped",
                "reason": "disabled_by_flag",
            }

        row["hpo"] = hpo_result
        if hpo_level1_result is not None:
            row["hpo_level1"] = hpo_level1_result
        if hpo_level2_result is not None:
            row["hpo_level2"] = hpo_level2_result
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
        row["panel_thresholds"] = panel_thresholds_payload
        row["error_analysis"] = error_analysis_payload
        row["compute_cost"] = compute_cost_payload
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