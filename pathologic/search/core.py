"""Core orchestration utilities for search workflows."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from time import monotonic
from typing import Any

from pathologic.search import bootstrap as _search_bootstrap
from pathologic.search import artifacts as _search_artifacts
from pathologic.search import orchestration as _search_orchestration
from pathologic.search import reporting as _search_reporting
from pathologic.search.logging import (
    emit,
)
from pathologic.search.spec import BUDGET_PROFILES


def safe_metric(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("-inf")
    if math.isnan(parsed):
        return float("-inf")
    return parsed


def ensure_no_holdout_leakage(summary: dict[str, Any]) -> None:
    for key in (
        "train_val_shared_genes",
        "train_test_shared_genes",
        "val_test_shared_genes",
    ):
        if key in summary and int(summary[key]) != 0:
            raise RuntimeError(f"Leakage detected: {key}={summary[key]}")


def rank_leaderboard(rows: list[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = item.get("test_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        return (
            safe_metric(metrics.get(objective)),
            safe_metric(metrics.get("mcc")),
            safe_metric(metrics.get("roc_auc")),
            -float(item.get("runtime_seconds", 0.0)),
        )

    return sorted(rows, key=sort_key, reverse=True)


def _configure_windows_joblib_cpu_detection() -> None:
    if os.name != "nt":
        return
    if os.environ.get("LOKY_MAX_CPU_COUNT"):
        return
    cpu_count = os.cpu_count()
    if not isinstance(cpu_count, int) or cpu_count < 1:
        return
    cpu_limit = max(1, cpu_count - 1)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(cpu_limit)

    try:
        from joblib.externals.loky.backend import context as loky_context

        loky_context.physical_cores_cache = cpu_limit
    except Exception:
        pass


def _suppress_known_parallel_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Could not find the number of physical cores",
    )


def run_exhaustive_search(args: argparse.Namespace) -> dict[str, Any]:
    _configure_windows_joblib_cpu_detection()
    _suppress_known_parallel_warnings()

    budget = BUDGET_PROFILES[args.budget_profile]
    context = _search_bootstrap.bootstrap_search_run(
        args,
        budget=budget,
    )

    emit(
        f"[search] candidates={len(context.candidates)} objective={args.objective}",
        color="cyan",
        bold=True,
        run_logger=context.run_logger,
    )
    emit(
        f"[search] inner_search={'quiet' if context.quiet_inner_search else 'verbose'}",
        color="cyan",
        run_logger=context.run_logger,
    )
    emit(
        f"[search] hpo_split=cross_validation n_splits={context.cv_splits}",
        color="cyan",
        run_logger=context.run_logger,
    )
    emit(
        (
            "[search] explainability="
            + ("disabled" if bool(args.disable_explainability) else "enabled")
            + f" top_k_features={int(args.explain_top_k_features)}"
            + f" top_k_samples={int(args.explain_top_k_samples)}"
            + f" background_size={int(args.explain_background_size)}"
        ),
        color="cyan",
        run_logger=context.run_logger,
    )
    emit(
        f"[search] run_log={context.log_path}",
        color="cyan",
        run_logger=context.run_logger,
    )

    leaderboard, successful_models = _search_orchestration.run_candidate_search_loop(
        args=args,
        candidates=context.candidates,
        budget=budget,
        quiet_inner_search=context.quiet_inner_search,
        outer_train_csv=context.outer_train_csv,
        outer_calibration_csv=context.outer_calibration_csv,
        outer_test_csv=context.outer_test_csv,
        outer_test_df=context.outer_test_df,
        outer_calibration_df=context.outer_calibration_df,
        run_dir=context.run_dir,
        feature_columns=context.feature_columns,
        cv_splits=context.cv_splits,
        x_train_nas=context.x_train_nas,
        y_train_nas=context.y_train_nas,
        x_val_nas=context.x_val_nas,
        y_val_nas=context.y_val_nas,
        run_logger=context.run_logger,
    )

    ranked = rank_leaderboard([r for r in leaderboard if r.get("status") == "ok"], args.objective)
    if not ranked:
        raise RuntimeError("All candidates failed. Inspect leaderboard.json for details.")

    elapsed = float(monotonic() - context.started)

    objective_weight = float(args.calibration_weight_objective)
    ece_weight = float(args.calibration_weight_ece)
    brier_weight = float(args.calibration_weight_brier)

    calibration_summary_rows, candidate_method_ranking, candidate_calibration_ranking = (
        _search_reporting.compute_calibration_rankings(
            leaderboard=leaderboard,
            objective=args.objective,
            objective_weight=objective_weight,
            ece_weight=ece_weight,
            brier_weight=brier_weight,
        )
    )
    objective_best, best = _search_reporting.select_calibration_aware_winner(
        ranked=ranked,
        candidate_calibration_ranking=candidate_calibration_ranking,
    )

    if (
        not bool(args.disable_error_analysis)
        and str(args.error_analysis_mode).strip().lower() in {"hybrid", "full"}
    ):
        winner_name = str(best.get("candidate", ""))
        winner_model = successful_models.get(winner_name)
        if winner_model is not None:
            try:
                winner_error_payload = _search_artifacts.compute_candidate_error_analysis_artifacts(
                    model=winner_model,
                    dataset=context.outer_test_df,
                    run_dir=context.run_dir,
                    candidate_name=winner_name,
                    feature_columns=context.feature_columns,
                    detailed=True,
                )
                for row in leaderboard:
                    if str(row.get("candidate", "")) == winner_name:
                        row["error_analysis"] = winner_error_payload
                        break
            except Exception as exc:
                context.run_logger.warning(
                    "winner error analysis failed candidate=%s error=%s",
                    winner_name,
                    str(exc),
                )

    calibration_summary_path, error_analysis_summary_path, train_report_path = _search_reporting.write_run_reports(
        run_dir=context.run_dir,
        objective=args.objective,
        budget_profile=args.budget_profile,
        seed=int(args.seed),
        candidates_total=len(context.candidates),
        candidates_ok=len(ranked),
        leaderboard=leaderboard,
        best=best,
        objective_best=objective_best,
        elapsed_seconds=elapsed,
        prep_stats=context.prep_stats,
        feature_count=len(context.feature_columns),
        split_summary=context.split_summary,
        outer_train_rows=len(context.outer_base_train_df),
        outer_calibration_rows=len(context.outer_calibration_df),
        outer_test_rows=len(context.outer_test_df),
        objective_weight=objective_weight,
        ece_weight=ece_weight,
        brier_weight=brier_weight,
        calibration_summary_rows=calibration_summary_rows,
        candidate_method_ranking=candidate_method_ranking,
        candidate_calibration_ranking=candidate_calibration_ranking,
    )

    if args.delete_prepared:
        Path(context.prepared_csv).unlink(missing_ok=True)

    summary = {
        "run_dir": str(context.run_dir),
        "log_file": str(context.log_path),
        "objective": args.objective,
        "winner": best.get("candidate"),
        "winner_hybrid_config": best.get("hybrid_config"),
        "winner_selection_mode": "calibration_aware",
        "objective_only_winner": objective_best.get("candidate"),
        "winner_metrics": best.get("test_metrics", {}),
        "calibration_summary_file": str(calibration_summary_path),
        "calibration_summary_html_file": str(context.run_dir / "calibration_summary.html"),
        "error_analysis_summary_file": str(error_analysis_summary_path),
        "train_report_file": str(train_report_path),
        "train_report_html_file": str(context.run_dir / "train_report.html"),
        "elapsed_seconds": elapsed,
        "candidates_total": len(context.candidates),
        "candidates_ok": len(ranked),
    }
    context.run_logger.info("final_summary=%s", json.dumps(summary, ensure_ascii=True))
    return summary
