"""CLI entrypoint helpers for exhaustive search orchestration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

from pathologic.search.core import run_exhaustive_search
from pathologic.search.data import resolve_search_defaults_from_defaults
from pathologic.search.logging import colorize
from pathologic.search.spec import BUDGET_PROFILES


def build_arg_parser() -> argparse.ArgumentParser:
    search_defaults = resolve_search_defaults_from_defaults()
    explain_defaults_raw = search_defaults.get("explainability")
    explain_defaults = explain_defaults_raw if isinstance(explain_defaults_raw, dict) else {}

    hybrid_strategy_default = str(search_defaults.get("hybrid_strategy", "soft_voting"))
    hybrid_weighting_policy_default = str(search_defaults.get("hybrid_weighting_policy", "auto"))
    hybrid_weighting_objective_default = str(search_defaults.get("hybrid_weighting_objective", "f1"))
    hybrid_normalize_default = bool(search_defaults.get("hybrid_normalize_weights", True))
    hybrid_meta_model_default = str(search_defaults.get("hybrid_meta_model", "logreg"))
    hybrid_stacking_cv_default = int(search_defaults.get("hybrid_stacking_cv", 3))
    hybrid_blend_size_default = float(search_defaults.get("hybrid_blend_size", 0.2))
    hybrid_tune_strategy_default = bool(search_defaults.get("hybrid_tune_strategy_and_params", True))
    regularization_profile_default = str(search_defaults.get("regularization_profile", "auto"))
    regularization_models_default_raw = search_defaults.get(
        "regularization_models",
        ["xgboost", "lightgbm", "catboost", "tabnet", "mlp"],
    )
    if isinstance(regularization_models_default_raw, list):
        regularization_models_default = ",".join(
            str(item).strip() for item in regularization_models_default_raw if str(item).strip()
        )
    else:
        regularization_models_default = str(regularization_models_default_raw)
    optimize_regularization_in_nas_default = bool(
        search_defaults.get("optimize_regularization_in_nas", False)
    )
    default_model_pool_raw = search_defaults.get("default_model_pool", "xgboost,tabnet")
    if isinstance(default_model_pool_raw, str) and default_model_pool_raw.strip():
        default_model_pool = default_model_pool_raw
    else:
        default_model_pool = "xgboost,tabnet"
    error_analysis_mode_default = str(search_defaults.get("error_analysis_mode", "hybrid"))
    explain_top_k_features_default = int(explain_defaults.get("top_k_features", 5))
    explain_top_k_samples_default = int(explain_defaults.get("top_k_samples", 10))
    explain_background_size_default = int(explain_defaults.get("background_size", 100))
    explain_fp_top_k_default = int(explain_defaults.get("fp_top_k", 10))
    explain_fp_min_negative_count_default = int(explain_defaults.get("fp_min_negative_count", 1))

    parser = argparse.ArgumentParser(
        description="Run leakage-safe exhaustive model search for PathoLogic",
    )
    parser.add_argument("data_csv", help="Input CSV path")
    parser.add_argument(
        "--output-dir",
        default="results/model_search",
        help="Artifact output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--objective", default="f1", help="Selection objective metric")
    parser.add_argument(
        "--budget-profile",
        default="aggressive",
        choices=sorted(BUDGET_PROFILES.keys()),
        help="Search budget profile",
    )
    parser.add_argument(
        "--model-pool",
        default=default_model_pool,
        help=(
            "Comma-separated model pool to evaluate. "
            "Defaults to configured search.default_model_pool "
            "(and hybrid combinations unless disabled)."
        ),
    )
    parser.add_argument(
        "--tune-engine",
        default="optuna",
        choices=["optuna", "random", "grid"],
    )
    parser.add_argument(
        "--nas-strategy",
        default="low_fidelity",
        choices=["low_fidelity", "weight_sharing"],
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Override HPO trial count")
    parser.add_argument(
        "--nas-candidates",
        type=int,
        default=None,
        help="Override NAS candidate count",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=None,
        help="Override inner CV fold count",
    )
    parser.add_argument(
        "--outer-test-size",
        type=float,
        default=0.2,
        help="Outer holdout test ratio",
    )
    parser.add_argument(
        "--outer-val-size",
        type=float,
        default=0.2,
        help="Outer holdout val ratio",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Legacy alias for --model-pool (used if --model-pool is empty)",
    )
    parser.add_argument("--exclude-models", default=None, help="Comma-separated model exclude list")
    parser.add_argument(
        "--disable-hybrids",
        action="store_true",
        help="Disable hybrid pair candidates",
    )
    parser.add_argument(
        "--hybrid-strategy",
        default=hybrid_strategy_default,
        choices=["soft_voting", "hard_voting", "stacking", "blending"],
        help="Hybrid pair strategy applied to generated pair candidates",
    )
    parser.add_argument(
        "--hybrid-tune-strategy-and-params",
        action="store_true",
        help=(
            "Expand hybrid pair tuning search space to include strategy, weighting policy, "
            "meta model, stacking cv, blending size, and voting weight ratio"
        ),
    )
    parser.add_argument(
        "--regularization-profile",
        default=regularization_profile_default,
        choices=["auto", "off"],
        help=(
            "Regularization search-space behavior: auto injects model-appropriate "
            "regularization params into candidate search spaces; off removes them."
        ),
    )
    parser.add_argument(
        "--regularization-models",
        default=regularization_models_default,
        help=(
            "Comma-separated model aliases allowed to receive regularization optimization "
            "(example: xgboost,lightgbm,catboost)."
        ),
    )
    parser.add_argument(
        "--optimize-regularization-in-nas",
        action="store_true",
        help="Enable regularization parameters in NAS search space (disabled by default).",
    )
    parser.add_argument(
        "--hybrid-weights",
        default=None,
        help="Comma-separated pair weights (example: 0.7,0.3)",
    )
    parser.add_argument(
        "--hybrid-weighting-policy",
        default=hybrid_weighting_policy_default,
        choices=["auto", "manual", "equal", "inverse_error", "objective_proportional"],
        help="Hybrid voting weight policy",
    )
    parser.add_argument(
        "--hybrid-weighting-objective",
        default=hybrid_weighting_objective_default,
        choices=["f1", "precision", "recall", "accuracy", "roc_auc"],
        help="Objective used when dynamic hybrid weighting policy is enabled",
    )
    parser.add_argument(
        "--disable-hybrid-normalize-weights",
        action="store_true",
        help="Disable normalization of provided hybrid member weights",
    )
    parser.add_argument(
        "--hybrid-meta-model",
        default=hybrid_meta_model_default,
        help="Meta model alias for stacking/blending hybrid strategies",
    )
    parser.add_argument(
        "--hybrid-stacking-cv",
        type=int,
        default=hybrid_stacking_cv_default,
        help="Cross-validation fold count for stacking strategy",
    )
    parser.add_argument(
        "--hybrid-blend-size",
        type=float,
        default=hybrid_blend_size_default,
        help="Holdout ratio for blending strategy",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Limit evaluated candidate count",
    )
    parser.add_argument(
        "--delete-prepared",
        action="store_true",
        help="Delete prepared dataset after run",
    )
    parser.add_argument(
        "--verbose-inner-search",
        action="store_true",
        help="Show detailed progress/log output for HPO and NAS internals",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins for ECE and reliability diagram",
    )
    parser.add_argument(
        "--calibration-weight-objective",
        type=float,
        default=1.0,
        help="Calibration-aware winner score weight for objective metric",
    )
    parser.add_argument(
        "--calibration-weight-ece",
        type=float,
        default=1.0,
        help="Calibration-aware winner penalty weight for ECE",
    )
    parser.add_argument(
        "--calibration-weight-brier",
        type=float,
        default=1.0,
        help="Calibration-aware winner penalty weight for Brier score",
    )
    parser.add_argument(
        "--disable-explainability",
        action="store_true",
        help="Disable explainability and false-positive hotspot artifact generation",
    )
    parser.add_argument(
        "--disable-error-analysis",
        action="store_true",
        help="Disable multi-dimensional error analysis artifact generation",
    )
    parser.add_argument(
        "--error-analysis-mode",
        default=error_analysis_mode_default,
        choices=["summary", "full", "hybrid"],
        help=(
            "summary: lightweight per-candidate summaries, "
            "full: detailed per-candidate analysis, "
            "hybrid: summary for all candidates and detailed for winner"
        ),
    )
    parser.add_argument(
        "--explain-top-k-features",
        type=int,
        default=explain_top_k_features_default,
        help="Top-k global features to retain in explainability summary",
    )
    parser.add_argument(
        "--explain-top-k-samples",
        type=int,
        default=explain_top_k_samples_default,
        help="Top-k samples to include in explainability sample-level narratives",
    )
    parser.add_argument(
        "--explain-background-size",
        type=int,
        default=explain_background_size_default,
        help="Background sample size for attribution backend",
    )
    parser.add_argument(
        "--explain-fp-top-k",
        type=int,
        default=explain_fp_top_k_default,
        help="Top-k false-positive hotspots to retain",
    )
    parser.add_argument(
        "--explain-fp-min-negative-count",
        type=int,
        default=explain_fp_min_negative_count_default,
        help="Minimum negative support required for false-positive hotspot enrichment",
    )

    parser.set_defaults(disable_hybrid_normalize_weights=not hybrid_normalize_default)
    parser.set_defaults(hybrid_tune_strategy_and_params=hybrid_tune_strategy_default)
    parser.set_defaults(optimize_regularization_in_nas=optimize_regularization_in_nas_default)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(colorize("[1/4] Preparing exhaustive search run...", "cyan", bold=True))
    summary = run_exhaustive_search(args)
    print(colorize("[2/4] Search completed.", "green", bold=True))
    print(colorize("[3/4] Winner: " + str(summary["winner"]), "green", bold=True))
    print(colorize("[4/4] Final summary:", "cyan", bold=True))
    print(colorize(json.dumps(summary, ensure_ascii=True), "magenta"))
    return 0
