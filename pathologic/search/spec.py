"""Search dataclasses and profile constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


HYBRID_TUNING_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "quick": {
        "strategy": ["soft_voting", "hard_voting"],
        "weighting_policy": ["auto", "manual", "equal", "objective_proportional"],
        "meta_model_alias": ["logreg"],
        "weight_ratio": {"low": 0.35, "high": 0.65},
        "cv": {"low": 2, "high": 3},
        "blend_size": {"low": 0.15, "high": 0.25},
    },
    "balanced": {
        "strategy": ["soft_voting", "hard_voting", "stacking", "blending"],
        "weighting_policy": [
            "auto",
            "manual",
            "equal",
            "inverse_error",
            "objective_proportional",
        ],
        "meta_model_alias": ["logreg", "random_forest"],
        "weight_ratio": {"low": 0.2, "high": 0.8},
        "cv": {"low": 2, "high": 4},
        "blend_size": {"low": 0.1, "high": 0.3},
    },
    "aggressive": {
        "strategy": ["soft_voting", "hard_voting", "stacking", "blending"],
        "weighting_policy": [
            "auto",
            "manual",
            "equal",
            "inverse_error",
            "objective_proportional",
        ],
        "meta_model_alias": ["logreg", "random_forest", "xgboost"],
        "weight_ratio": {"low": 0.1, "high": 0.9},
        "cv": {"low": 2, "high": 5},
        "blend_size": {"low": 0.1, "high": 0.35},
    },
}


@dataclass(frozen=True)
class BudgetProfile:
    """Search budget profile."""

    n_trials: int
    timeout_minutes: float
    nas_candidates: int
    cv_splits: int


@dataclass(frozen=True)
class CandidateSpec:
    """Single candidate definition for leaderboard evaluation."""

    name: str
    kind: str
    members: tuple[str, ...]
    tuning_search_space: dict[str, dict[str, Any]]


BUDGET_PROFILES: dict[str, BudgetProfile] = {
    "aggressive": BudgetProfile(n_trials=40, timeout_minutes=120.0, nas_candidates=20, cv_splits=5),
    "balanced": BudgetProfile(n_trials=15, timeout_minutes=45.0, nas_candidates=8, cv_splits=4),
    "quick": BudgetProfile(n_trials=3, timeout_minutes=10.0, nas_candidates=2, cv_splits=3),
}
