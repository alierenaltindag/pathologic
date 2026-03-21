"""HPO and NAS execution helpers for search workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pathologic import PathoLogic
from pathologic.data.loader import build_holdout_split
from pathologic.data.preprocessor import FoldPreprocessor
from pathologic.nas.search import NASearch
from pathologic.search.spec import BudgetProfile, CandidateSpec


def _safe_metric(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("-inf")
    if np.isnan(parsed):
        return float("-inf")
    return parsed


def run_hpo(
    *,
    model: PathoLogic,
    train_csv: str,
    objective: str,
    tune_engine: str,
    budget: BudgetProfile,
    cv_splits: int,
    n_trials_override: int | None,
    on_trial_complete: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not model._resolve_model_config().get("tuning_search_space"):  # noqa: SLF001
        return {"status": "skipped", "reason": "missing_tuning_search_space"}

    n_trials = int(n_trials_override) if n_trials_override is not None else budget.n_trials
    return model.tune(
        train_csv,
        engine=tune_engine,
        objective=objective,
        n_trials=n_trials,
        max_trials=n_trials,
        timeout_minutes=budget.timeout_minutes,
        callbacks=[on_trial_complete] if on_trial_complete is not None else None,
        split={
            "mode": "cross_validation",
            "cross_validation": {"n_splits": int(cv_splits), "stratified": True},
        },
    )


def build_nas_arrays(
    *,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = build_holdout_split(
        train_df,
        label_column="label",
        gene_column="gene_id",
        test_size=0.2,
        val_size=0.2,
        stratified=True,
        random_state=seed,
    )
    inner_train = train_df.iloc[split["train"]]
    inner_val = train_df.iloc[split["val"]]

    probe_defaults = PathoLogic("xgboost").defaults
    preprocess_cfg = probe_defaults.get("train", {}).get("preprocess", {})
    if not isinstance(preprocess_cfg, dict):
        preprocess_cfg = {}

    missing_value_policy = str(preprocess_cfg.get("missing_value_policy", "impute"))
    if missing_value_policy not in {"impute", "drop_rows"}:
        raise ValueError(
            "Config field 'preprocess.missing_value_policy' must be one of: "
            "drop_rows, impute"
        )

    processor = FoldPreprocessor(
        numeric_features=feature_columns,
        gene_column="gene_id",
        missing_value_policy=missing_value_policy,
        impute_strategy=str(preprocess_cfg.get("impute_strategy", "median")),
        scaler=str(preprocess_cfg.get("scaler", "standard")),
        per_gene=bool(preprocess_cfg.get("per_gene", True)),
        per_gene_features=(
            [str(value) for value in preprocess_cfg.get("per_gene_features", [])]
            if isinstance(preprocess_cfg.get("per_gene_features"), list)
            else None
        ),
        scaler_features=(
            [str(value) for value in preprocess_cfg.get("scaler_features", [])]
            if isinstance(preprocess_cfg.get("scaler_features"), list)
            else None
        ),
        add_missing_indicators=bool(preprocess_cfg.get("add_missing_indicators", False)),
        missing_indicator_features=(
            [str(value) for value in preprocess_cfg.get("missing_indicator_features", [])]
            if isinstance(preprocess_cfg.get("missing_indicator_features"), list)
            else None
        ),
    )

    train_processed = processor.fit_transform(inner_train)
    val_processed = processor.transform(inner_val)
    x_train = train_processed[feature_columns].to_numpy(dtype=float)
    y_train = train_processed["label"].to_numpy(dtype=int)
    x_val = val_processed[feature_columns].to_numpy(dtype=float)
    y_val = val_processed["label"].to_numpy(dtype=int)

    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError(
            "preprocess.missing_value_policy='drop_rows' removed all rows in NAS arrays."
        )
    return x_train, y_train, x_val, y_val


def run_nas(
    *,
    candidate: CandidateSpec,
    seed: int,
    nas_strategy: str,
    budget: BudgetProfile,
    nas_candidates_override: int | None,
    search_space: dict[str, dict[str, Any]],
    base_model_params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    on_candidate_complete: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not search_space:
        return {"status": "skipped", "reason": "missing_search_space"}

    n_candidates = (
        int(nas_candidates_override)
        if nas_candidates_override is not None
        else budget.nas_candidates
    )
    if n_candidates <= 0:
        return {"status": "skipped", "reason": "nas_candidates_disabled"}

    def scorer(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> float:
        del y_score
        return float(f1_score(y_true, y_pred, zero_division=0))

    runner = NASearch.for_model(
        candidate.name,
        strategy=nas_strategy,
        random_state=seed,
        direction="maximize",
        base_model_params=base_model_params,
        score_fn=scorer,
        fidelity_param_key="epochs",
    )
    result = runner.search(
        search_space=search_space,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        n_candidates=n_candidates,
        budget={"min_fidelity": 1, "max_fidelity": 5},
        callbacks=[on_candidate_complete] if on_candidate_complete is not None else None,
    )
    best_params = {
        key: value
        for key, value in dict(result.best_candidate.params).items()
        if key != "epochs"
    }
    return {
        "status": "ok",
        "strategy": result.strategy,
        "best_score": float(result.best_score),
        "best_params": best_params,
        "trials": len(result.trials),
        "stopped_reason": result.stopped_reason,
    }


def select_best_params(
    hpo_result: dict[str, Any],
    nas_result: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    hpo_score = _safe_metric(hpo_result.get("best_score"))
    nas_score = _safe_metric(nas_result.get("best_score"))

    if nas_score > hpo_score and isinstance(nas_result.get("best_params"), dict):
        return dict(nas_result["best_params"]), "nas"
    if isinstance(hpo_result.get("best_params"), dict):
        return dict(hpo_result["best_params"]), "hpo"
    return {}, "defaults"
