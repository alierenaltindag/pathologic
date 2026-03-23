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
from pathologic.models import get_model_metadata
from pathologic.nas.search import NASearch
from pathologic.search.spec import BudgetProfile, CandidateSpec


NEURAL_MODEL_FAMILIES = frozenset({"neural-network", "tabular-neural-network"})


def _candidate_aliases(candidate: CandidateSpec) -> tuple[str, ...]:
    if candidate.kind == "single" and candidate.members:
        return (str(candidate.members[0]),)
    return tuple(str(alias) for alias in candidate.members)


def candidate_model_families(candidate: CandidateSpec) -> tuple[str, ...]:
    aliases = _candidate_aliases(candidate)
    families: list[str] = []
    for alias in aliases:
        metadata = get_model_metadata(alias)
        families.append(str(metadata.family))
    return tuple(families)


def is_neural_model_alias(alias: str) -> bool:
    metadata = get_model_metadata(str(alias))
    return str(metadata.family) in NEURAL_MODEL_FAMILIES


def hybrid_neural_member_aliases(candidate: CandidateSpec) -> tuple[str, ...]:
    if candidate.kind != "hybrid_pair":
        return tuple()
    aliases: list[str] = []
    for alias in candidate.members:
        if is_neural_model_alias(str(alias)):
            aliases.append(str(alias))
    return tuple(aliases)


def should_run_nas_for_candidate(candidate: CandidateSpec) -> bool:
    if candidate.kind == "single":
        families = candidate_model_families(candidate)
        return any(family in NEURAL_MODEL_FAMILIES for family in families)
    if candidate.kind == "hybrid_pair":
        return bool(hybrid_neural_member_aliases(candidate))
    return False


def candidate_stage_order(candidate: CandidateSpec) -> tuple[str, ...]:
    if candidate.kind == "hybrid_pair":
        return (
            "nas",
            "hpo_level1",
            "hpo_level2",
            "train",
            "evaluate",
            "explainability",
            "calibration_fit",
            "calibration_eval",
        )

    if should_run_nas_for_candidate(candidate):
        return (
            "nas",
            "hpo",
            "train",
            "evaluate",
            "explainability",
            "calibration_fit",
            "calibration_eval",
        )
    return (
        "hpo",
        "nas",
        "train",
        "evaluate",
        "explainability",
        "calibration_fit",
        "calibration_eval",
    )


def skipped_nas_result(*, reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": str(reason),
        "trials": 0,
    }


def split_hybrid_hpo_search_space(
    search_space: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Split hybrid tuning space into member-level and strategy/meta-level spaces."""
    level1_member_space: dict[str, dict[str, Any]] = {}
    level2_strategy_space: dict[str, dict[str, Any]] = {}

    for key, spec in search_space.items():
        if not isinstance(spec, dict):
            continue

        key_text = str(key)
        if key_text.startswith("member__"):
            level1_member_space[key_text] = dict(spec)
            continue

        if (
            key_text == "strategy"
            or key_text == "meta_model_alias"
            or key_text.startswith("strategy__")
            or key_text.startswith("meta__")
        ):
            level2_strategy_space[key_text] = dict(spec)

    return level1_member_space, level2_strategy_space


def build_hybrid_neural_nas_search_space(
    *,
    candidate: CandidateSpec,
    search_space: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return only neural-member namespaced keys for hybrid NAS."""
    neural_aliases = set(hybrid_neural_member_aliases(candidate))
    if not neural_aliases:
        return {}

    filtered: dict[str, dict[str, Any]] = {}
    for key, spec in search_space.items():
        if not isinstance(spec, dict):
            continue
        key_text = str(key)
        if not key_text.startswith("member__"):
            continue
        parts = key_text.split("__", 2)
        if len(parts) != 3:
            continue
        member_alias = str(parts[1])
        if member_alias not in neural_aliases:
            continue
        filtered[key_text] = dict(spec)
    return filtered


def merge_hpo_level_results(
    *,
    level1_result: dict[str, Any],
    level2_result: dict[str, Any],
) -> dict[str, Any]:
    """Merge two-level hybrid HPO outputs into one backward-compatible payload."""
    level1_best = level1_result.get("best_params")
    level2_best = level2_result.get("best_params")

    merged_best_params: dict[str, Any] = {}
    if isinstance(level1_best, dict):
        merged_best_params.update(level1_best)
    if isinstance(level2_best, dict):
        merged_best_params.update(level2_best)

    level2_score = _safe_metric(level2_result.get("best_score"))
    level1_score = _safe_metric(level1_result.get("best_score"))
    if level2_score != float("-inf"):
        best_score = float(level2_score)
    elif level1_score != float("-inf"):
        best_score = float(level1_score)
    else:
        best_score = float("nan")

    if isinstance(level2_best, dict):
        selected_source = "hpo_level2_after_level1"
    elif isinstance(level1_best, dict):
        selected_source = "hpo_level1"
    else:
        selected_source = "defaults"

    statuses = {str(level1_result.get("status")), str(level2_result.get("status"))}
    if "ok" in statuses:
        status = "ok"
    elif "failed" in statuses:
        status = "failed"
    else:
        status = "skipped"

    trials = 0
    for payload in (level1_result, level2_result):
        value = payload.get("trials")
        if isinstance(value, int):
            trials += int(value)
            continue
        if isinstance(value, list):
            trials += len(value)

    return {
        "status": status,
        "best_params": merged_best_params,
        "best_score": best_score,
        "selected_params_source": selected_source,
        "trials": trials,
    }


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
    search_space_override: dict[str, dict[str, Any]] | None = None,
    base_model_params_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_model_config = model._resolve_model_config()  # noqa: SLF001
    if search_space_override is None:
        raw_space = resolved_model_config.get("tuning_search_space")
    else:
        raw_space = search_space_override

    if not isinstance(raw_space, dict) or not raw_space:
        return {"status": "skipped", "reason": "missing_tuning_search_space"}

    search_space = {
        str(key): dict(spec)
        for key, spec in raw_space.items()
        if isinstance(spec, dict)
    }
    if not search_space:
        return {"status": "skipped", "reason": "missing_tuning_search_space"}

    base_model_params = {
        str(key): value for key, value in dict(base_model_params_override or {}).items()
    }

    n_trials = int(n_trials_override) if n_trials_override is not None else budget.n_trials
    previous_runtime_config = dict(getattr(model, "_runtime_model_config", {}) or {})
    runtime_config = dict(resolved_model_config)
    runtime_config["tuning_search_space"] = search_space
    runtime_config.update(base_model_params)

    try:
        model._runtime_model_config = runtime_config  # noqa: SLF001
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
    finally:
        model._runtime_model_config = previous_runtime_config  # noqa: SLF001


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
        allow_same_gene_overlap=True,
        random_state=seed,
    )
    inner_train = train_df.iloc[split["train"]]
    inner_val = train_df.iloc[split["val"]]

    probe_defaults = PathoLogic("xgboost").defaults
    preprocess_cfg = probe_defaults.get("train", {}).get("preprocess", {})
    if not isinstance(preprocess_cfg, dict):
        preprocess_cfg = {}

    tabnet_missingness_mode = str(
        preprocess_cfg.get("tabnet_missingness_mode", "auto")
    ).strip().lower()
    if tabnet_missingness_mode == "auto":
        preprocess_cfg = dict(preprocess_cfg)
        preprocess_cfg["missing_value_policy"] = "impute"
        preprocess_cfg["impute_strategy"] = str(
            preprocess_cfg.get("tabnet_impute_strategy", "median")
        )
        preprocess_cfg["add_missing_indicators"] = True

    missing_value_policy = str(preprocess_cfg.get("missing_value_policy", "impute"))
    if missing_value_policy not in {"impute", "drop_rows", "none"}:
        raise ValueError(
            "Config field 'preprocess.missing_value_policy' must be one of: "
            "drop_rows, impute, none"
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
