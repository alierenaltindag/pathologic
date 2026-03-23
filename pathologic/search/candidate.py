"""Candidate construction helpers for search workflows."""

from __future__ import annotations

import argparse
from itertools import combinations
from typing import Any, Mapping, cast

from pathologic import PathoLogic
from pathologic.models import list_registered_models
from pathologic.search.spec import CandidateSpec, HYBRID_TUNING_PROFILE_DEFAULTS

_ALIAS_NORMALIZATION = {
    "xgb": "xgboost",
    "xgnet": "xgboost",
}

_REGULARIZATION_PARAM_KEYS: dict[str, set[str]] = {
    "xgboost": {"reg_alpha", "reg_lambda"},
    "lightgbm": {"reg_alpha", "reg_lambda"},
    "catboost": {"l2_leaf_reg"},
    "tabnet": {"weight_decay"},
    "mlp": {"alpha"},
}


def _normalize_alias(alias: str) -> str:
    normalized = str(alias).strip().lower()
    return _ALIAS_NORMALIZATION.get(normalized, normalized)


def parse_regularization_models(raw: str | list[str] | None) -> list[str] | None:
    """Parse and normalize regularization model scope list."""
    if raw is None:
        return None
    if isinstance(raw, str):
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, list):
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    else:
        return None

    if not tokens:
        return None

    parsed: list[str] = []
    for token in tokens:
        alias = _normalize_alias(token)
        if alias not in parsed:
            parsed.append(alias)
    return parsed


def _regularization_keys_for_alias(alias: str) -> set[str]:
    return set(_REGULARIZATION_PARAM_KEYS.get(_normalize_alias(alias), set()))


def _regularization_space_for_alias(alias: str) -> dict[str, dict[str, Any]]:
    """Collect regularization-only search specs for one model alias."""
    full_space = model_tuning_search_space(alias)
    key_set = _regularization_keys_for_alias(alias)
    if not key_set:
        return {}
    return {
        key: dict(spec)
        for key, spec in full_space.items()
        if key in key_set and isinstance(spec, dict)
    }


def build_member_regularization_tuning_search_space(
    *,
    members: tuple[str, ...],
    regularization_models: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build namespaced regularization search space for hybrid members."""
    enabled_models = {
        _normalize_alias(alias) for alias in regularization_models
    } if regularization_models else set(_REGULARIZATION_PARAM_KEYS)

    search_space: dict[str, dict[str, Any]] = {}
    for member in members:
        normalized_member = _normalize_alias(member)
        if normalized_member not in enabled_models:
            continue
        member_reg_space = _regularization_space_for_alias(normalized_member)
        for key, spec in member_reg_space.items():
            search_space[f"member__{normalized_member}__{key}"] = dict(spec)
    return search_space


def strip_regularization_search_space(
    *,
    search_space: Mapping[str, dict[str, Any]],
    members: tuple[str, ...],
    regularization_models: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Remove regularization parameters from flat or namespaced search spaces."""
    enabled_models = {
        _normalize_alias(alias) for alias in regularization_models
    } if regularization_models else set(_REGULARIZATION_PARAM_KEYS)
    member_set = {_normalize_alias(alias) for alias in members}

    def _should_drop(key: str) -> bool:
        if key.startswith("member__"):
            parts = key.split("__", 2)
            if len(parts) != 3:
                return False
            alias = _normalize_alias(parts[1])
            param = parts[2]
            if alias not in enabled_models:
                return False
            return param in _regularization_keys_for_alias(alias)

        if len(member_set) == 1:
            alias = next(iter(member_set))
            if alias in enabled_models and key in _regularization_keys_for_alias(alias):
                return True
        return False

    return {
        str(key): dict(spec)
        for key, spec in search_space.items()
        if isinstance(spec, dict) and not _should_drop(str(key))
    }


def model_tuning_search_space(alias: str) -> dict[str, dict[str, Any]]:
    try:
        probe = PathoLogic(alias)
        config = probe._resolve_model_config()  # noqa: SLF001
        raw = config.get("tuning_search_space")
        if isinstance(raw, dict) and raw:
            return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}
    except Exception:
        return {}
    return {}


def build_hybrid_tuning_search_space(members: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    search_space: dict[str, dict[str, Any]] = {}
    for member in members:
        member_space = model_tuning_search_space(member)
        for key, spec in member_space.items():
            search_space[f"member__{member}__{key}"] = dict(spec)
    return search_space


def build_pair_tuning_search_space(member_a: str, member_b: str) -> dict[str, dict[str, Any]]:
    search_space: dict[str, dict[str, Any]] = {}
    search_space = build_hybrid_tuning_search_space((member_a, member_b))
    return search_space


def build_hybrid_strategy_tuning_search_space(
    *,
    budget_profile: str = "aggressive",
    search_defaults: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Search space for hybrid strategy and strategy-specific parameters."""
    profile_name = str(budget_profile).strip().lower()
    base_profile = dict(
        HYBRID_TUNING_PROFILE_DEFAULTS.get(profile_name, HYBRID_TUNING_PROFILE_DEFAULTS["aggressive"])
    )

    profile_overrides: dict[str, Any] = {}
    if isinstance(search_defaults, Mapping):
        tuning_space_raw = search_defaults.get("hybrid_tuning_space")
        if isinstance(tuning_space_raw, Mapping):
            profile_raw = tuning_space_raw.get(profile_name)
            if isinstance(profile_raw, Mapping):
                profile_overrides = dict(profile_raw)

    merged = {**base_profile, **profile_overrides}

    strategy_values = merged.get("strategy")
    if not isinstance(strategy_values, list) or not strategy_values:
        strategy_values = base_profile["strategy"]
    weighting_policy_values = merged.get("weighting_policy")
    if not isinstance(weighting_policy_values, list) or not weighting_policy_values:
        weighting_policy_values = base_profile["weighting_policy"]
    meta_model_values = merged.get("meta_model_alias")
    if not isinstance(meta_model_values, list) or not meta_model_values:
        meta_model_values = base_profile["meta_model_alias"]

    weight_ratio_cfg = merged.get("weight_ratio")
    if not isinstance(weight_ratio_cfg, Mapping):
        weight_ratio_cfg = cast(Mapping[str, Any], base_profile["weight_ratio"])
    cv_cfg = merged.get("cv")
    if not isinstance(cv_cfg, Mapping):
        cv_cfg = cast(Mapping[str, Any], base_profile["cv"])
    blend_size_cfg = merged.get("blend_size")
    if not isinstance(blend_size_cfg, Mapping):
        blend_size_cfg = cast(Mapping[str, Any], base_profile["blend_size"])

    return {
        "strategy": {
            "type": "categorical",
            "values": [str(value) for value in strategy_values],
        },
        "strategy__weighting_policy": {
            "type": "categorical",
            "values": [str(value) for value in weighting_policy_values],
        },
        "strategy__weight_ratio": {
            "type": "float",
            "low": float(weight_ratio_cfg.get("low", 0.1)),
            "high": float(weight_ratio_cfg.get("high", 0.9)),
        },
        "meta_model_alias": {
            "type": "categorical",
            "values": [str(value) for value in meta_model_values],
        },
        "strategy__cv": {
            "type": "int",
            "low": int(cv_cfg.get("low", 2)),
            "high": int(cv_cfg.get("high", 5)),
        },
        "strategy__blend_size": {
            "type": "float",
            "low": float(blend_size_cfg.get("low", 0.1)),
            "high": float(blend_size_cfg.get("high", 0.35)),
        },
    }


def build_candidate_specs(
    *,
    include_models: list[str] | None,
    explicit_candidates: list[str] | None = None,
    exclude_models: list[str] | None,
    include_hybrids: bool,
    max_candidates: int | None,
    hybrid_tune_strategy_and_params: bool = False,
    hybrid_tuning_search_space: Mapping[str, dict[str, Any]] | None = None,
    max_hybrid_combination_size: int = 2,
    regularization_profile: str = "auto",
    regularization_models: list[str] | None = None,
) -> list[CandidateSpec]:
    profile_name = str(regularization_profile).strip().lower()
    if profile_name not in {"auto", "off"}:
        raise ValueError("regularization_profile must be one of: auto, off")

    normalized_regularization_models = parse_regularization_models(regularization_models)

    available = sorted(list_registered_models())
    explicit_entries: list[tuple[str, ...]] = []
    if explicit_candidates:
        for raw_item in explicit_candidates:
            token = str(raw_item).strip().lower()
            if not token:
                continue
            members = tuple(_normalize_alias(part) for part in token.split("+") if part.strip())
            if not members:
                continue
            if len(set(members)) != len(members):
                raise ValueError(f"Explicit candidate has duplicate members: {raw_item}")
            unknown = [member for member in members if member not in available]
            if unknown:
                unknown_text = ", ".join(sorted(unknown))
                raise ValueError(
                    f"Unknown model alias in explicit candidate '{raw_item}': {unknown_text}"
                )

            explicit_entries.append(members)

        if not explicit_entries:
            raise ValueError("No valid explicit candidates parsed from --only-candidates.")

    include_set = set(include_models) if include_models else set(available)
    exclude_set = set(exclude_models or [])
    singles = [alias for alias in available if alias in include_set and alias not in exclude_set]

    candidates: list[CandidateSpec] = []

    def _single_search_space(alias: str) -> dict[str, dict[str, Any]]:
        single_space = model_tuning_search_space(alias)
        if profile_name == "off":
            single_space = strip_regularization_search_space(
                search_space=single_space,
                members=(alias,),
                regularization_models=None,
            )
        elif normalized_regularization_models is not None:
            single_space = strip_regularization_search_space(
                search_space=single_space,
                members=(alias,),
                regularization_models=None,
            )
            if _normalize_alias(alias) in {
                _normalize_alias(item) for item in normalized_regularization_models
            }:
                single_space.update(_regularization_space_for_alias(alias))
        return single_space

    def _hybrid_search_space(members: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        hybrid_space = build_hybrid_tuning_search_space(members)
        if profile_name == "off":
            hybrid_space = strip_regularization_search_space(
                search_space=hybrid_space,
                members=members,
                regularization_models=None,
            )
        elif normalized_regularization_models is not None:
            hybrid_space = strip_regularization_search_space(
                search_space=hybrid_space,
                members=members,
                regularization_models=None,
            )
            hybrid_space.update(
                build_member_regularization_tuning_search_space(
                    members=members,
                    regularization_models=normalized_regularization_models,
                )
            )

        if hybrid_tune_strategy_and_params:
            profile_space = {
                str(key): dict(value)
                for key, value in dict(hybrid_tuning_search_space or {}).items()
                if isinstance(value, dict)
            }
            if not profile_space:
                profile_space = build_hybrid_strategy_tuning_search_space()
            hybrid_space.update(profile_space)

        return hybrid_space

    if explicit_entries:
        seen_names: set[str] = set()
        for members in explicit_entries:
            candidate_name = "+".join(members)
            if candidate_name in seen_names:
                continue
            seen_names.add(candidate_name)

            if len(members) == 1:
                alias = members[0]
                candidates.append(
                    CandidateSpec(
                        name=alias,
                        kind="single",
                        members=(alias,),
                        tuning_search_space=_single_search_space(alias),
                    )
                )
                continue

            candidates.append(
                CandidateSpec(
                    name=candidate_name,
                    kind="hybrid_pair",
                    members=members,
                    tuning_search_space=_hybrid_search_space(members),
                )
            )

        if max_candidates is not None and max_candidates > 0:
            candidates = candidates[:max_candidates]
        return candidates

    for alias in singles:
        candidates.append(
            CandidateSpec(
                name=alias,
                kind="single",
                members=(alias,),
                tuning_search_space=_single_search_space(alias),
            )
        )

    if include_hybrids and singles:
        requested_max_size = int(max_hybrid_combination_size)
        upper_size = min(requested_max_size, len(singles))
        if upper_size < 2:
            upper_size = 1

        for combination_size in range(2, upper_size + 1):
            for members in combinations(singles, combination_size):
                pair_name = "+".join(members)
                pair_search_space = _hybrid_search_space(tuple(members))
                candidates.append(
                    CandidateSpec(
                        name=pair_name,
                        kind="hybrid_pair",
                        members=tuple(members),
                        tuning_search_space=pair_search_space,
                    )
                )

    if max_candidates is not None and max_candidates > 0:
        candidates = candidates[:max_candidates]
    return candidates


def parse_hybrid_weights(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        return None
    return values


def resolve_hybrid_strategy_config(
    candidate: CandidateSpec,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """Resolve effective hybrid strategy settings for a pair candidate."""
    strategy_name = str(getattr(args, "hybrid_strategy", "soft_voting"))
    strategy_params: dict[str, Any] = {
        "weighting_policy": str(getattr(args, "hybrid_weighting_policy", "auto")),
        "weighting_objective": str(getattr(args, "hybrid_weighting_objective", "f1")),
        "normalize_weights": not bool(getattr(args, "disable_hybrid_normalize_weights", False)),
    }
    weights = parse_hybrid_weights(getattr(args, "hybrid_weights", None))
    if weights is not None:
        if len(weights) != len(candidate.members):
            raise ValueError(
                "--hybrid-weights must provide one value per member in pair candidates. "
                f"Expected {len(candidate.members)}, got {len(weights)}."
            )
        strategy_params["weights"] = weights

    if strategy_name == "stacking":
        strategy_params["cv"] = int(getattr(args, "hybrid_stacking_cv", 3))
    elif strategy_name == "blending":
        strategy_params["blend_size"] = float(getattr(args, "hybrid_blend_size", 0.2))

    return strategy_name, strategy_params


def resolve_hybrid_config_for_report(
    *,
    candidate: CandidateSpec,
    args: argparse.Namespace,
    selected_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve effective hybrid config after applying tuned strategy overrides."""
    strategy_name, strategy_params = resolve_hybrid_strategy_config(candidate, args)
    resolved_meta_model = str(getattr(args, "hybrid_meta_model", "logreg"))

    if isinstance(selected_params, dict):
        if "strategy" in selected_params:
            strategy_name = str(selected_params["strategy"])
        if "meta_model_alias" in selected_params:
            resolved_meta_model = str(selected_params["meta_model_alias"])
        for key, value in selected_params.items():
            if not str(key).startswith("strategy__"):
                continue
            param_name = str(key).split("__", 1)[1]
            if param_name:
                strategy_params[param_name] = value

    return {
        "strategy": strategy_name,
        "params": strategy_params,
        "meta_model": resolved_meta_model,
    }


def model_for_candidate_with_hybrid_config(candidate: CandidateSpec, args: argparse.Namespace) -> PathoLogic:
    if candidate.kind == "single":
        return PathoLogic(candidate.name)

    builder = PathoLogic.builder().add_model(candidate.members[0]).add_model(candidate.members[1])
    strategy_name, strategy_params = resolve_hybrid_strategy_config(candidate, args)

    if strategy_name == "stacking":
        builder = builder.meta_model(str(getattr(args, "hybrid_meta_model", "logreg")))
    elif strategy_name == "blending":
        builder = builder.meta_model(str(getattr(args, "hybrid_meta_model", "logreg")))

    builder = builder.strategy(strategy_name, **strategy_params)
    if candidate.tuning_search_space:
        builder = builder.tuning_search_space(candidate.tuning_search_space)
    return PathoLogic.from_builder(builder)


def model_for_candidate(candidate: CandidateSpec) -> PathoLogic:
    default_args = argparse.Namespace(
        hybrid_strategy="soft_voting",
        hybrid_weights=None,
        hybrid_weighting_policy="auto",
        hybrid_weighting_objective="f1",
        disable_hybrid_normalize_weights=False,
        hybrid_meta_model="logreg",
        hybrid_stacking_cv=3,
        hybrid_blend_size=0.2,
    )
    return model_for_candidate_with_hybrid_config(candidate, default_args)
