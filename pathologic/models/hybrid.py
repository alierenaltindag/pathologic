"""Hybrid model logic for voting and stacking ensembles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pathologic.models.registry import build_model

_ALIAS_NORMALIZATION = {
    "xgb": "xgboost",
    "rf": "random_forest",
    "lr": "logreg",
}


def normalize_model_alias(alias: str) -> str:
    """Normalize short aliases used in hybrid compositions."""
    normalized = alias.strip().lower()
    return _ALIAS_NORMALIZATION.get(normalized, normalized)


def parse_hybrid_alias(alias: str) -> list[str]:
    """Parse plus-composed alias into normalized member aliases."""
    members = [normalize_model_alias(item) for item in alias.split("+") if item.strip()]
    if len(members) < 2:
        raise ValueError(
            "Hybrid alias must include at least two members separated by '+'."
        )
    if len(set(members)) != len(members):
        raise ValueError("Hybrid alias cannot contain duplicate member models.")
    return members


def _member_params_from_config(model_params: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Extract per-member parameters from hybrid model config."""
    if model_params is None:
        return {}
    members = model_params.get("members", {})
    if not isinstance(members, dict):
        raise ValueError("Config field 'models.<hybrid_alias>.members' must be a mapping.")

    normalized: dict[str, dict[str, Any]] = {}
    for alias_raw, params in members.items():
        if not isinstance(alias_raw, str):
            raise ValueError("Config field 'models.<hybrid_alias>.members' keys must be strings.")
        alias = normalize_model_alias(alias_raw)
        if params is None:
            normalized[alias] = {}
            continue
        if not isinstance(params, dict):
            raise ValueError(
                "Config field 'models.<hybrid_alias>.members.<model>' must be a mapping."
            )
        safe_params = dict(params)
        safe_params.pop("random_state", None)
        normalized[alias] = safe_params
    return normalized


def _meta_model_from_config(model_params: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Extract optional meta-model definition from hybrid config."""
    if model_params is None:
        return "logreg", {}
    raw = model_params.get("meta_model")
    if raw is None:
        return "logreg", {}
    if not isinstance(raw, dict):
        raise ValueError("Config field 'models.<hybrid_alias>.meta_model' must be a mapping.")

    alias_raw = raw.get("alias", "logreg")
    if not isinstance(alias_raw, str) or not alias_raw.strip():
        raise ValueError("Config field 'models.<hybrid_alias>.meta_model.alias' must be a string.")
    alias = normalize_model_alias(alias_raw)

    params_raw = raw.get("params", {})
    if not isinstance(params_raw, dict):
        raise ValueError(
            "Config field 'models.<hybrid_alias>.meta_model.params' must be a mapping."
        )
    params = dict(params_raw)
    params.pop("random_state", None)
    return alias, params


def _strategy_from_config(model_params: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Extract ensemble strategy with normalized value and strategy params."""
    if model_params is None:
        return "soft_voting", {}
    strategy_raw = str(model_params.get("strategy", "soft_voting")).strip().lower()
    strategy_aliases = {
        "soft": "soft_voting",
        "voting": "soft_voting",
        "soft_vote": "soft_voting",
        "hard": "hard_voting",
        "hard_vote": "hard_voting",
    }
    strategy = strategy_aliases.get(strategy_raw, strategy_raw)
    allowed = {"soft_voting", "hard_voting", "stacking", "blending"}
    if strategy not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported hybrid strategy '{strategy}'. Supported: {allowed_list}")

    strategy_params_raw = model_params.get("strategy_params", {})
    if strategy_params_raw is None:
        return strategy, {}
    if not isinstance(strategy_params_raw, dict):
        raise ValueError("Config field 'models.<hybrid_alias>.strategy_params' must be a mapping.")
    return strategy, dict(strategy_params_raw)


def _apply_namespaced_params(
    *,
    raw_params: Mapping[str, Any],
    member_params: dict[str, dict[str, Any]],
    strategy_params: dict[str, Any],
    meta_model_alias: str,
    meta_model_params: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str, dict[str, Any]]:
    """Apply namespaced trial/runtime params into hybrid config sections."""
    for key, value in raw_params.items():
        if key in {"members", "strategy", "strategy_params", "meta_model", "tuning_search_space"}:
            continue

        if key.startswith("member__"):
            parts = key.split("__", 2)
            if len(parts) != 3 or not parts[1] or not parts[2]:
                raise ValueError(
                    "Hybrid member parameter keys must match 'member__<alias>__<param>'."
                )
            alias = normalize_model_alias(parts[1])
            param_name = parts[2]
            member_params.setdefault(alias, {})[param_name] = value
            continue

        if key.startswith("meta__"):
            param_name = key.split("__", 1)[1]
            if not param_name:
                raise ValueError("Hybrid meta parameter keys must match 'meta__<param>'.")
            meta_model_params[param_name] = value
            continue

        if key.startswith("strategy__"):
            param_name = key.split("__", 1)[1]
            if not param_name:
                raise ValueError(
                    "Hybrid strategy parameter keys must match 'strategy__<param>'."
                )
            strategy_params[param_name] = value
            continue

        if key == "meta_model_alias":
            if not isinstance(value, str):
                raise ValueError("meta_model_alias must be a string.")
            meta_model_alias = normalize_model_alias(value)
            continue

        if key == "strategy":
            continue

    return member_params, strategy_params, meta_model_alias, meta_model_params


def _build_estimators(
    *,
    member_aliases: list[str],
    random_state: int,
    member_params: Mapping[str, dict[str, Any]] | None,
) -> list[tuple[str, Any]]:
    """Build estimator list for sklearn ensemble wrappers."""
    params_lookup = dict(member_params or {})
    estimators: list[tuple[str, Any]] = []
    for alias in member_aliases:
        params = dict(params_lookup.get(alias, {}))
        params.pop("random_state", None)
        estimator = build_model(alias, random_state=random_state, **params).estimator
        estimators.append((alias, estimator))
    return estimators


def _resolve_member_feature_indices(
    *,
    member_aliases: list[str],
    member_feature_map: Mapping[str, list[str]] | None,
    feature_names: list[str] | None,
) -> dict[str, np.ndarray] | None:
    """Resolve member-specific feature name lists into column indices."""
    if member_feature_map is None:
        return None
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(
            "Hybrid member feature routing requires a non-empty feature_names list."
        )

    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    resolved: dict[str, np.ndarray] = {}
    for alias in member_aliases:
        raw_features = member_feature_map.get(alias)
        if raw_features is None:
            raw_features = member_feature_map.get(normalize_model_alias(alias))
        if not isinstance(raw_features, list) or not raw_features:
            raise ValueError(
                "Hybrid member feature routing must provide a non-empty list for "
                f"'{alias}'."
            )

        unknown = [str(feature) for feature in raw_features if str(feature) not in name_to_index]
        if unknown:
            raise ValueError(
                "Hybrid member feature routing references unknown features: "
                + ", ".join(unknown)
            )

        indices = np.array(
            [name_to_index[str(feature)] for feature in raw_features],
            dtype=int,
        )
        resolved[alias] = np.unique(indices)
    return resolved


def _parse_member_weights(
    *,
    member_aliases: list[str],
    weights_raw: Any,
) -> list[float] | None:
    if weights_raw is None:
        return None

    if isinstance(weights_raw, list):
        if len(weights_raw) != len(member_aliases):
            raise ValueError("Voting weights length must match member model count.")
        return [float(item) for item in weights_raw]

    if isinstance(weights_raw, Mapping):
        parsed: list[float] = []
        for alias in member_aliases:
            if alias not in weights_raw:
                raise ValueError(
                    "Voting weights mapping must include all members. "
                    f"Missing key: {alias}"
                )
            parsed.append(float(weights_raw[alias]))
        return parsed

    raise ValueError("Voting weights must be a list or mapping.")


class _FeatureSubsetClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper to route only selected columns to estimator."""

    def __init__(self, estimator: Any, feature_indices: np.ndarray) -> None:
        self.estimator = estimator
        self.feature_indices = np.asarray(feature_indices, dtype=int)

    def fit(self, x: np.ndarray, y: np.ndarray) -> _FeatureSubsetClassifier:
        self.estimator.fit(x[:, self.feature_indices], y)
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(x[:, self.feature_indices])).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(x[:, self.feature_indices]))


def _build_member_models(
    *,
    member_aliases: list[str],
    random_state: int,
    member_params: Mapping[str, dict[str, Any]] | None,
) -> list[Any]:
    """Build model wrappers used by manual hybrid ensemble strategies."""
    params_lookup = dict(member_params or {})
    models: list[Any] = []
    for alias in member_aliases:
        params = dict(params_lookup.get(alias, {}))
        params.pop("random_state", None)
        models.append(build_model(alias, random_state=random_state, **params))
    return models


class VotingEnsembleModel:
    """Soft-voting ensemble over registered model aliases."""

    def __init__(
        self,
        *,
        member_aliases: list[str],
        voting: str = "soft",
        weights: list[float] | None = None,
        weighting_policy: str = "auto",
        weighting_objective: str = "f1",
        normalize_weights: bool = True,
        random_state: int = 42,
        member_params: Mapping[str, dict[str, Any]] | None = None,
        member_feature_indices: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        if len(member_aliases) < 2:
            raise ValueError("Voting ensemble requires at least 2 member models.")
        if voting not in {"soft", "hard"}:
            raise ValueError("Voting mode must be one of: soft, hard")
        if weights is not None and len(weights) != len(member_aliases):
            raise ValueError("Voting weights length must match member model count.")
        if weights is not None and any(float(item) < 0.0 for item in weights):
            raise ValueError("Voting weights must be non-negative.")

        self.member_aliases = list(member_aliases)
        self.voting = voting
        self.weights = list(weights) if weights is not None else None
        policy = str(weighting_policy).strip().lower()
        allowed_policies = {"auto", "manual", "equal", "inverse_error", "objective_proportional"}
        if policy not in allowed_policies:
            allowed_str = ", ".join(sorted(allowed_policies))
            raise ValueError(f"Unsupported weighting policy '{weighting_policy}'. Supported: {allowed_str}")
        self.weighting_policy = policy
        self.weighting_objective = str(weighting_objective).strip().lower()
        self.normalize_weights = bool(normalize_weights)
        self._manual_weights = (
            np.asarray(self.weights, dtype=float) if self.weights is not None else None
        )
        self._effective_weights: np.ndarray | None = None
        self._member_weight_scores: dict[str, float] = {}
        self._member_feature_indices = {
            key: np.asarray(value, dtype=int)
            for key, value in dict(member_feature_indices or {}).items()
        }
        self._member_models = _build_member_models(
            member_aliases=member_aliases,
            random_state=random_state,
            member_params=member_params,
        )
        self.model = self

    def _member_input(self, alias: str, x: np.ndarray) -> np.ndarray:
        indices = self._member_feature_indices.get(alias)
        if indices is None:
            return x
        return x[:, indices]

    def fit(self, x: np.ndarray, y: np.ndarray) -> VotingEnsembleModel:
        for alias, model in zip(self.member_aliases, self._member_models, strict=True):
            model.fit(self._member_input(alias, x), y)
        self._effective_weights = self._compute_effective_weights(x, y)
        return self

    def fine_tune(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        freeze_layers: str | None = None,
        learning_rate: float | None = None,
        epochs: int | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ) -> VotingEnsembleModel:
        del freeze_layers, learning_rate, epochs, scheduler_config
        return self.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.voting == "soft":
            proba = self.predict_proba(x)
            return (proba[:, -1] >= 0.5).astype(int)

        votes = np.column_stack(
            [
                np.asarray(model.predict(self._member_input(alias, x))).reshape(-1)
                for alias, model in zip(self.member_aliases, self._member_models, strict=True)
            ]
        )
        weights = self._resolved_weight_array()
        if weights is None:
            vote_score = votes.mean(axis=1)
        else:
            vote_score = (votes * weights).sum(axis=1) / max(weights.sum(), 1e-12)
        return (vote_score >= 0.5).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        for alias, model in zip(self.member_aliases, self._member_models, strict=True):
            proba = np.asarray(model.predict_proba(self._member_input(alias, x)))
            score = proba[:, -1] if proba.ndim > 1 else proba
            columns.append(score.reshape(-1, 1))

        matrix = np.hstack(columns)
        weights = self._resolved_weight_array()
        if weights is None:
            positive = matrix.mean(axis=1)
        else:
            positive = (matrix * weights).sum(axis=1) / max(weights.sum(), 1e-12)

        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def effective_member_weights(self) -> dict[str, float]:
        weights = self._resolved_weight_array()
        if weights is None:
            equal = np.full(len(self.member_aliases), 1.0 / max(len(self.member_aliases), 1), dtype=float)
            return {
                alias: float(value)
                for alias, value in zip(self.member_aliases, equal, strict=True)
            }

        normalized = self._normalize_weight_array(weights)
        return {
            alias: float(value)
            for alias, value in zip(self.member_aliases, normalized, strict=True)
        }

    def member_weight_scores(self) -> dict[str, float]:
        return dict(self._member_weight_scores)

    def _resolved_weight_array(self) -> np.ndarray | None:
        if self._effective_weights is not None:
            return np.asarray(self._effective_weights, dtype=float)
        if self._manual_weights is not None:
            return np.asarray(self._manual_weights, dtype=float)
        return None

    def _normalize_weight_array(self, weights: np.ndarray) -> np.ndarray:
        clipped = np.asarray(weights, dtype=float)
        clipped = np.where(np.isfinite(clipped), clipped, 0.0)
        clipped = np.clip(clipped, 0.0, None)
        total = float(clipped.sum())
        if total <= 0.0:
            return np.full(len(self.member_aliases), 1.0 / max(len(self.member_aliases), 1), dtype=float)
        return clipped / total

    def _score_member(
        self,
        *,
        model: Any,
        x_member: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        objective = self.weighting_objective
        if objective == "roc_auc":
            if np.unique(y_true).size < 2:
                return 0.5
            probs = np.asarray(model.predict_proba(x_member))
            score = probs if probs.ndim == 1 else probs[:, -1]
            return float(roc_auc_score(y_true, score))

        y_pred = np.asarray(model.predict(x_member)).reshape(-1)
        if objective == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))
        if objective == "precision":
            return float(precision_score(y_true, y_pred, zero_division=0))
        if objective == "recall":
            return float(recall_score(y_true, y_pred, zero_division=0))
        if objective in {"accuracy", "acc"}:
            return float(accuracy_score(y_true, y_pred))
        return float(f1_score(y_true, y_pred, zero_division=0))

    def _compute_effective_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        policy = self.weighting_policy
        if policy == "auto":
            policy = "manual" if self._manual_weights is not None else "equal"

        if policy == "manual":
            if self._manual_weights is None:
                return self._normalize_weight_array(np.ones(len(self.member_aliases), dtype=float))
            return (
                self._normalize_weight_array(self._manual_weights)
                if self.normalize_weights
                else np.asarray(self._manual_weights, dtype=float)
            )

        if policy == "equal":
            if self._manual_weights is not None:
                return (
                    self._normalize_weight_array(self._manual_weights)
                    if self.normalize_weights
                    else np.asarray(self._manual_weights, dtype=float)
                )
            return self._normalize_weight_array(np.ones(len(self.member_aliases), dtype=float))

        if policy in {"objective_proportional", "inverse_error"}:
            raw_scores: list[float] = []
            self._member_weight_scores = {}
            for alias, model in zip(self.member_aliases, self._member_models, strict=True):
                score = self._score_member(
                    model=model,
                    x_member=self._member_input(alias, x),
                    y_true=y,
                )
                bounded = float(np.clip(score, 0.0, 1.0))
                self._member_weight_scores[alias] = bounded
                raw_scores.append(bounded)

            values = np.asarray(raw_scores, dtype=float)
            if policy == "inverse_error":
                errors = np.maximum(1.0 - values, 1e-6)
                weights = 1.0 / errors
            else:
                weights = np.maximum(values, 1e-6)

            if np.isfinite(weights).sum() != len(weights):
                return self._normalize_weight_array(np.ones(len(self.member_aliases), dtype=float))
            return self._normalize_weight_array(weights)

        return self._normalize_weight_array(np.ones(len(self.member_aliases), dtype=float))


class StackingEnsembleModel:
    """Stacking ensemble with logistic regression meta-learner."""

    def __init__(
        self,
        *,
        member_aliases: list[str],
        random_state: int = 42,
        cv: int = 3,
        meta_model_alias: str = "logreg",
        meta_model_params: Mapping[str, Any] | None = None,
        member_params: Mapping[str, dict[str, Any]] | None = None,
        member_feature_indices: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        if len(member_aliases) < 2:
            raise ValueError("Stacking ensemble requires at least 2 member models.")

        estimators = _build_estimators(
            member_aliases=member_aliases,
            random_state=random_state,
            member_params=member_params,
        )
        meta_params = dict(meta_model_params or {})
        meta_params.pop("random_state", None)
        meta_estimator = build_model(
            normalize_model_alias(meta_model_alias),
            random_state=random_state,
            **meta_params,
        ).estimator
        if member_feature_indices is not None:
            wrapped_estimators: list[tuple[str, Any]] = []
            for alias, estimator in estimators:
                indices = member_feature_indices.get(alias)
                if indices is None:
                    wrapped_estimators.append((alias, estimator))
                    continue
                wrapped_estimators.append(
                    (alias, _FeatureSubsetClassifier(estimator, np.asarray(indices, dtype=int)))
                )
            estimators = wrapped_estimators

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_estimator,
            cv=cv,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> StackingEnsembleModel:
        self.model.fit(x, y)
        return self

    def fine_tune(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        freeze_layers: str | None = None,
        learning_rate: float | None = None,
        epochs: int | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ) -> StackingEnsembleModel:
        del freeze_layers, learning_rate, epochs, scheduler_config
        return self.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(x)).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x))


class BlendingEnsembleModel:
    """Blending ensemble with holdout-trained meta-learner."""

    def __init__(
        self,
        *,
        member_aliases: list[str],
        random_state: int = 42,
        member_params: Mapping[str, dict[str, Any]] | None = None,
        blend_size: float = 0.2,
        meta_model_alias: str = "logreg",
        meta_model_params: Mapping[str, Any] | None = None,
        member_feature_indices: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        if len(member_aliases) < 2:
            raise ValueError("Blending ensemble requires at least 2 member models.")
        if blend_size <= 0.0 or blend_size >= 0.5:
            raise ValueError("blend_size must satisfy 0 < blend_size < 0.5")

        self.member_aliases = list(member_aliases)
        self.random_state = int(random_state)
        self.blend_size = float(blend_size)
        self._member_feature_indices = {
            key: np.asarray(value, dtype=int)
            for key, value in dict(member_feature_indices or {}).items()
        }
        self._member_models: list[Any] = [
            build_model(
                alias,
                random_state=self.random_state,
                **dict((member_params or {}).get(alias, {})),
            )
            for alias in self.member_aliases
        ]
        meta_params = dict(meta_model_params or {})
        meta_params.pop("random_state", None)
        self._meta_model = build_model(
            normalize_model_alias(meta_model_alias),
            random_state=self.random_state,
            **meta_params,
        )

    def _member_input(self, alias: str, x: np.ndarray) -> np.ndarray:
        indices = self._member_feature_indices.get(alias)
        if indices is None:
            return x
        return x[:, indices]

    def _resolve_blend_stratify_target(self, y: np.ndarray) -> np.ndarray | None:
        classes, counts = np.unique(y, return_counts=True)
        if classes.size <= 1:
            return None

        n_samples = int(y.shape[0])
        test_size = self._effective_blend_test_size(y)
        train_size = n_samples - test_size

        # Stratified holdout requires enough samples on both sides for each class.
        if test_size < classes.size or train_size < classes.size:
            return None
        if int(np.min(counts)) < 2:
            return None
        return y

    def _effective_blend_test_size(self, y: np.ndarray) -> int:
        n_samples = int(y.shape[0])
        n_classes = int(np.unique(y).size)
        requested = int(np.ceil(n_samples * self.blend_size))

        minimum = n_classes if n_classes > 1 else 1
        maximum = max(1, n_samples - max(1, n_classes))
        return min(max(requested, minimum), maximum)

    def fit(self, x: np.ndarray, y: np.ndarray) -> BlendingEnsembleModel:
        test_size = self._effective_blend_test_size(y)
        stratify_target = self._resolve_blend_stratify_target(y)
        x_base, x_blend, y_base, y_blend = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_target,
        )

        blend_features: list[np.ndarray] = []
        for alias, model in zip(self.member_aliases, self._member_models, strict=True):
            model.fit(self._member_input(alias, x_base), y_base)
            proba = np.asarray(model.predict_proba(self._member_input(alias, x_blend)))
            score = proba[:, -1] if proba.ndim > 1 else proba
            blend_features.append(score.reshape(-1, 1))

        meta_x = np.hstack(blend_features)
        meta_y = y_blend

        # If holdout accidentally collapses to one class, fall back to base slice.
        if np.unique(meta_y).size < 2:
            fallback_features: list[np.ndarray] = []
            for alias, model in zip(self.member_aliases, self._member_models, strict=True):
                proba = np.asarray(model.predict_proba(self._member_input(alias, x_base)))
                score = proba[:, -1] if proba.ndim > 1 else proba
                fallback_features.append(score.reshape(-1, 1))
            meta_x = np.hstack(fallback_features)
            meta_y = y_base

        self._meta_model.fit(meta_x, meta_y)
        return self

    def fine_tune(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        freeze_layers: str | None = None,
        learning_rate: float | None = None,
        epochs: int | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ) -> BlendingEnsembleModel:
        del freeze_layers, learning_rate, epochs, scheduler_config
        return self.fit(x, y)

    def _meta_features(self, x: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        for alias, model in zip(self.member_aliases, self._member_models, strict=True):
            proba = np.asarray(model.predict_proba(self._member_input(alias, x)))
            score = proba[:, -1] if proba.ndim > 1 else proba
            columns.append(score.reshape(-1, 1))
        return np.hstack(columns)

    def predict(self, x: np.ndarray) -> np.ndarray:
        meta_x = self._meta_features(x)
        return np.asarray(self._meta_model.predict(meta_x)).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        meta_x = self._meta_features(x)
        return np.asarray(self._meta_model.predict_proba(meta_x))


def build_default_hybrid(
    alias: str,
    *,
    random_state: int = 42,
    model_params: Mapping[str, Any] | None = None,
) -> Any:
    """Build hybrid ensemble from alias and optional runtime strategy config."""
    member_aliases = parse_hybrid_alias(alias)

    raw_params = dict(model_params or {})
    raw_feature_names = raw_params.pop("feature_names", None)
    raw_member_feature_map = raw_params.pop("member_feature_map", None)
    member_feature_map = (
        {
            normalize_model_alias(str(alias)): [str(feature) for feature in features]
            for alias, features in raw_member_feature_map.items()
            if isinstance(features, list)
        }
        if isinstance(raw_member_feature_map, Mapping)
        else None
    )

    member_params = _member_params_from_config(raw_params)
    strategy, strategy_params = _strategy_from_config(raw_params)
    meta_model_alias, meta_model_params = _meta_model_from_config(raw_params)
    member_params, strategy_params, meta_model_alias, meta_model_params = _apply_namespaced_params(
        raw_params=raw_params,
        member_params=member_params,
        strategy_params=strategy_params,
        meta_model_alias=meta_model_alias,
        meta_model_params=meta_model_params,
    )

    for alias_name in member_aliases:
        member_params.setdefault(alias_name, {})

    member_feature_indices = _resolve_member_feature_indices(
        member_aliases=member_aliases,
        member_feature_map=member_feature_map,
        feature_names=[str(name) for name in raw_feature_names]
        if isinstance(raw_feature_names, list)
        else None,
    )

    def _weights_from_strategy_params() -> list[float] | None:
        weights_raw = strategy_params.get("weights")
        if weights_raw is not None:
            return _parse_member_weights(
                member_aliases=member_aliases,
                weights_raw=weights_raw,
            )

        if "weight_ratio" not in strategy_params:
            return None
        if len(member_aliases) != 2:
            raise ValueError("strategy__weight_ratio is supported only for 2-member hybrids.")

        ratio = float(strategy_params["weight_ratio"])
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError("strategy__weight_ratio must satisfy 0 < ratio < 1.")
        return [ratio, 1.0 - ratio]

    if strategy == "soft_voting":
        weights = _weights_from_strategy_params()
        return VotingEnsembleModel(
            member_aliases=member_aliases,
            voting="soft",
            weights=weights,
            weighting_policy=str(strategy_params.get("weighting_policy", "auto")),
            weighting_objective=str(strategy_params.get("weighting_objective", "f1")),
            normalize_weights=bool(strategy_params.get("normalize_weights", True)),
            random_state=random_state,
            member_params=member_params,
            member_feature_indices=member_feature_indices,
        )

    if strategy == "hard_voting":
        weights = _weights_from_strategy_params()
        return VotingEnsembleModel(
            member_aliases=member_aliases,
            voting="hard",
            weights=weights,
            weighting_policy=str(strategy_params.get("weighting_policy", "auto")),
            weighting_objective=str(strategy_params.get("weighting_objective", "f1")),
            normalize_weights=bool(strategy_params.get("normalize_weights", True)),
            random_state=random_state,
            member_params=member_params,
            member_feature_indices=member_feature_indices,
        )

    if strategy == "stacking":
        cv_value = int(strategy_params.get("cv", 3))
        return StackingEnsembleModel(
            member_aliases=member_aliases,
            random_state=random_state,
            cv=cv_value,
            meta_model_alias=meta_model_alias,
            meta_model_params=meta_model_params,
            member_params=member_params,
            member_feature_indices=member_feature_indices,
        )

    blend_size = float(strategy_params.get("blend_size", 0.2))
    return BlendingEnsembleModel(
        member_aliases=member_aliases,
        random_state=random_state,
        member_params=member_params,
        blend_size=blend_size,
        meta_model_alias=meta_model_alias,
        meta_model_params=meta_model_params,
        member_feature_indices=member_feature_indices,
    )
