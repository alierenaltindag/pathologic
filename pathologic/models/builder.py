"""Fluent builder for runtime-configurable hybrid ensembles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ALLOWED_STRATEGIES = {"soft_voting", "hard_voting", "stacking", "blending"}


def _normalize_strategy(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "soft": "soft_voting",
        "soft_vote": "soft_voting",
        "voting": "soft_voting",
        "hard": "hard_voting",
        "hard_vote": "hard_voting",
    }
    return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class EnsembleSpec:
    """Immutable runtime ensemble specification."""

    member_aliases: tuple[str, ...]
    strategy: str
    member_params: dict[str, dict[str, Any]]
    meta_model_alias: str | None
    meta_model_params: dict[str, Any]
    strategy_params: dict[str, Any]
    tuning_search_space: dict[str, dict[str, Any]]

    @property
    def alias(self) -> str:
        return "+".join(self.member_aliases)

    def to_model_config(self) -> dict[str, Any]:
        """Serialize spec into model config payload consumed by `create_model`."""
        payload: dict[str, Any] = {
            "strategy": self.strategy,
            "members": {name: dict(params) for name, params in self.member_params.items()},
        }
        if self.meta_model_alias is not None:
            payload["meta_model"] = {
                "alias": self.meta_model_alias,
                "params": dict(self.meta_model_params),
            }
        if self.strategy_params:
            payload["strategy_params"] = dict(self.strategy_params)
        if self.tuning_search_space:
            payload["tuning_search_space"] = {
                key: dict(value) for key, value in self.tuning_search_space.items()
            }
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Serialize spec for reporting/export."""
        return {
            "alias": self.alias,
            "member_aliases": list(self.member_aliases),
            "strategy": self.strategy,
            "member_params": {k: dict(v) for k, v in self.member_params.items()},
            "meta_model": (
                {
                    "alias": self.meta_model_alias,
                    "params": dict(self.meta_model_params),
                }
                if self.meta_model_alias is not None
                else None
            ),
            "strategy_params": dict(self.strategy_params),
            "tuning_search_space": {
                key: dict(value) for key, value in self.tuning_search_space.items()
            },
        }

    def export(self, path: str) -> None:
        """Export spec snapshot to JSON for reproducibility."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=True, indent=2, sort_keys=True)


class ModelBuilder:
    """Fluent builder for dynamic hybrid model definitions."""

    def __init__(self) -> None:
        self._members: list[str] = []
        self._member_params: dict[str, dict[str, Any]] = {}
        self._strategy = "soft_voting"
        self._strategy_params: dict[str, Any] = {}
        self._meta_model_alias: str | None = None
        self._meta_model_params: dict[str, Any] = {}
        self._tuning_search_space: dict[str, dict[str, Any]] = {}

    def add_model(self, alias: str, **params: Any) -> ModelBuilder:
        """Add base model alias with optional constructor parameters."""
        normalized = alias.strip().lower()
        if not normalized:
            raise ValueError("Model alias must be a non-empty string.")
        if normalized in self._members:
            raise ValueError(f"Duplicate member model alias: {normalized}")
        self._members.append(normalized)
        self._member_params[normalized] = dict(params)
        return self

    def strategy(self, name: str, **params: Any) -> ModelBuilder:
        """Set ensemble strategy and strategy-specific parameters."""
        normalized = _normalize_strategy(name)
        if normalized not in _ALLOWED_STRATEGIES:
            allowed = ", ".join(sorted(_ALLOWED_STRATEGIES))
            raise ValueError(f"Unsupported strategy '{name}'. Supported: {allowed}")
        self._strategy = normalized
        self._strategy_params = dict(params)
        return self

    def member_weights(
        self,
        weights: list[float] | dict[str, float],
        *,
        normalize: bool = True,
    ) -> ModelBuilder:
        """Set per-member voting weights for soft/hard hybrid strategies."""
        if isinstance(weights, list):
            parsed = [float(value) for value in weights]
        elif isinstance(weights, dict):
            parsed = {str(key).strip().lower(): float(value) for key, value in weights.items()}
        else:
            raise ValueError("weights must be a list or mapping.")

        self._strategy_params["weights"] = parsed
        self._strategy_params["normalize_weights"] = bool(normalize)
        return self

    def dynamic_weighting(
        self,
        policy: str,
        *,
        objective: str = "f1",
    ) -> ModelBuilder:
        """Configure dynamic member weighting policy used by voting ensembles."""
        normalized = str(policy).strip().lower()
        allowed = {"auto", "manual", "equal", "inverse_error", "objective_proportional"}
        if normalized not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise ValueError(f"Unsupported weighting policy '{policy}'. Supported: {allowed_text}")

        self._strategy_params["weighting_policy"] = normalized
        self._strategy_params["weighting_objective"] = str(objective).strip().lower()
        return self

    def meta_model(self, alias: str, **params: Any) -> ModelBuilder:
        """Set meta-model alias and parameters for stacking/blending."""
        normalized = alias.strip().lower()
        if not normalized:
            raise ValueError("Meta model alias must be a non-empty string.")
        self._meta_model_alias = normalized
        self._meta_model_params = dict(params)
        return self

    def tuning_search_space(self, search_space: dict[str, dict[str, Any]]) -> ModelBuilder:
        """Set optional tuning search space using namespaced parameter keys."""
        if not isinstance(search_space, dict):
            raise ValueError("search_space must be a mapping.")
        self._tuning_search_space = {key: dict(value) for key, value in search_space.items()}
        return self

    def build(self) -> EnsembleSpec:
        """Validate and freeze builder state into an immutable spec."""
        if len(self._members) < 2:
            raise ValueError("Hybrid model requires at least 2 member models.")

        if self._strategy in {"stacking", "blending"} and self._meta_model_alias is None:
            raise ValueError(
                "Strategy requires a meta model. Call meta_model(...) for stacking/blending."
            )

        return EnsembleSpec(
            member_aliases=tuple(self._members),
            strategy=self._strategy,
            member_params={key: dict(value) for key, value in self._member_params.items()},
            meta_model_alias=self._meta_model_alias,
            meta_model_params=dict(self._meta_model_params),
            strategy_params=dict(self._strategy_params),
            tuning_search_space={
                key: dict(value) for key, value in self._tuning_search_space.items()
            },
        )
