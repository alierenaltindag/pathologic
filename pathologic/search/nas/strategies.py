"""NAS candidate generation strategies with budget-aware fidelity controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class NASCandidate:
    """Search candidate payload produced by NAS strategies."""

    candidate_id: str
    params: dict[str, Any]
    fidelity: int
    metadata: dict[str, Any]


class NASStrategy(Protocol):
    """Protocol for NAS candidate generation strategies."""

    name: str

    def generate(
        self,
        *,
        search_space: dict[str, dict[str, Any]],
        n_candidates: int,
        rng: np.random.Generator,
        budget: dict[str, Any] | None = None,
    ) -> list[NASCandidate]:
        """Generate NAS candidates under strategy-specific constraints."""


def _sample_param(spec: dict[str, Any], rng: np.random.Generator) -> Any:
    param_type = str(spec.get("type", "float")).strip().lower()
    if param_type == "categorical":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError("Categorical search spec requires non-empty 'values' list")
        return values[int(rng.integers(0, len(values)))]

    if param_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        values = list(range(low, high + 1, step))
        if not values:
            raise ValueError("Integer search spec generated no values.")
        return values[int(rng.integers(0, len(values)))]

    low = float(spec["low"])
    high = float(spec["high"])
    return float(rng.uniform(low, high))


class LowFidelityStrategy:
    """Random low-fidelity NAS strategy with explicit budget controls."""

    name = "low_fidelity"

    def __init__(
        self,
        *,
        fidelity_key: str = "epochs",
        min_fidelity: int = 1,
        max_fidelity: int = 5,
    ) -> None:
        if min_fidelity <= 0:
            raise ValueError("min_fidelity must be > 0")
        if max_fidelity < min_fidelity:
            raise ValueError("max_fidelity must be >= min_fidelity")
        self.fidelity_key = fidelity_key
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity

    def generate(
        self,
        *,
        search_space: dict[str, dict[str, Any]],
        n_candidates: int,
        rng: np.random.Generator,
        budget: dict[str, Any] | None = None,
    ) -> list[NASCandidate]:
        if n_candidates <= 0:
            raise ValueError("n_candidates must be > 0")

        settings = dict(budget or {})
        min_fidelity = int(settings.get("min_fidelity", self.min_fidelity))
        max_fidelity = int(settings.get("max_fidelity", self.max_fidelity))
        if max_fidelity < min_fidelity:
            raise ValueError("Budget max_fidelity must be >= min_fidelity")

        candidates: list[NASCandidate] = []
        for idx in range(n_candidates):
            params = {name: _sample_param(spec, rng) for name, spec in search_space.items()}
            fidelity = int(rng.integers(min_fidelity, max_fidelity + 1))
            params[self.fidelity_key] = fidelity
            candidates.append(
                NASCandidate(
                    candidate_id=f"lf-{idx}",
                    params=params,
                    fidelity=fidelity,
                    metadata={"strategy": self.name},
                )
            )
        return candidates


class WeightSharingStrategy:
    """Proxy weight-sharing strategy via shared backbone parameter groups."""

    name = "weight_sharing"

    def __init__(
        self,
        *,
        fidelity_key: str = "epochs",
        min_fidelity: int = 1,
        max_fidelity: int = 5,
        shared_keys: list[str] | None = None,
        shared_groups: int = 2,
    ) -> None:
        if min_fidelity <= 0:
            raise ValueError("min_fidelity must be > 0")
        if max_fidelity < min_fidelity:
            raise ValueError("max_fidelity must be >= min_fidelity")
        if shared_groups <= 0:
            raise ValueError("shared_groups must be > 0")
        self.fidelity_key = fidelity_key
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        self.shared_keys = list(shared_keys) if shared_keys is not None else []
        self.shared_groups = shared_groups

    def generate(
        self,
        *,
        search_space: dict[str, dict[str, Any]],
        n_candidates: int,
        rng: np.random.Generator,
        budget: dict[str, Any] | None = None,
    ) -> list[NASCandidate]:
        if n_candidates <= 0:
            raise ValueError("n_candidates must be > 0")

        settings = dict(budget or {})
        min_fidelity = int(settings.get("min_fidelity", self.min_fidelity))
        max_fidelity = int(settings.get("max_fidelity", self.max_fidelity))
        shared_groups = int(settings.get("shared_groups", self.shared_groups))
        shared_keys = list(settings.get("shared_keys", self.shared_keys))

        if max_fidelity < min_fidelity:
            raise ValueError("Budget max_fidelity must be >= min_fidelity")
        if shared_groups <= 0:
            raise ValueError("Budget shared_groups must be > 0")

        shared_pool: list[dict[str, Any]] = []
        for _ in range(shared_groups):
            group_params: dict[str, Any] = {}
            for key in shared_keys:
                if key in search_space:
                    group_params[key] = _sample_param(search_space[key], rng)
            shared_pool.append(group_params)

        candidates: list[NASCandidate] = []
        for idx in range(n_candidates):
            params = {name: _sample_param(spec, rng) for name, spec in search_space.items()}
            group_id = idx % shared_groups
            params.update(shared_pool[group_id])
            fidelity = int(rng.integers(min_fidelity, max_fidelity + 1))
            params[self.fidelity_key] = fidelity
            candidates.append(
                NASCandidate(
                    candidate_id=f"ws-{idx}",
                    params=params,
                    fidelity=fidelity,
                    metadata={"strategy": self.name, "shared_group": group_id},
                )
            )
        return candidates


def get_nas_strategy(name: str, **kwargs: Any) -> NASStrategy:
    """Instantiate NAS strategy by stable alias."""
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "lf": "low_fidelity",
        "low": "low_fidelity",
        "random_low_fidelity": "low_fidelity",
        "ws": "weight_sharing",
        "weightsharing": "weight_sharing",
    }
    resolved = aliases.get(normalized, normalized)

    if resolved == "low_fidelity":
        return LowFidelityStrategy(**kwargs)
    if resolved == "weight_sharing":
        return WeightSharingStrategy(**kwargs)

    raise ValueError(
        "NAS strategy must be one of: low_fidelity, weight_sharing"
    )
