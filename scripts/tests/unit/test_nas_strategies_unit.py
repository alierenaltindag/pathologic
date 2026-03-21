"""Unit tests for NAS strategy selection and budget behavior."""

from __future__ import annotations

import numpy as np
import pytest

from pathologic.nas import (
    LowFidelityStrategy,
    WeightSharingStrategy,
    get_nas_strategy,
)


def test_get_nas_strategy_supported_names() -> None:
    low = get_nas_strategy("low_fidelity")
    ws = get_nas_strategy("weight_sharing")

    assert isinstance(low, LowFidelityStrategy)
    assert isinstance(ws, WeightSharingStrategy)


def test_get_nas_strategy_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match="NAS strategy"):
        get_nas_strategy("unknown")


def test_low_fidelity_strategy_respects_budget() -> None:
    rng = np.random.default_rng(42)
    strategy = LowFidelityStrategy(fidelity_key="epochs", min_fidelity=2, max_fidelity=6)
    search_space = {"lr": {"type": "float", "low": 0.001, "high": 0.01}}

    candidates = strategy.generate(
        search_space=search_space,
        n_candidates=8,
        rng=rng,
        budget={"min_fidelity": 3, "max_fidelity": 4},
    )

    assert len(candidates) == 8
    for candidate in candidates:
        assert 3 <= candidate.fidelity <= 4
        assert candidate.params["epochs"] == candidate.fidelity


def test_weight_sharing_strategy_shares_grouped_keys() -> None:
    rng = np.random.default_rng(7)
    strategy = WeightSharingStrategy(
        fidelity_key="epochs",
        shared_keys=["backbone_width"],
        shared_groups=2,
    )
    search_space = {
        "backbone_width": {"type": "categorical", "values": [32, 64, 128]},
        "head_dropout": {"type": "float", "low": 0.0, "high": 0.3},
    }

    candidates = strategy.generate(
        search_space=search_space,
        n_candidates=6,
        rng=rng,
        budget={"min_fidelity": 1, "max_fidelity": 2, "shared_groups": 2},
    )

    assert len(candidates) == 6
    group_to_width: dict[int, int] = {}
    for candidate in candidates:
        group = int(candidate.metadata["shared_group"])
        width = int(candidate.params["backbone_width"])
        if group in group_to_width:
            assert group_to_width[group] == width
        else:
            group_to_width[group] = width
