"""Unit tests for NAS search budget and scoring contract."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from pathologic.search.nas import NASearch


def test_nas_search_returns_best_candidate() -> None:
    search = NASearch(
        strategy="low_fidelity",
        random_state=42,
        max_evaluations=10,
        direction="maximize",
    )
    search_space = {
        "x": {"type": "float", "low": -2.0, "high": 4.0},
    }

    result = search.search(
        search_space=search_space,
        n_candidates=8,
        evaluate_candidate=lambda params: -(float(params["x"]) - 1.5) ** 2,
        budget={"min_fidelity": 1, "max_fidelity": 3},
    )

    assert result.strategy == "low_fidelity"
    assert len(result.trials) > 0
    assert "x" in result.best_candidate.params


def test_nas_search_early_stops_with_patience() -> None:
    search = NASearch(
        strategy="low_fidelity",
        random_state=42,
        max_evaluations=100,
        patience=2,
        min_improvement=0.0,
    )

    result = search.search(
        search_space={"x": {"type": "float", "low": 0.0, "high": 1.0}},
        n_candidates=30,
        evaluate_candidate=lambda params: 1.0,
        budget={"min_fidelity": 1, "max_fidelity": 2},
    )

    assert result.stopped_reason in {"early_stopping", "max_evaluations", "completed"}
    assert len(result.trials) <= 3


def test_nas_search_is_reproducible_with_same_seed() -> None:
    kwargs = {
        "strategy": "weight_sharing",
        "random_state": 9,
        "max_evaluations": 12,
        "strategy_kwargs": {"shared_keys": ["x"], "shared_groups": 2},
    }
    search_a = NASearch(**kwargs)
    search_b = NASearch(**kwargs)
    search_space = {
        "x": {"type": "float", "low": 0.0, "high": 1.0},
        "y": {"type": "int", "low": 1, "high": 3},
    }

    result_a = search_a.search(
        search_space=search_space,
        n_candidates=10,
        evaluate_candidate=lambda params: float(params["x"] + params["y"]),
        budget={"min_fidelity": 1, "max_fidelity": 2},
    )
    result_b = search_b.search(
        search_space=search_space,
        n_candidates=10,
        evaluate_candidate=lambda params: float(params["x"] + params["y"]),
        budget={"min_fidelity": 1, "max_fidelity": 2},
    )

    assert result_a.best_candidate.params == result_b.best_candidate.params
    assert result_a.best_score == result_b.best_score


def test_nas_for_model_runs_without_manual_objective() -> None:
    x, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model_search = NASearch.for_model(
        "logreg",
        strategy="low_fidelity",
        random_state=42,
        max_evaluations=6,
    )
    result = model_search.search(
        search_space={
            "c": {"type": "float", "low": 0.1, "high": 2.0},
            "max_iter": {"type": "int", "low": 100, "high": 220, "step": 60},
        },
        x_train=np.asarray(x_train),
        y_train=np.asarray(y_train),
        x_val=np.asarray(x_val),
        y_val=np.asarray(y_val),
        n_candidates=10,
        budget={"min_fidelity": 1, "max_fidelity": 3},
    )

    assert result.best_score >= 0.5
    assert "c" in result.best_candidate.params
