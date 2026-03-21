"""Unit tests for tuner engines and deterministic behavior."""

from __future__ import annotations

from pathologic.engine import Tuner


def test_random_tuner_returns_best_trial() -> None:
    tuner = Tuner(engine="random", random_state=42)

    def objective(params: dict[str, float]) -> float:
        x = float(params["x"])
        return -(x - 2.0) ** 2

    result = tuner.tune(
        objective=objective,
        search_space={"x": {"type": "float", "low": -2.0, "high": 4.0}},
        n_trials=8,
        direction="maximize",
    )

    assert result.engine == "random"
    assert "x" in result.best_params
    assert len(result.trials) == 8


def test_grid_tuner_uses_categorical_values() -> None:
    tuner = Tuner(engine="grid", random_state=42)

    def objective(params: dict[str, int]) -> float:
        return float(params["x"])

    result = tuner.tune(
        objective=objective,
        search_space={"x": {"type": "categorical", "values": [1, 3, 2]}},
        n_trials=3,
        direction="maximize",
    )

    assert result.engine == "grid"
    assert result.best_params["x"] == 3
    assert result.best_score == 3.0


def test_optuna_engine_falls_back_when_unavailable(monkeypatch) -> None:
    tuner = Tuner(engine="optuna", random_state=42)
    monkeypatch.setattr(tuner, "_tune_optuna", lambda **kwargs: None)

    def objective(params: dict[str, float]) -> float:
        return float(params["x"])

    result = tuner.tune(
        objective=objective,
        search_space={"x": {"type": "float", "low": 0.0, "high": 1.0}},
        n_trials=4,
        direction="maximize",
    )

    assert result.engine == "random"
    assert len(result.trials) == 4


def test_tuner_callbacks_receive_each_trial() -> None:
    tuner = Tuner(engine="random", random_state=42)
    seen_scores: list[float] = []

    def callback(trial_info: dict[str, object]) -> None:
        seen_scores.append(float(trial_info["score"]))

    result = tuner.tune(
        objective=lambda params: float(params["x"]),
        search_space={"x": {"type": "float", "low": 0.0, "high": 1.0}},
        n_trials=5,
        direction="maximize",
        callbacks=[callback],
    )

    assert len(seen_scores) == len(result.trials)


def test_tuner_early_stops_when_no_improvement() -> None:
    tuner = Tuner(engine="random", random_state=42)
    result = tuner.tune(
        objective=lambda params: 1.0,
        search_space={"x": {"type": "float", "low": 0.0, "high": 1.0}},
        n_trials=50,
        direction="maximize",
        early_stopping={"enabled": True, "patience": 2, "min_improvement": 0.0},
    )

    assert len(result.trials) <= 3


def test_tuner_random_is_reproducible_with_same_seed() -> None:
    search_space = {
        "x": {"type": "float", "low": 0.0, "high": 1.0},
        "y": {"type": "int", "low": 1, "high": 3},
    }

    def objective(params: dict[str, float]) -> float:
        return float(params["x"] + params["y"])

    tuner_a = Tuner(engine="random", random_state=7)
    tuner_b = Tuner(engine="random", random_state=7)

    result_a = tuner_a.tune(
        objective=objective,
        search_space=search_space,
        n_trials=8,
        direction="maximize",
    )
    result_b = tuner_b.tune(
        objective=objective,
        search_space=search_space,
        n_trials=8,
        direction="maximize",
    )

    assert result_a.best_params == result_b.best_params
    assert result_a.best_score == result_b.best_score
