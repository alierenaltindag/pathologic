"""Hyperparameter tuning engine with Optuna and deterministic fallbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from time import monotonic
from typing import Any

import numpy as np

from pathologic.utils.logger import get_logger
from pathologic.utils.progress import is_progress_enabled, step_progress


@dataclass(frozen=True)
class TuningResult:
    """Serializable result of a tuning session."""

    engine: str
    best_params: dict[str, Any]
    best_score: float
    trials: list[dict[str, Any]]


class Tuner:
    """Tune objective function using Optuna or deterministic fallbacks."""

    def __init__(self, *, engine: str = "random", random_state: int = 42) -> None:
        normalized = engine.strip().lower()
        if normalized not in {"optuna", "random", "grid"}:
            raise ValueError("Tuner engine must be one of: optuna, random, grid")
        self.engine = normalized
        self.random_state = random_state
        self.logger = get_logger("pathologic.engine.tuner")
        self._rng = np.random.default_rng(random_state)

    def tune(
        self,
        *,
        objective: Callable[[dict[str, Any]], float],
        search_space: dict[str, dict[str, Any]],
        n_trials: int,
        timeout_seconds: float | None = None,
        direction: str = "maximize",
        callbacks: list[Callable[[dict[str, Any]], None]] | None = None,
        early_stopping: dict[str, Any] | None = None,
    ) -> TuningResult:
        """Run tuning and return best trial summary."""
        if n_trials <= 0:
            raise ValueError("n_trials must be > 0")
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be either 'maximize' or 'minimize'")

        if self.engine == "optuna":
            optuna_result = self._tune_optuna(
                objective=objective,
                search_space=search_space,
                n_trials=n_trials,
                timeout_seconds=timeout_seconds,
                direction=direction,
                callbacks=callbacks,
                early_stopping=early_stopping,
            )
            if optuna_result is not None:
                return optuna_result
            self.logger.warning("Optuna unavailable. Falling back to random search.")

        if self.engine == "grid":
            return self._tune_grid(
                objective=objective,
                search_space=search_space,
                n_trials=n_trials,
                timeout_seconds=timeout_seconds,
                direction=direction,
                callbacks=callbacks,
                early_stopping=early_stopping,
            )

        return self._tune_random(
            objective=objective,
            search_space=search_space,
            n_trials=n_trials,
            timeout_seconds=timeout_seconds,
            direction=direction,
            callbacks=callbacks,
            early_stopping=early_stopping,
        )

    def _tune_optuna(
        self,
        *,
        objective: Callable[[dict[str, Any]], float],
        search_space: dict[str, dict[str, Any]],
        n_trials: int,
        timeout_seconds: float | None,
        direction: str,
        callbacks: list[Callable[[dict[str, Any]], None]] | None,
        early_stopping: dict[str, Any] | None,
    ) -> TuningResult | None:
        try:
            import optuna  # pylint: disable=import-outside-toplevel  # pylint: disable=import-outside-toplevel  # type: ignore[import-not-found]
        except Exception:
            return None

        trials_out: list[dict[str, Any]] = []

        best_score = -float("inf") if direction == "maximize" else float("inf")
        non_improving_trials = 0
        es_enabled = bool((early_stopping or {}).get("enabled", False))
        es_patience = int((early_stopping or {}).get("patience", 5))
        es_min_improvement = float((early_stopping or {}).get("min_improvement", 0.0))

        def optuna_objective(trial: Any) -> float:
            nonlocal best_score
            nonlocal non_improving_trials
            params: dict[str, Any] = {}
            for name, spec in search_space.items():
                params[name] = self._suggest_optuna_param(trial, name, spec)
            score = float(objective(params))
            trial_info = {"params": params, "score": score}
            trials_out.append(trial_info)
            self._run_callbacks(callbacks, trial_info)

            improved = self._improvement_amount(score, best_score, direction) > es_min_improvement
            if improved:
                best_score = score
                non_improving_trials = 0
            else:
                non_improving_trials += 1

            if es_enabled and non_improving_trials >= es_patience:
                raise optuna.TrialPruned("Early stopping patience reached.")

            return score

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)
        show_progress = is_progress_enabled()
        with step_progress(total=n_trials, desc="tune optuna", enabled=show_progress) as bar:
            def wrapped_optuna_objective(trial: Any) -> float:
                score = optuna_objective(trial)
                ppbar.update(1)
                bar.set_postfix(best=f"{best_score:.4f}")
                return score

            study.optimize(
                wrapped_optuna_objective,
                n_trials=n_trials,
                timeout=timeout_seconds,
                n_jobs=2,  # MAC_OPTIMIZATION: Aynı anda 2 farklı modeli paralel test ederek süreyi yarıya indirir.
            )

        best = study.best_trial
        return TuningResult(
            engine="optuna",
            best_params=dict(best.params),
            best_score=float(best.value),
            trials=trials_out,
        )

    def _tune_grid(
        self,
        *,
        objective: Callable[[dict[str, Any]], float],
        search_space: dict[str, dict[str, Any]],
        n_trials: int,
        timeout_seconds: float | None,
        direction: str,
        callbacks: list[Callable[[dict[str, Any]], None]] | None,
        early_stopping: dict[str, Any] | None,
    ) -> TuningResult:
        all_candidates = self._grid_candidates(search_space)
        start = monotonic()
        trial_records: list[dict[str, Any]] = []
        best_params: dict[str, Any] = {}
        best_score = -float("inf") if direction == "maximize" else float("inf")
        non_improving_trials = 0
        es_enabled = bool((early_stopping or {}).get("enabled", False))
        es_patience = int((early_stopping or {}).get("patience", 5))
        es_min_improvement = float((early_stopping or {}).get("min_improvement", 0.0))

        with step_progress(
            total=min(len(all_candidates), n_trials),
            desc="tune grid",
            enabled=is_progress_enabled(),
        ) as bar:
            for params in all_candidates:
                if len(trial_records) >= n_trials:
                    break
                if timeout_seconds is not None and (monotonic() - start) >= timeout_seconds:
                    break

                score = float(objective(params))
                trial_info = {"params": params, "score": score}
                trial_records.append(trial_info)
                self._run_callbacks(callbacks, trial_info)
                if self._is_better(score, best_score, direction):
                    improvement = self._improvement_amount(score, best_score, direction)
                    best_score = score
                    best_params = dict(params)
                    if improvement > es_min_improvement:
                        non_improving_trials = 0
                    else:
                        non_improving_trials += 1
                else:
                    non_improving_trials += 1

                ppbar.update(1)
                bar.set_postfix(best=f"{best_score:.4f}")

                if es_enabled and non_improving_trials >= es_patience:
                    break

        return TuningResult(
            engine="grid",
            best_params=best_params,
            best_score=best_score,
            trials=trial_records,
        )

    def _tune_random(
        self,
        *,
        objective: Callable[[dict[str, Any]], float],
        search_space: dict[str, dict[str, Any]],
        n_trials: int,
        timeout_seconds: float | None,
        direction: str,
        callbacks: list[Callable[[dict[str, Any]], None]] | None,
        early_stopping: dict[str, Any] | None,
    ) -> TuningResult:
        start = monotonic()
        trial_records: list[dict[str, Any]] = []
        best_params: dict[str, Any] = {}
        best_score = -float("inf") if direction == "maximize" else float("inf")
        non_improving_trials = 0
        es_enabled = bool((early_stopping or {}).get("enabled", False))
        es_patience = int((early_stopping or {}).get("patience", 5))
        es_min_improvement = float((early_stopping or {}).get("min_improvement", 0.0))

        with step_progress(
            total=n_trials,
            desc="tune random",
            enabled=is_progress_enabled(),
        ) as bar:
            for _ in range(n_trials):
                if timeout_seconds is not None and (monotonic() - start) >= timeout_seconds:
                    break
                params = {
                    name: self._sample_random_param(spec)
                    for name, spec in search_space.items()
                }
                score = float(objective(params))
                trial_info = {"params": params, "score": score}
                trial_records.append(trial_info)
                self._run_callbacks(callbacks, trial_info)
                if self._is_better(score, best_score, direction):
                    improvement = self._improvement_amount(score, best_score, direction)
                    best_score = score
                    best_params = dict(params)
                    if improvement > es_min_improvement:
                        non_improving_trials = 0
                    else:
                        non_improving_trials += 1
                else:
                    non_improving_trials += 1

                ppbar.update(1)
                bar.set_postfix(best=f"{best_score:.4f}")

                if es_enabled and non_improving_trials >= es_patience:
                    break

        return TuningResult(
            engine="random",
            best_params=best_params,
            best_score=best_score,
            trials=trial_records,
        )

    @staticmethod
    def _is_better(score: float, best: float, direction: str) -> bool:
        if direction == "maximize":
            return score > best
        return score < best

    @staticmethod
    def _improvement_amount(score: float, best: float, direction: str) -> float:
        if np.isinf(best):
            return float("inf")
        if direction == "maximize":
            return float(score - best)
        return float(best - score)

    @staticmethod
    def _run_callbacks(
        callbacks: list[Callable[[dict[str, Any]], None]] | None,
        trial_info: dict[str, Any],
    ) -> None:
        if callbacks is None:
            return
        for callback in callbacks:
            callback(dict(trial_info))

    def _sample_random_param(self, spec: dict[str, Any]) -> Any:
        param_type = str(spec.get("type", "float")).strip().lower()
        if param_type == "categorical":
            values = spec.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError("Categorical search spec requires non-empty 'values' list")
            index = int(self._rng.integers(0, len(values)))
            return values[index]

        if param_type == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            return int(self._rng.integers(low, high + 1))

        low_float = float(spec["low"])
        high_float = float(spec["high"])
        return float(self._rng.uniform(low_float, high_float))

    @staticmethod
    def _grid_candidates(search_space: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        keys = list(search_space.keys())
        values: list[list[Any]] = []

        for name in keys:
            spec = search_space[name]
            param_type = str(spec.get("type", "categorical")).strip().lower()
            if param_type == "categorical":
                options = spec.get("values")
                if not isinstance(options, list) or not options:
                    raise ValueError(
                        f"Grid search for '{name}' requires non-empty categorical values"
                    )
                values.append(list(options))
                continue

            if param_type == "int":
                low = int(spec["low"])
                high = int(spec["high"])
                step = int(spec.get("step", 1))
                values.append(list(range(low, high + 1, step)))
                continue

            low_float = float(spec["low"])
            high_float = float(spec["high"])
            step_float = float(spec.get("step", (high_float - low_float) / 4.0))
            if step_float <= 0:
                raise ValueError(f"Grid step must be > 0 for parameter '{name}'")
            generated: list[float] = []
            current = low_float
            while current <= high_float + 1e-12:
                generated.append(float(current))
                current += step_float
            values.append(generated)

        return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]

    @staticmethod
    def _suggest_optuna_param(trial: Any, name: str, spec: dict[str, Any]) -> Any:
        param_type = str(spec.get("type", "float")).strip().lower()
        if param_type == "categorical":
            values = spec.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError("Categorical search spec requires non-empty 'values' list")
            return trial.suggest_categorical(name, values)

        if param_type == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            return trial.suggest_int(name, low, high, step=step)

        low_float = float(spec["low"])
        high_float = float(spec["high"])
        log = bool(spec.get("log", False))
        step_raw = spec.get("step")
        if step_raw is None:
            return trial.suggest_float(name, low_float, high_float, log=log)
        return trial.suggest_float(
            name,
            low_float,
            high_float,
            step=float(step_raw),
            log=log,
        )
