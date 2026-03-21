"""NAS search orchestration with budget limits and early stopping."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score

from pathologic.models import create_model
from pathologic.search.nas.strategies import NASCandidate, NASStrategy, get_nas_strategy
from pathologic.utils.logger import get_logger
from pathologic.utils.progress import is_progress_enabled, step_progress


@dataclass(frozen=True)
class NASTrialResult:
    """Evaluation record for one NAS candidate."""

    candidate: NASCandidate
    score: float
    elapsed_seconds: float


@dataclass(frozen=True)
class NASResult:
    """Final result payload for NAS search session."""

    strategy: str
    best_candidate: NASCandidate
    best_score: float
    trials: list[NASTrialResult]
    stopped_reason: str


class ModelBoundNASearch:
    """Convenience wrapper for running NAS directly against a model alias."""

    def __init__(
        self,
        *,
        nas: NASearch,
        model_alias: str,
        model_random_state: int,
        base_model_params: dict[str, Any] | None = None,
        score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float] | None = None,
        fidelity_param_key: str | None = "epochs",
    ) -> None:
        self.nas = nas
        self.model_alias = model_alias
        self.model_random_state = model_random_state
        self.base_model_params = dict(base_model_params or {})
        self.score_fn = score_fn or self._default_score
        self.fidelity_param_key = fidelity_param_key

    def search(
        self,
        *,
        search_space: dict[str, dict[str, Any]],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        n_candidates: int,
        budget: dict[str, Any] | None = None,
        callbacks: list[Callable[[dict[str, Any]], None]] | None = None,
    ) -> NASResult:
        """Run NAS by evaluating sampled candidate params on selected model alias."""

        def evaluate_candidate(params: dict[str, Any]) -> float:
            model_params = dict(self.base_model_params)
            for key, value in params.items():
                if self.fidelity_param_key is not None and key == self.fidelity_param_key:
                    continue
                model_params[key] = value

            model = create_model(
                self.model_alias,
                random_state=self.model_random_state,
                model_params=model_params,
            )
            model.fit(x_train, y_train)
            y_pred = np.asarray(model.predict(x_val)).reshape(-1)
            y_score: np.ndarray | None = None
            if hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(x_val))
                y_score = probabilities[:, -1] if probabilities.ndim > 1 else probabilities
            return float(self.score_fn(np.asarray(y_val), y_pred, y_score))

        return self.nas.search(
            search_space=search_space,
            evaluate_candidate=evaluate_candidate,
            n_candidates=n_candidates,
            budget=budget,
            callbacks=callbacks,
        )

    @staticmethod
    def _default_score(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> float:
        del y_score
        return float(accuracy_score(y_true, y_pred))


class NASearch:
    """Run NAS with pluggable strategy and deterministic controls."""

    def __init__(
        self,
        *,
        strategy: str | NASStrategy = "low_fidelity",
        random_state: int = 42,
        direction: str = "maximize",
        max_evaluations: int | None = None,
        max_seconds: float | None = None,
        patience: int | None = None,
        min_improvement: float = 0.0,
        strategy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be one of: maximize, minimize")

        if isinstance(strategy, str):
            self.strategy = get_nas_strategy(strategy, **dict(strategy_kwargs or {}))
        else:
            self.strategy = strategy

        self.random_state = random_state
        self.direction = direction
        self.max_evaluations = max_evaluations
        self.max_seconds = max_seconds
        self.patience = patience
        self.min_improvement = min_improvement
        self._rng = np.random.default_rng(random_state)
        self.logger = get_logger("pathologic.search.nas.search")

    @classmethod
    def for_model(
        cls,
        model_alias: str,
        *,
        strategy: str | NASStrategy = "low_fidelity",
        random_state: int = 42,
        direction: str = "maximize",
        max_evaluations: int | None = None,
        max_seconds: float | None = None,
        patience: int | None = None,
        min_improvement: float = 0.0,
        strategy_kwargs: dict[str, Any] | None = None,
        model_random_state: int | None = None,
        base_model_params: dict[str, Any] | None = None,
        score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float] | None = None,
        fidelity_param_key: str | None = "epochs",
    ) -> ModelBoundNASearch:
        """Create a model-bound NAS runner for ergonomics.

        This helper removes the need to manually define `evaluate_candidate` for
        simple model tuning workflows.
        """
        nas = cls(
            strategy=strategy,
            random_state=random_state,
            direction=direction,
            max_evaluations=max_evaluations,
            max_seconds=max_seconds,
            patience=patience,
            min_improvement=min_improvement,
            strategy_kwargs=strategy_kwargs,
        )
        resolved_model_seed = random_state if model_random_state is None else model_random_state
        return ModelBoundNASearch(
            nas=nas,
            model_alias=model_alias,
            model_random_state=resolved_model_seed,
            base_model_params=base_model_params,
            score_fn=score_fn,
            fidelity_param_key=fidelity_param_key,
        )

    def search(
        self,
        *,
        search_space: dict[str, dict[str, Any]],
        evaluate_candidate: Callable[[dict[str, Any]], float],
        n_candidates: int,
        budget: dict[str, Any] | None = None,
        callbacks: list[Callable[[dict[str, Any]], None]] | None = None,
    ) -> NASResult:
        """Run NAS strategy and return best candidate under budget constraints."""
        if n_candidates <= 0:
            raise ValueError("n_candidates must be > 0")

        candidates = self.strategy.generate(
            search_space=search_space,
            n_candidates=n_candidates,
            rng=self._rng,
            budget=budget,
        )
        if not candidates:
            raise ValueError("NAS strategy returned zero candidates.")

        best_candidate = candidates[0]
        best_score = -float("inf") if self.direction == "maximize" else float("inf")
        no_improve_steps = 0
        trials: list[NASTrialResult] = []
        started = monotonic()
        stopped_reason = "completed"

        with step_progress(
            total=len(candidates),
            desc="nas candidates",
            enabled=is_progress_enabled(),
        ) as bar:
            for index, candidate in enumerate(candidates):
                if self.max_evaluations is not None and index >= self.max_evaluations:
                    stopped_reason = "max_evaluations"
                    break
                elapsed = monotonic() - started
                if self.max_seconds is not None and elapsed >= self.max_seconds:
                    stopped_reason = "timeout"
                    break

                score = float(evaluate_candidate(dict(candidate.params)))
                trial_elapsed = monotonic() - started
                trials.append(
                    NASTrialResult(
                        candidate=candidate,
                        score=score,
                        elapsed_seconds=trial_elapsed,
                    )
                )
                self._run_callbacks(
                    callbacks,
                    {
                        "candidate_index": len(trials),
                        "candidate_total": len(candidates),
                        "score": float(score),
                        "params": dict(candidate.params),
                    },
                )

                if self._is_better(score, best_score):
                    improvement = self._improvement_amount(score, best_score)
                    best_score = score
                    best_candidate = candidate
                    if improvement > self.min_improvement:
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1

                bar.update(1)
                bar.set_postfix(best=f"{best_score:.4f}")

                if self.patience is not None and no_improve_steps >= self.patience:
                    stopped_reason = "early_stopping"
                    break

        self.logger.info(
            "NAS finished with strategy=%s, trials=%d, stopped_reason=%s",
            self.strategy.name,
            len(trials),
            stopped_reason,
        )
        return NASResult(
            strategy=self.strategy.name,
            best_candidate=best_candidate,
            best_score=float(best_score),
            trials=trials,
            stopped_reason=stopped_reason,
        )

    def _is_better(self, score: float, best: float) -> bool:
        if self.direction == "maximize":
            return score > best
        return score < best

    def _improvement_amount(self, score: float, best: float) -> float:
        if np.isinf(best):
            return float("inf")
        if self.direction == "maximize":
            return float(score - best)
        return float(best - score)

    @staticmethod
    def _run_callbacks(
        callbacks: list[Callable[[dict[str, Any]], None]] | None,
        payload: dict[str, Any],
    ) -> None:
        if not callbacks:
            return
        for callback in callbacks:
            callback(dict(payload))
