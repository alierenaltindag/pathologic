"""CatBoost wrapper with optional graceful fallback."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from pathologic.models.registry import register
from pathologic.utils import get_logger
from pathologic.utils.hardware import detect_preferred_device

_LOGGER = get_logger(__name__)


@register(name="catboost", family="gbdt")
class CatBoostWrapper:
    """CatBoost-style wrapper.

    Uses CatBoostClassifier when available, otherwise falls back to
    RandomForestClassifier for bootstrap environments.
    """

    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        subsample: float = 1.0,
        bootstrap_type: str | None = None,
        class_weight: str | None = None,
        class_weights: list[float] | dict[int, float] | None = None,
        early_stopping: dict[str, Any] | None = None,
        task_type: str | None = None,
        random_state: int = 42,
    ) -> None:
        self._using_fallback = False
        self._random_state = random_state
        self._early_stopping_cfg = dict(early_stopping or {})
        try:
            catboost_module = importlib.import_module("catboost")
            catboost_classifier = catboost_module.CatBoostClassifier

            common_params = {
                "iterations": iterations,
                "learning_rate": learning_rate,
                "depth": depth,
                "l2_leaf_reg": float(l2_leaf_reg),
                "random_strength": float(random_strength),
                "random_seed": random_state,
                "logging_level": "Silent",
            }
            if bootstrap_type is not None:
                common_params["bootstrap_type"] = str(bootstrap_type)

            if float(subsample) < 1.0:
                common_params.setdefault("bootstrap_type", "Bernoulli")
                common_params["subsample"] = float(subsample)

            bootstrap_name = str(common_params.get("bootstrap_type", "Bayesian")).lower()
            if bootstrap_name == "bayesian":
                common_params["bagging_temperature"] = float(bagging_temperature)

            if class_weights is not None:
                common_params["class_weights"] = class_weights
            elif class_weight == "balanced":
                common_params["auto_class_weights"] = "Balanced"

            resolved_task_type = task_type
            if resolved_task_type is None and detect_preferred_device() == "cuda":
                resolved_task_type = "GPU"

            try:
                if resolved_task_type is not None:
                    self.estimator = catboost_classifier(
                        **common_params,
                        task_type=resolved_task_type,
                    )
                else:
                    self.estimator = catboost_classifier(**common_params)
            except Exception:
                self.estimator = catboost_classifier(**common_params)
                if resolved_task_type == "GPU":
                    _LOGGER.warning(
                        "Native catboost GPU init failed; falling back to CPU backend."
                    )
        except Exception:
            self._using_fallback = True
            _LOGGER.warning(
                "catboost is not available; using RandomForestClassifier fallback. "
                "Install optional dependency group 'models' for native backend."
            )
            self.estimator = RandomForestClassifier(
                n_estimators=200,
                class_weight=("balanced" if class_weight == "balanced" else None),
                random_state=random_state,
            )

    def fit(self, x: np.ndarray, y: np.ndarray) -> CatBoostWrapper:
        early_enabled = bool(self._early_stopping_cfg.get("enabled", False))
        if not early_enabled:
            self.estimator.fit(x, y)
            return self

        validation_split = float(self._early_stopping_cfg.get("validation_split", 0.2))
        patience = int(self._early_stopping_cfg.get("patience", 10))
        restore_best = bool(self._early_stopping_cfg.get("restore_best_weights", True))

        if not (0.0 < validation_split < 1.0) or len(x) <= 4:
            self.estimator.fit(x, y)
            return self

        stratify_target: np.ndarray | None = None
        if np.unique(y).size > 1:
            stratify_target = y
        try:
            x_train, x_val, y_train, y_val = train_test_split(
                x,
                y,
                test_size=validation_split,
                random_state=self._random_state,
                stratify=stratify_target,
            )
        except Exception:
            self.estimator.fit(x, y)
            return self

        fit_kwargs: dict[str, Any] = {
            "eval_set": (x_val, y_val),
            "use_best_model": restore_best,
            "verbose": False,
        }
        if patience > 0:
            fit_kwargs["early_stopping_rounds"] = patience

        try:
            self.estimator.fit(x_train, y_train, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("verbose", None)
            try:
                self.estimator.fit(x_train, y_train, **fit_kwargs)
            except Exception:
                self.estimator.fit(x, y)
        except Exception:
            self.estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(x)).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(x))
        logits = np.asarray(self.estimator.decision_function(x)).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])
