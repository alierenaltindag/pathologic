"""XGBoost wrapper with optional graceful fallback."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from pathologic.models.registry import register
from pathologic.utils import get_logger
from pathologic.utils.hardware import detect_preferred_device

_LOGGER = get_logger(__name__)


@register(name="xgboost", family="gbdt")
class XGBoostWrapper:
    """XGBoost-style wrapper.

    Uses XGBClassifier when available, otherwise falls back to
    HistGradientBoostingClassifier for bootstrap environments.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float | None = None,
        class_weight: str | None = None,
        early_stopping: dict[str, Any] | None = None,
        tree_method: str | None = None,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self._using_fallback = False
        self._random_state = random_state
        self._early_stopping_cfg = dict(early_stopping or {})
        try:
            xgboost_module = importlib.import_module("xgboost")
            xgb_classifier = xgboost_module.XGBClassifier

            common_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "min_child_weight": float(min_child_weight),
                "subsample": float(subsample),
                "colsample_bytree": float(colsample_bytree),
                "gamma": float(gamma),
                "reg_alpha": float(reg_alpha),
                "reg_lambda": float(reg_lambda),
                "random_state": random_state,
            }
            if scale_pos_weight is not None:
                common_params["scale_pos_weight"] = float(scale_pos_weight)

            if class_weight == "balanced" and scale_pos_weight is None:
                # Keep compatibility for callers that pass only class_weight.
                common_params["scale_pos_weight"] = 1.0

            preferred_device = device
            # MAC_OPTIMIZATION: XGBoost natively targets Apple Silicon CPU very efficiently.
            # We explicitly only target GPU parameters if CUDA is present, avoiding 
            # unsupported native M1/M2/M3 device injections.
            if preferred_device is None and detect_preferred_device() == "cuda":
                preferred_device = "cuda"

            gpu_params: dict[str, str] = {}
            if preferred_device == "cuda":
                gpu_params["device"] = "cuda"
                gpu_params["tree_method"] = tree_method or "hist"
            elif tree_method is not None:
                gpu_params["tree_method"] = tree_method

            try:
                self.estimator = xgb_classifier(**common_params, **gpu_params)
            except Exception:
                self.estimator = xgb_classifier(**common_params)
                if preferred_device == "cuda":
                    _LOGGER.warning(
                        "Native xgboost GPU init failed; falling back to CPU backend."
                    )
        except Exception:
            self._using_fallback = True
            _LOGGER.warning(
                "xgboost is not available; using HistGradientBoostingClassifier fallback. "
                "Install optional dependency group 'models' for native backend."
            )
            self.estimator = HistGradientBoostingClassifier(random_state=random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> XGBoostWrapper:
        early_enabled = bool(self._early_stopping_cfg.get("enabled", False))
        if not early_enabled or not hasattr(self.estimator, "fit"):
            self.estimator.fit(x, y)
            return self

        validation_split = float(self._early_stopping_cfg.get("validation_split", 0.2))
        patience = int(self._early_stopping_cfg.get("patience", 10))

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

        fit_kwargs: dict[str, Any] = {"eval_set": [(x_val, y_val)]}
        if patience > 0:
            fit_kwargs["early_stopping_rounds"] = patience

        try:
            self.estimator.fit(x_train, y_train, verbose=False, **fit_kwargs)
        except TypeError:
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
