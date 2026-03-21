"""LightGBM wrapper with optional graceful fallback."""

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


@register(name="lightgbm", family="gbdt")
class LightGBMWrapper:
    """LightGBM-style wrapper.

    Uses LGBMClassifier when available, otherwise falls back to
    HistGradientBoostingClassifier for bootstrap environments.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        scale_pos_weight: float | None = None,
        class_weight: str | dict | None = None,
        importance_type: str = "split",
        early_stopping: dict[str, Any] | None = None,
        device: str | None = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self._using_fallback = False
        self._random_state = random_state
        self._early_stopping_cfg = dict(early_stopping or {})
        
        try:
            lgb_module = importlib.import_module("lightgbm")
            lgbm_classifier = lgb_module.LGBMClassifier

            params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "min_child_samples": min_child_samples,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "random_state": random_state,
                "importance_type": importance_type,
                "n_jobs": -1,
                **kwargs
            }

            if scale_pos_weight is not None:
                params["scale_pos_weight"] = float(scale_pos_weight)
            
            if class_weight:
                params["class_weight"] = class_weight

            # Hardware detection
            preferred_device = device
            if preferred_device is None:
                detected = detect_preferred_device()
                if detected == "cuda":
                    preferred_device = "cuda"
            
            if preferred_device == "cuda":
                params["device"] = "gpu" # LightGBM uses 'gpu' for device param
            
            try:
                self.estimator = lgbm_classifier(**params)
            except Exception as e:
                _LOGGER.warning(f"Native lightgbm GPU init failed: {e}; falling back to CPU.")
                if "device" in params:
                    del params["device"]
                self.estimator = lgbm_classifier(**params)

        except (ImportError, Exception) as e:
            self._using_fallback = True
            _LOGGER.warning(
                f"lightgbm is not available ({e}); using HistGradientBoostingClassifier fallback. "
                "Install 'lightgbm' for native backend."
            )
            # HistGradientBoostingClassifier has different param names, we map the basics
            self.estimator = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=n_estimators,
                max_leaf_nodes=num_leaves,
                max_depth=max_depth if max_depth > 0 else None,
                random_state=random_state
            )

    def fit(self, X: Any, y: Any, **kwargs: Any) -> LightGBMWrapper:
        """Fit the underlying estimator with optional early stopping."""
        early_enabled = bool(self._early_stopping_cfg.get("enabled", False))
        if self._using_fallback or not early_enabled:
            self.estimator.fit(X, y, **kwargs)
            return self

        validation_split = float(self._early_stopping_cfg.get("validation_split", 0.2))
        patience = int(self._early_stopping_cfg.get("patience", 10))

        if not (0.0 < validation_split < 1.0) or len(X) <= 4:
            self.estimator.fit(X, y, **kwargs)
            return self

        stratify_target: np.ndarray | None = None
        if np.unique(y).size > 1:
            stratify_target = y

        try:
            x_train, x_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=validation_split,
                random_state=self._random_state,
                stratify=stratify_target,
            )
        except Exception:
            self.estimator.fit(X, y, **kwargs)
            return self

        fit_kwargs: dict[str, Any] = {**kwargs, "eval_set": [(x_val, y_val)]}
        if patience > 0:
            fit_kwargs["early_stopping_rounds"] = patience

        try:
            self.estimator.fit(x_train, y_train, **fit_kwargs)
        except TypeError:
            # Some versions have strict sklearn fit signatures; retry without the optional arg.
            fit_kwargs.pop("early_stopping_rounds", None)
            try:
                self.estimator.fit(x_train, y_train, **fit_kwargs)
            except Exception:
                self.estimator.fit(X, y, **kwargs)
        except Exception:
            self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict classes."""
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities."""
        return self.estimator.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances from the underlying estimator."""
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        if hasattr(self.estimator, "n_features_in_"):
            # Fallback for HistGradientBoostingClassifier which doesn't expose standard importances
            return np.zeros(self.estimator.n_features_in_)
        return np.array([])
