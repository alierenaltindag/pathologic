"""LightGBM wrapper with optional graceful fallback."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
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

    _cuda_backend_status: str = "unknown"

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
        self._lgbm_classifier: Any | None = None
        
        try:
            lgb_module = importlib.import_module("lightgbm")
            lgbm_classifier = lgb_module.LGBMClassifier
            self._lgbm_classifier = lgbm_classifier

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
            preferred_device = str(device).strip().lower() if device is not None else None
            if preferred_device is None:
                detected = detect_preferred_device()
                if detected == "cuda":
                    preferred_device = "cuda"
                else:
                    preferred_device = None
            
            if preferred_device == "cuda":
                if type(self)._cuda_backend_status == "unsupported":
                    params["device"] = "gpu"
                else:
                    # Prefer native CUDA build when available.
                    params["device_type"] = "cuda"
            elif preferred_device in {"gpu", "opencl"}:
                params["device"] = "gpu"
            
            try:
                self.estimator = lgbm_classifier(**params)
            except Exception as e:
                if preferred_device == "cuda":
                    _LOGGER.warning(
                        "Native lightgbm CUDA init failed: %s; retrying OpenCL GPU backend.",
                        e,
                    )
                    alt_params = dict(params)
                    alt_params.pop("device_type", None)
                    alt_params["device"] = "gpu"
                    try:
                        self.estimator = lgbm_classifier(**alt_params)
                    except Exception as gpu_exc:
                        _LOGGER.warning(
                            "Native lightgbm OpenCL init failed: %s; falling back to CPU.",
                            gpu_exc,
                        )
                        cpu_params = dict(params)
                        cpu_params.pop("device", None)
                        cpu_params.pop("device_type", None)
                        self.estimator = lgbm_classifier(**cpu_params)
                else:
                    _LOGGER.warning(f"Native lightgbm GPU init failed: {e}; falling back to CPU.")
                    if "device" in params:
                        del params["device"]
                    if "device_type" in params:
                        del params["device_type"]
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

    @staticmethod
    def _is_lgbm_gpu_runtime_failure(exc: Exception) -> bool:
        message = str(exc).strip().lower()
        if not message:
            return False
        patterns = (
            "opencl",
            "cuda",
            "gpu",
            "no opencl device found",
            "cuda tree learner",
        )
        return any(pattern in message for pattern in patterns)

    @staticmethod
    def _is_lgbm_cuda_tree_not_enabled_failure(exc: Exception) -> bool:
        message = str(exc).strip().lower()
        if not message:
            return False
        return (
            "cuda tree learner was not enabled in this build" in message
            or "recompile with cmake option -duse_cuda=1" in message
        )

    def _fallback_estimator_for_runtime_failure(self, exc: Exception) -> bool:
        if self._using_fallback or self._lgbm_classifier is None:
            return False

        current_params = dict(getattr(self.estimator, "get_params", lambda: {})())
        device = str(current_params.get("device", "")).strip().lower()
        device_type = str(current_params.get("device_type", "")).strip().lower()
        if device not in {"gpu", "cuda"} and device_type not in {"gpu", "cuda"}:
            return False

        if device_type == "cuda" and self._is_lgbm_cuda_tree_not_enabled_failure(exc):
            type(self)._cuda_backend_status = "unsupported"
            opencl_params = dict(current_params)
            opencl_params.pop("device_type", None)
            opencl_params["device"] = "gpu"
            try:
                self.estimator = self._lgbm_classifier(**opencl_params)
                _LOGGER.warning(
                    "LightGBM CUDA build is unavailable; retrying with OpenCL GPU backend."
                )
                return True
            except Exception:
                pass

        current_params.pop("device", None)
        current_params.pop("device_type", None)
        self.estimator = self._lgbm_classifier(**current_params)
        _LOGGER.warning(
            "LightGBM GPU runtime failed; switched to CPU backend for this candidate."
        )
        return True

    def _fit_with_gpu_fallback(self, x: Any, y: Any, **kwargs: Any) -> None:
        attempts = 0
        while attempts < 3:
            attempts += 1
            try:
                self.estimator.fit(x, y, **kwargs)
                return
            except Exception as exc:
                if (
                    self._is_lgbm_gpu_runtime_failure(exc)
                    and self._fallback_estimator_for_runtime_failure(exc)
                ):
                    continue
                raise

    def fit(self, X: Any, y: Any, **kwargs: Any) -> LightGBMWrapper:
        """Fit the underlying estimator with optional early stopping."""
        early_enabled = bool(self._early_stopping_cfg.get("enabled", False))
        if self._using_fallback or not early_enabled:
            self._fit_with_gpu_fallback(X, y, **kwargs)
            return self

        validation_split = float(self._early_stopping_cfg.get("validation_split", 0.2))
        patience = int(self._early_stopping_cfg.get("patience", 10))

        x_val_external = kwargs.pop("x_val", None)
        y_val_external = kwargs.pop("y_val", None)
        if x_val_external is not None and y_val_external is not None and len(x_val_external) > 0:
            fit_kwargs: dict[str, Any] = {**kwargs, "eval_set": [(x_val_external, y_val_external)]}
            if patience > 0:
                fit_kwargs["early_stopping_rounds"] = patience
            try:
                self._fit_with_gpu_fallback(X, y, **fit_kwargs)
            except TypeError:
                fit_kwargs.pop("early_stopping_rounds", None)
                try:
                    self._fit_with_gpu_fallback(X, y, **fit_kwargs)
                except Exception:
                    self._fit_with_gpu_fallback(X, y, **kwargs)
            except Exception:
                self._fit_with_gpu_fallback(X, y, **kwargs)
            return self

        if not (0.0 < validation_split < 1.0) or len(X) <= 4:
            self._fit_with_gpu_fallback(X, y, **kwargs)
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
            self._fit_with_gpu_fallback(X, y, **kwargs)
            return self

        fit_kwargs: dict[str, Any] = {**kwargs, "eval_set": [(x_val, y_val)]}
        if patience > 0:
            fit_kwargs["early_stopping_rounds"] = patience

        try:
            self._fit_with_gpu_fallback(x_train, y_train, **fit_kwargs)
        except TypeError:
            # Some versions have strict sklearn fit signatures; retry without the optional arg.
            fit_kwargs.pop("early_stopping_rounds", None)
            try:
                self._fit_with_gpu_fallback(x_train, y_train, **fit_kwargs)
            except Exception:
                self._fit_with_gpu_fallback(X, y, **kwargs)
        except Exception:
            self._fit_with_gpu_fallback(X, y, **kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict classes."""
        return self.estimator.predict(self._normalize_inference_input(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities."""
        return self.estimator.predict_proba(self._normalize_inference_input(X))

    def _normalize_inference_input(self, X: Any) -> Any:
        """Provide named columns when estimator expects feature names."""
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return X

        feature_names = getattr(self.estimator, "feature_name_", None)
        if not isinstance(feature_names, list) or len(feature_names) != X.shape[1]:
            return X

        normalized_columns = [str(name) for name in feature_names]
        if any(not name for name in normalized_columns):
            return X

        return pd.DataFrame(X, columns=normalized_columns)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances from the underlying estimator."""
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        if hasattr(self.estimator, "n_features_in_"):
            # Fallback for HistGradientBoostingClassifier which doesn't expose standard importances
            return np.zeros(self.estimator.n_features_in_)
        return np.array([])
