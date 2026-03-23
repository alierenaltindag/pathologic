"""TabNet wrapper with fallback implementation for bootstrap stage."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from pathologic.models.registry import register
from pathologic.utils import get_logger
from pathologic.utils.hardware import detect_preferred_device

_LOGGER = get_logger(__name__)


@register(name="tabnet", family="tabular-neural-network")
class TabNetWrapper:
    """TabNet-style wrapper.

    Uses pytorch-tabnet when available, otherwise uses an MLP fallback to keep
    API and tests functional in environments without optional dependency.
    """

    def __init__(
        self,
        *,
        max_epochs: int = 40,
        patience: int = 10,
        learning_rate: float = 0.02,
        weight_decay: float = 0.0,
        optimizer_name: str = "adam",
        scheduler_name: str = "none",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.9,
        class_weight: str | None = None,
        early_stopping: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        lambda_sparse: float = 1e-3,
        mask_type: str = "sparsemax",
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        fallback_hidden_layer_sizes: Sequence[int] = (64, 32),
        fallback_max_iter: int = 400,
        device_name: str | None = None,
        random_state: int = 42,
    ) -> None:
        self._using_fallback = False
        self._is_native_tabnet = False
        self._max_epochs = max_epochs
        self._patience = patience
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._optimizer_name = str(optimizer_name).strip().lower()
        self._scheduler_name = str(scheduler_name).strip().lower()
        self._scheduler_patience = int(scheduler_patience)
        self._scheduler_factor = float(scheduler_factor)
        self._scheduler_step_size = int(scheduler_step_size)
        self._scheduler_gamma = float(scheduler_gamma)
        self._class_weight = class_weight
        self._early_stopping_cfg = dict(early_stopping or {})
        self._optimizer_cfg = dict(optimizer or {})
        self._scheduler_cfg = dict(scheduler or {})
        self._n_d = int(n_d)
        self._n_a = int(n_a)
        self._n_steps = int(n_steps)
        self._gamma = float(gamma)
        self._lambda_sparse = float(lambda_sparse)
        self._mask_type = str(mask_type)
        self._batch_size = int(batch_size)
        self._virtual_batch_size = int(virtual_batch_size)
        self._random_state = random_state
        try:
            tabnet_module = importlib.import_module("pytorch_tabnet.tab_model")
            tabnet_classifier = tabnet_module.TabNetClassifier

            self._is_native_tabnet = True
            resolved_device_name = device_name
            if resolved_device_name is None and detect_preferred_device() == "cuda":
                resolved_device_name = "cuda"

            optimizer_fn, optimizer_params = self._resolve_native_optimizer()
            scheduler_fn, scheduler_params = self._resolve_native_scheduler()

            tabnet_kwargs: dict[str, Any] = {
                "seed": random_state,
                "verbose": 0,
                "n_d": self._n_d,
                "n_a": self._n_a,
                "n_steps": self._n_steps,
                "gamma": self._gamma,
                "lambda_sparse": self._lambda_sparse,
                "mask_type": self._mask_type,
                "optimizer_fn": optimizer_fn,
                "optimizer_params": optimizer_params,
            }
            if scheduler_fn is not None:
                tabnet_kwargs["scheduler_fn"] = scheduler_fn
                tabnet_kwargs["scheduler_params"] = scheduler_params
            if resolved_device_name is not None:
                tabnet_kwargs["device_name"] = resolved_device_name

            try:
                self.estimator = tabnet_classifier(**tabnet_kwargs)
            except Exception:
                fallback_kwargs = dict(tabnet_kwargs)
                fallback_kwargs.pop("device_name", None)
                self.estimator = tabnet_classifier(**fallback_kwargs)
                if resolved_device_name == "cuda":
                    _LOGGER.warning(
                        "Native tabnet GPU init failed; falling back to default backend."
                    )
        except Exception:
            self._using_fallback = True
            _LOGGER.warning(
                "pytorch-tabnet is not available; using MLPClassifier fallback. "
                "Install optional dependency group 'models' for native backend."
            )
            self.estimator = MLPClassifier(
                hidden_layer_sizes=tuple(fallback_hidden_layer_sizes),
                random_state=random_state,
                max_iter=fallback_max_iter,
            )

    @staticmethod
    def _to_numpy_features(x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x.to_numpy(dtype=float)
        if isinstance(x, pd.Series):
            return x.to_numpy(dtype=float).reshape(-1, 1)
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    @staticmethod
    def _to_numpy_labels(y: Any) -> np.ndarray:
        if isinstance(y, pd.DataFrame):
            return y.to_numpy().reshape(-1)
        if isinstance(y, pd.Series):
            return y.to_numpy().reshape(-1)
        if isinstance(y, np.ndarray):
            return y.reshape(-1)
        return np.asarray(y).reshape(-1)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TabNetWrapper:
        x_np = self._to_numpy_features(x)
        y_np = self._to_numpy_labels(y)

        if self._is_native_tabnet:
            patience = int(self._early_stopping_cfg.get("patience", self._patience))
            early_enabled = bool(self._early_stopping_cfg.get("enabled", True))
            validation_split = float(self._early_stopping_cfg.get("validation_split", 0.2))

            fit_kwargs: dict[str, Any] = {
                "max_epochs": self._max_epochs,
                "patience": patience if early_enabled else max(self._max_epochs, 1),
                "batch_size": self._batch_size,
                "virtual_batch_size": self._virtual_batch_size,
            }
            if early_enabled and x_val is not None and y_val is not None and len(x_val) > 0:
                x_val_np = self._to_numpy_features(x_val)
                y_val_np = self._to_numpy_labels(y_val)
                fit_kwargs["eval_set"] = [(x_val_np, y_val_np)]
                self.estimator.fit(x_np, y_np, **fit_kwargs)
                return self

            if early_enabled and 0.0 < validation_split < 1.0 and len(x_np) > 4:
                stratify_target: np.ndarray | None = None
                if np.unique(y_np).size > 1:
                    stratify_target = y_np
                try:
                    x_train, x_val, y_train, y_val = train_test_split(
                        x_np,
                        y_np,
                        test_size=validation_split,
                        random_state=self._random_state,
                        stratify=stratify_target,
                    )
                    fit_kwargs["eval_set"] = [(x_val, y_val)]
                    self.estimator.fit(x_train, y_train, **fit_kwargs)
                    return self
                except Exception:
                    pass

            self.estimator.fit(x_np, y_np, **fit_kwargs)
            return self

        sample_weight = None
        if self._class_weight == "balanced":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_np)

        if sample_weight is not None:
            self.estimator.fit(x_np, y_np, sample_weight=sample_weight)
        else:
            self.estimator.fit(x_np, y_np)
        return self

    def _resolve_native_optimizer(self) -> tuple[Any, dict[str, Any]]:
        optim_module = importlib.import_module("torch.optim")
        name = self._optimizer_name
        params = dict(self._optimizer_cfg)
        if "name" in params:
            name = str(params.pop("name")).strip().lower()

        mapping = {
            "adam": "Adam",
            "adamw": "AdamW",
            "sgd": "SGD",
            "rmsprop": "RMSprop",
        }
        class_name = mapping.get(name)
        if class_name is None:
            _LOGGER.warning("Unsupported tabnet optimizer '%s'; falling back to adam.", name)
            class_name = "Adam"

        params.setdefault("lr", float(self._learning_rate))
        params.setdefault("weight_decay", float(self._weight_decay))
        optimizer_fn = getattr(optim_module, class_name)
        return optimizer_fn, params

    def _resolve_native_scheduler(self) -> tuple[Any | None, dict[str, Any]]:
        sched_module = importlib.import_module("torch.optim.lr_scheduler")
        name = self._scheduler_name
        params = dict(self._scheduler_cfg)
        if "name" in params:
            name = str(params.pop("name")).strip().lower()

        if name in {"", "none", "off", "disabled"}:
            return None, {}

        mapping = {
            "step": "StepLR",
            "steplr": "StepLR",
            "multistep": "MultiStepLR",
            "multisteplr": "MultiStepLR",
            "exponential": "ExponentialLR",
            "exponentiallr": "ExponentialLR",
            "cosine": "CosineAnnealingLR",
            "cosineannealing": "CosineAnnealingLR",
            "cosineannealinglr": "CosineAnnealingLR",
            "reduce_on_plateau": "ReduceLROnPlateau",
            "reduceonplateau": "ReduceLROnPlateau",
            "plateau": "ReduceLROnPlateau",
        }
        class_name = mapping.get(name)
        if class_name is None:
            _LOGGER.warning("Unsupported tabnet scheduler '%s'; disabling scheduler.", name)
            return None, {}

        if class_name == "StepLR":
            params.setdefault("step_size", int(self._scheduler_step_size))
            params.setdefault("gamma", float(self._scheduler_gamma))
        elif class_name == "MultiStepLR":
            params.setdefault(
                "milestones",
                [int(self._scheduler_step_size), int(self._scheduler_step_size) * 2],
            )
            params.setdefault("gamma", float(self._scheduler_gamma))
        elif class_name == "ExponentialLR":
            params.setdefault("gamma", float(self._scheduler_gamma))
        elif class_name == "CosineAnnealingLR":
            params.setdefault("T_max", max(int(self._max_epochs), 1))
            params.setdefault("eta_min", 0.0)
        elif class_name == "ReduceLROnPlateau":
            _LOGGER.warning(
                "TabNet backend invokes scheduler.step() without metrics; "
                "falling back from ReduceLROnPlateau to StepLR."
            )
            class_name = "StepLR"
            params = {
                "step_size": int(self._scheduler_step_size),
                "gamma": float(self._scheduler_gamma),
            }

        scheduler_fn = getattr(sched_module, class_name)
        return scheduler_fn, params

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = self.estimator.predict(self._to_numpy_features(x))
        return np.asarray(predictions).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(self._to_numpy_features(x)))

        logits = np.asarray(
            self.estimator.decision_function(self._to_numpy_features(x))
        ).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])
