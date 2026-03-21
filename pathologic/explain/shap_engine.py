"""Attribution backend for explainability with optional SHAP support."""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AttributionResult:
    """Raw attribution arrays produced by an attribution backend."""

    backend: str
    contributions: np.ndarray
    global_importance: np.ndarray
    diagnostics: dict[str, str]


class ShapAttributionEngine:
    """Compute feature attributions with SHAP when available and safe fallback otherwise."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        background_size: int = 100,
        random_state: int = 42,
    ) -> None:
        self.backend = backend.strip().lower()
        self.background_size = max(int(background_size), 1)
        self.random_state = int(random_state)

    def compute(
        self,
        *,
        model: Any,
        x_background: np.ndarray,
        x_target: np.ndarray,
    ) -> AttributionResult:
        selected_backend = self._select_backend(model)

        if selected_backend != "proxy":
            shap_result, shap_error = self._try_shap(
                model=model,
                x_background=x_background,
                x_target=x_target,
                backend=selected_backend,
            )
            if shap_result is not None:
                return shap_result
            if self.backend not in {"auto", "proxy"}:
                raise RuntimeError(
                    "Explainability backend was requested but SHAP initialization failed. "
                    f"Requested backend: {selected_backend}. "
                    f"Reason: {shap_error or 'unknown_error'}"
                )

            return self._proxy_contributions(
                model=model,
                x_background=x_background,
                x_target=x_target,
                diagnostics={
                    "fallback_from": selected_backend,
                    "fallback_reason": str(shap_error or "shap_initialization_failed"),
                },
            )

        return self._proxy_contributions(
            model=model,
            x_background=x_background,
            x_target=x_target,
            diagnostics={
                "fallback_reason": (
                    "hybrid_ensemble_auto_proxy"
                    if self.backend == "auto" and self._is_hybrid_ensemble(model)
                    else "proxy_backend_requested_or_selected"
                )
            },
        )

    def _try_shap(
        self,
        *,
        model: Any,
        x_background: np.ndarray,
        x_target: np.ndarray,
        backend: str,
    ) -> tuple[AttributionResult | None, str | None]:
        try:
            import shap  # type: ignore[import-not-found]
        except Exception as exc:
            return None, f"import_error: {exc}"

        prediction_fn = self._predict_positive_probability
        background = self._sample_background(x_background)

        try:
            with self._deterministic_context():
                model_obj = self._resolve_model_object(model)
                if backend == "tree":
                    explainer = shap.TreeExplainer(
                        model_obj,
                        background,
                        feature_perturbation="interventional",
                        model_output="probability",
                    )
                    values = self._normalize_values(explainer.shap_values(x_target))
                    return AttributionResult(
                        backend="tree_shap",
                        contributions=values,
                        global_importance=np.mean(np.abs(values), axis=0),
                        diagnostics={},
                    ), None

                if backend == "linear":
                    explainer = shap.LinearExplainer(model_obj, background)
                    values = self._normalize_values(explainer.shap_values(x_target))
                    return AttributionResult(
                        backend="linear_shap",
                        contributions=values,
                        global_importance=np.mean(np.abs(values), axis=0),
                        diagnostics={},
                    ), None

                if backend == "deep":
                    try:
                        import torch
                    except Exception:
                        return None, "torch_import_error"
                    torch_model = self._resolve_torch_model(model)
                    if torch_model is None:
                        return None, "torch_model_unavailable"

                    device = next(torch_model.parameters()).device
                    background_tensor = torch.tensor(background, dtype=torch.float32, device=device)
                    target_tensor = torch.tensor(x_target, dtype=torch.float32, device=device)
                    torch_model.eval()

                    explainer = shap.DeepExplainer(torch_model, background_tensor)
                    values = self._normalize_values(explainer.shap_values(target_tensor))
                    return AttributionResult(
                        backend="deep_shap",
                        contributions=values,
                        global_importance=np.mean(np.abs(values), axis=0),
                        diagnostics={},
                    ), None

                explainer = shap.Explainer(lambda values: prediction_fn(model, values), background)
                explained = explainer(x_target)
                values = self._normalize_values(explained.values)
                return AttributionResult(
                    backend="shap",
                    contributions=values,
                    global_importance=np.mean(np.abs(values), axis=0),
                    diagnostics={},
                ), None
        except Exception as exc:
            return None, f"shap_runtime_error: {exc}"

    def _proxy_contributions(
        self,
        *,
        model: Any,
        x_background: np.ndarray,
        x_target: np.ndarray,
        diagnostics: dict[str, str] | None = None,
    ) -> AttributionResult:
        with self._deterministic_context():
            rng = np.random.default_rng(self.random_state)
            baseline_scores = self._predict_positive_probability(model, x_target)

            feature_count = x_target.shape[1]
            importances = np.zeros(feature_count, dtype=float)
            for index in range(feature_count):
                perturbed = np.array(x_target, copy=True)
                column_values = np.array(x_background[:, index], copy=True)
                rng.shuffle(column_values)
                if column_values.shape[0] >= x_target.shape[0]:
                    perturbed[:, index] = column_values[: x_target.shape[0]]
                else:
                    repeats = int(np.ceil(x_target.shape[0] / column_values.shape[0]))
                    tiled = np.tile(column_values, repeats)
                    perturbed[:, index] = tiled[: x_target.shape[0]]
                shifted_scores = self._predict_positive_probability(model, perturbed)
                importances[index] = float(np.mean(np.abs(baseline_scores - shifted_scores)))

            background_mean = np.mean(x_background, axis=0)
            centered = x_target - background_mean
            contributions = centered * importances.reshape(1, -1)

        return AttributionResult(
            backend="proxy",
            contributions=contributions,
            global_importance=np.asarray(importances, dtype=float),
            diagnostics=dict(diagnostics or {}),
        )

    def _select_backend(self, model: Any) -> str:
        if self.backend in {"proxy", "tree", "linear", "deep", "shap"}:
            return self.backend

        if self._is_hybrid_ensemble(model):
            return "proxy"

        model_obj = self._resolve_model_object(model)
        class_name = type(model_obj).__name__.lower()
        module_name = type(model_obj).__module__.lower()
        joined = f"{module_name}.{class_name}"

        # FAST EXIT: Ensemble wrappers are notoriously slow for exact permutation SHAP
        # Force proxy (permutation sampling) for them even if they contain Torch/XGB members.
        if any(term in class_name for term in ("ensemble", "voting", "stacking", "blending")):
            return "proxy"

        if "torch" in module_name or self._is_torch_model(model_obj):
            return "deep"
        if any(token in joined for token in ("forest", "boost", "xgb", "tree", "catboost")):
            return "tree"
        if any(token in joined for token in ("logistic", "linear", "ridge", "lasso")):
            return "linear"
        return "shap"

    @staticmethod
    def _resolve_model_object(model: Any) -> Any:
        if hasattr(model, "estimator"):
            return model.estimator
        if hasattr(model, "model"):
            return model.model
        return model

    @staticmethod
    def _resolve_torch_model(model: Any) -> Any | None:
        candidate = ShapAttributionEngine._resolve_model_object(model)
        if ShapAttributionEngine._is_torch_model(candidate):
            return candidate
        if ShapAttributionEngine._is_torch_model(model):
            return model
        return None

    @staticmethod
    def _is_torch_model(model: Any) -> bool:
        return bool(
            hasattr(model, "parameters")
            and callable(getattr(model, "parameters"))
            and hasattr(model, "eval")
        )

    @staticmethod
    def _is_hybrid_ensemble(model: Any) -> bool:
        aliases = getattr(model, "member_aliases", None)
        members = getattr(model, "_member_models", None)
        return isinstance(aliases, list) and isinstance(members, list) and len(aliases) >= 2

    @staticmethod
    def _normalize_values(values_raw: Any) -> np.ndarray:
        values = np.asarray(values_raw)
        if values.ndim == 3:
            values = values[:, :, -1]
        if values.ndim != 2:
            raise ValueError("Attribution backend returned unexpected shape.")
        return values

    @contextmanager
    def _deterministic_context(self) -> Any:
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        torch_module = None
        previous_benchmark: bool | None = None
        previous_cudnn_deterministic: bool | None = None
        try:
            import torch

            torch_module = torch
            torch.manual_seed(self.random_state)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                previous_benchmark = bool(torch.backends.cudnn.benchmark)
                previous_cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            torch_module = None

        try:
            yield
        finally:
            if torch_module is not None:
                try:
                    torch_module.use_deterministic_algorithms(False)
                except Exception:
                    pass
                if previous_benchmark is not None:
                    torch_module.backends.cudnn.benchmark = previous_benchmark
                if previous_cudnn_deterministic is not None:
                    torch_module.backends.cudnn.deterministic = previous_cudnn_deterministic

    def _sample_background(self, x_background: np.ndarray) -> np.ndarray:
        if x_background.shape[0] <= self.background_size:
            return np.asarray(x_background, dtype=float)

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(x_background.shape[0], size=self.background_size, replace=False)
        return np.asarray(x_background[indices], dtype=float)

    @staticmethod
    def _predict_positive_probability(model: Any, x_values: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(x_values))
        if probabilities.ndim == 1:
            return probabilities.reshape(-1)
        return probabilities[:, -1].reshape(-1)
