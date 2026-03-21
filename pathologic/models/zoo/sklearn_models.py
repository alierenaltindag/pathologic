"""Native scikit-learn model wrappers."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pathologic.models.registry import register


@register(name="random_forest", family="sklearn-tree")
class RandomForestWrapper:
    """RandomForest classifier wrapper."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float | int | None = "sqrt",
        class_weight: str | dict[int, float] | None = None,
        n_jobs: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> RandomForestWrapper:
        self.estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(x)).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(x))


@register(name="hist_gbdt", family="sklearn-boosting")
class HistGradientBoostingWrapper:
    """HistGradientBoosting classifier wrapper."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        early_stopping: str | bool = "auto",
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        class_weight: str | dict[int, float] | None = None,
        random_state: int = 42,
    ) -> None:
        self.estimator = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            class_weight=class_weight,
            random_state=random_state,
        )
        self._validation_fraction = float(validation_fraction)

    def fit(self, x: np.ndarray, y: np.ndarray) -> HistGradientBoostingWrapper:
        # Small datasets can fail stratified internal validation splitting.
        if bool(self.estimator.early_stopping):
            n_samples = int(np.asarray(y).shape[0])
            n_classes = int(np.unique(y).size)
            validation_size = int(np.ceil(n_samples * self._validation_fraction))
            if n_classes > 1 and validation_size < n_classes:
                self.estimator.set_params(early_stopping=False)
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


@register(name="logreg", family="sklearn-linear")
class LogisticRegressionWrapper:
    """LogisticRegression classifier wrapper."""

    def __init__(
        self,
        *,
        c: float = 1.0,
        max_iter: int = 400,
        solver: str = "lbfgs",
        tol: float = 1e-4,
        fit_intercept: bool = True,
        class_weight: str | dict[int, float] | None = None,
        random_state: int = 42,
    ) -> None:
        self.estimator = LogisticRegression(
            C=c,
            max_iter=max_iter,
            solver=solver,
            tol=tol,
            fit_intercept=fit_intercept,
            class_weight=class_weight,
            random_state=random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticRegressionWrapper:
        self.estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(x)).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(x))
