"""Integration test for Phase 5 NAS candidate selection and trainer integration."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from pathologic.engine import Trainer, TrainerConfig
from pathologic.models import create_model
from pathologic.search.nas import NASearch


@pytest.mark.integration
def test_nas_selects_candidate_and_trains_with_trainer() -> None:
    x, y = make_classification(
        n_samples=140,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        random_state=42,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    search = NASearch(
        strategy="low_fidelity",
        random_state=42,
        max_evaluations=8,
        direction="maximize",
    )

    def evaluate_candidate(params: dict[str, float]) -> float:
        model = create_model(
            "logreg",
            random_state=42,
            model_params={"c": float(params["c"]), "max_iter": int(params["max_iter"])},
        )
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_val)).reshape(-1)
        return float((predictions == y_val).mean())

    result = search.search(
        search_space={
            "c": {"type": "float", "low": 0.1, "high": 3.0},
            "max_iter": {"type": "int", "low": 120, "high": 300, "step": 60},
        },
        n_candidates=12,
        evaluate_candidate=evaluate_candidate,
        budget={"min_fidelity": 1, "max_fidelity": 3},
    )

    best_model = create_model(
        "logreg",
        random_state=42,
        model_params={
            "c": float(result.best_candidate.params["c"]),
            "max_iter": int(result.best_candidate.params["max_iter"]),
        },
    )
    trainer = Trainer(TrainerConfig(device="cpu", mixed_precision=False))
    train_result = trainer.fit(
        model=best_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )

    assert result.best_score >= 0.5
    assert train_result.model is not None
    assert "f1" in train_result.metrics
