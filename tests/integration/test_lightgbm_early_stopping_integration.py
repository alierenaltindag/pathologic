from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

from pathologic.models.factory import create_model


def test_lightgbm_fit_with_early_stopping_config_does_not_raise() -> None:
    x, y = make_classification(
        n_samples=120,
        n_features=12,
        n_informative=6,
        n_redundant=0,
        random_state=42,
    )
    x = x.astype(np.float32)

    model = create_model(
        "lightgbm",
        random_state=42,
        model_params={
            "n_estimators": 80,
            "learning_rate": 0.05,
            "early_stopping": {
                "enabled": True,
                "validation_split": 0.2,
                "patience": 10,
            },
        },
    )

    model.fit(x, y)
    predictions = model.predict(x)

    assert predictions.shape == (x.shape[0],)
