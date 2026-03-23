from __future__ import annotations

import numpy as np
import pandas as pd

from pathologic.models.zoo.tabnet import TabNetWrapper


class _FakeTabNetClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, x, y, **kwargs):
        if not isinstance(x, np.ndarray):
            raise TypeError("expected numpy x")
        if not isinstance(y, np.ndarray):
            raise TypeError("expected numpy y")
        return self

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("expected numpy x")
        return np.zeros(x.shape[0], dtype=int)

    def predict_proba(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("expected numpy x")
        probs = np.full((x.shape[0], 2), 0.5, dtype=float)
        return probs


class _FakeTabNetModule:
    TabNetClassifier = _FakeTabNetClassifier


def test_tabnet_wrapper_converts_dataframe_input_to_numpy(monkeypatch) -> None:
    def _fake_import_module(name: str):
        if name == "pytorch_tabnet.tab_model":
            return _FakeTabNetModule
        if name == "torch.optim":
            class _Optim:
                class Adam:
                    pass

                class AdamW:
                    pass

                class SGD:
                    pass

                class RMSprop:
                    pass

            return _Optim
        if name == "torch.optim.lr_scheduler":
            class _Sched:
                class StepLR:
                    pass

            return _Sched
        raise ImportError(name)

    monkeypatch.setattr(
        "pathologic.models.zoo.tabnet.importlib.import_module",
        _fake_import_module,
    )

    model = TabNetWrapper(random_state=42)
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0, 1, 0, 1])

    model.fit(x, y)
    preds = model.predict(x)
    probs = model.predict_proba(x)

    assert preds.shape == (4,)
    assert probs.shape == (4, 2)
