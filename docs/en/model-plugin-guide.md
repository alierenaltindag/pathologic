# PathoLogic Model Plugin Guide

This guide explains how to add a new model through the registry.

## Wrapper Contract

Every model wrapper should provide:

- `fit(x, y)`
- `predict(x)`
- `predict_proba(x)`

See existing wrappers in:

- [pathologic/models/zoo/sklearn_models.py](pathologic/models/zoo/sklearn_models.py)
- [pathologic/models/zoo/mlp.py](pathologic/models/zoo/mlp.py)

## Register a Model

```python
from pathologic.models.registry import register

@register(
    name="my_model",
    family="sklearn-custom",
    explainability_supported=True,
    supports_predict_proba=True,
)
class MyModelWrapper:
    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit(self, x, y):
        return self

    def predict(self, x):
        ...

    def predict_proba(self, x):
        ...
```

## Add Config

Add model defaults under [pathologic/configs/models](pathologic/configs/models).

Example keys:

- core hyperparameters
- `tuning_search_space` for tune workflow

## Validate Plugin

Recommended checks:

- Unit wrapper contract tests in [tests/unit/test_model_wrappers_unit.py](tests/unit/test_model_wrappers_unit.py)
- Registry tests in [tests/unit/test_registry_unit.py](tests/unit/test_registry_unit.py)
- End-to-end integration in [tests/integration/test_phase3_model_integration.py](tests/integration/test_phase3_model_integration.py)
