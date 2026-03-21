# PathoLogic Model Ekleme Rehberi

Bu rehber, registry tabanli model ekleme akisinin temelini aciklar.

## Wrapper Sozlesmesi

Her model wrapper su metotlari saglamalidir:

- `fit(x, y)`
- `predict(x)`
- `predict_proba(x)`

Mevcut wrapper ornekleri:

- [pathologic/models/zoo/sklearn_models.py](pathologic/models/zoo/sklearn_models.py)
- [pathologic/models/zoo/mlp.py](pathologic/models/zoo/mlp.py)

## Modeli Registry'ye Kaydet

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

## Config Ekle

Model varsayilanlarini [pathologic/configs/models](pathologic/configs/models) altina ekleyin.

Tipik alanlar:

- cekirdek hiperparametreler
- tuning icin `tuning_search_space`

## Plugin Dogrulama

Onerilen kontroller:

- Wrapper kontrat testleri: [tests/unit/test_model_wrappers_unit.py](tests/unit/test_model_wrappers_unit.py)
- Registry testleri: [tests/unit/test_registry_unit.py](tests/unit/test_registry_unit.py)
- Uctan uca entegrasyon: [tests/integration/test_phase3_model_integration.py](tests/integration/test_phase3_model_integration.py)
