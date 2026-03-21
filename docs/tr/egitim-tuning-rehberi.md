# PathoLogic Egitim ve Tuning Rehberi

## Egitim Akisi

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
model.train("variants.csv")
report = model.evaluate("variants.csv")
print(report["metrics"])
```

## Split Konfigurasyonu

Split ayarlari [pathologic/configs/runtime/train.yaml](pathologic/configs/runtime/train.yaml) dosyasindadir.

Cross-validation modu:

```yaml
split:
  mode: cross_validation
  cross_validation:
    n_splits: 3
    stratified: true
```

Holdout modu:

```yaml
split:
  mode: holdout
  holdout:
    test_size: 0.2
    val_size: 0.2
    stratified: true
```

## Tuning Akisi

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
result = model.tune("variants.csv")
print(result["best_params"])
print(result["best_score"])
```

Tuning runtime config: [pathologic/configs/runtime/tune.yaml](pathologic/configs/runtime/tune.yaml).

## Reproducibility

- [pathologic/configs/defaults.yaml](pathologic/configs/defaults.yaml) icinde sabit `seed` kullanin.
- Karsilastirmali kosular icin ayni veri ve ayni config'i koruyun.
- Gen bazli leakage riski olan senaryolarda grouped split'i kapatmayin.

## Donanim Davranisi

Device fallback sirasi CUDA -> MPS -> CPU.

Uygulama referanslari:

- [pathologic/utils/hardware.py](pathologic/utils/hardware.py)
- [pathologic/engine/trainer.py](pathologic/engine/trainer.py)

## Ilgili Ornekler

- [docs/examples/example_01_basic_workflow.py](docs/examples/example_01_basic_workflow.py)
- [docs/examples/example_02_ensemble_builder.py](docs/examples/example_02_ensemble_builder.py)
- [docs/examples/example_03_finetune.py](docs/examples/example_03_finetune.py)
