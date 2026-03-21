# PathoLogic Training and Tuning Guide

## Train Flow

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
model.train("variants.csv")
report = model.evaluate("variants.csv")
print(report["metrics"])
```

## Split Configuration

Runtime split config lives in [pathologic/configs/runtime/train.yaml](pathologic/configs/runtime/train.yaml).

Cross-validation mode:

```yaml
split:
  mode: cross_validation
  cross_validation:
    n_splits: 3
    stratified: true
```

Holdout mode:

```yaml
split:
  mode: holdout
  holdout:
    test_size: 0.2
    val_size: 0.2
    stratified: true
```

## Tune Flow

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
result = model.tune("variants.csv")
print(result["best_params"])
print(result["best_score"])
```

Tune runtime config: [pathologic/configs/runtime/tune.yaml](pathologic/configs/runtime/tune.yaml).

## Reproducibility

- Keep a fixed `seed` in [pathologic/configs/defaults.yaml](pathologic/configs/defaults.yaml).
- Reuse same dataset and config for comparable runs.
- Keep grouped split enabled when gene-level leakage risk exists.

## Hardware Behavior

Device fallback order is CUDA -> MPS -> CPU.

Implementation references:

- [pathologic/utils/hardware.py](pathologic/utils/hardware.py)
- [pathologic/engine/trainer.py](pathologic/engine/trainer.py)

## Related Examples

- [docs/examples/example_01_basic_workflow.py](docs/examples/example_01_basic_workflow.py)
- [docs/examples/example_02_ensemble_builder.py](docs/examples/example_02_ensemble_builder.py)
- [docs/examples/example_03_finetune.py](docs/examples/example_03_finetune.py)
