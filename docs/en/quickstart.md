# PathoLogic Quickstart

This guide gets you from raw CSV to first prediction in about 30 minutes.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## 2. Minimal Workflow

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
model.train("variants.csv")
predictions = model.predict("variants.csv")
report = model.evaluate("variants.csv")

print(predictions[0])
print(report["metrics"])
```

Reference implementation: [docs/examples/example_01_basic_workflow.py](docs/examples/example_01_basic_workflow.py).

## 3. Ensemble Workflow

```python
from pathologic import PathoLogic

builder = (
    PathoLogic.builder()
    .add_model("logreg", c=1.0)
    .add_model("random_forest", n_estimators=100)
    .add_model("hist_gbdt", learning_rate=0.1)
    .strategy("stacking", cv=3)
    .meta_model("logreg", c=0.7)
)

ensemble = PathoLogic.from_builder(builder)
ensemble.train("variants.csv")
print(ensemble.predict("variants.csv")[:2])
```

Reference implementation: [docs/examples/example_02_ensemble_builder.py](docs/examples/example_02_ensemble_builder.py).

## 4. Fine-Tune Workflow

```python
from pathologic import PathoLogic

model = PathoLogic("mlp")
model.train("variants.csv")
ft = model.fine_tune(
    "variants.csv",
    freeze_layers="backbone_last2",
    learning_rate=0.0005,
    epochs=5,
)
print(ft["metric_delta"])
```

Reference implementation: [docs/examples/example_03_finetune.py](docs/examples/example_03_finetune.py).

## 5. Explainability Workflow

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
model.train("variants.csv")
report = model.explain("variants.csv")
print(report["backend"])
print(report["global_feature_importance"][:3])
```

Reference implementation: [docs/examples/example_04_explainability_report.py](docs/examples/example_04_explainability_report.py).

## 6. Runtime Kwargs Overrides

All core API methods accept optional kwargs overrides. These values override runtime YAML defaults only for that call.

```python
from pathologic import PathoLogic

model = PathoLogic("mlp")

# train() overrides
model.train(
    "variants.csv",
    validation_split=0.3,
    learning_rate=0.0008,
    batch_size=128,
    split={"mode": "holdout", "holdout": {"test_size": 0.2, "val_size": 0.2}},
    preprocess={"impute_strategy": "median", "scaler": "standard"},
)

# evaluate()/predict() overrides
eval_report = model.evaluate("variants.csv", threshold=0.6, metrics=["f1", "roc_auc"])
pred_rows = model.predict("variants.csv", threshold=0.6)

# fine_tune() overrides
ft_report = model.fine_tune(
    "variants.csv",
    freeze_layers="none",
    learning_rate=0.001,
    epochs=3,
    scheduler={"name": "reduce_on_plateau", "patience": 2},
)

print(eval_report["metrics"])
print(pred_rows[0])
print(ft_report["metric_delta"])
```

Notes:

- Unknown kwargs raise an explicit error.
- Method kwargs precedence is: call kwargs > runtime config > defaults.
- Model-specific overrides are supported (for example, MLP `learning_rate`, `epochs`, `batch_size`).

## Next Docs

- Data schema: [docs/en/data-schema.md](docs/en/data-schema.md)
- Model extension: [docs/en/model-plugin-guide.md](docs/en/model-plugin-guide.md)
- Train and tune: [docs/en/training-and-tuning.md](docs/en/training-and-tuning.md)
- Explainability details: [docs/en/explainability-and-error-analysis.md](docs/en/explainability-and-error-analysis.md)
