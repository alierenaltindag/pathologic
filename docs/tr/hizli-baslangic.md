# PathoLogic Hizli Baslangic

Bu rehber, ham CSV dosyasindan ilk tahmine yaklasik 30 dakikada ulasmanizi hedefler.

## 1. Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## 2. Minimal Akis

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
model.train("variants.csv")
predictions = model.predict("variants.csv")
report = model.evaluate("variants.csv")

print(predictions[0])
print(report["metrics"])
```

Referans uygulama: [docs/examples/example_01_basic_workflow.py](docs/examples/example_01_basic_workflow.py).

## 3. Ensemble Akisi

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

Referans uygulama: [docs/examples/example_02_ensemble_builder.py](docs/examples/example_02_ensemble_builder.py).

## 4. Fine-Tune Akisi

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

Referans uygulama: [docs/examples/example_03_finetune.py](docs/examples/example_03_finetune.py).

## 5. Explainability Akisi

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
model.train("variants.csv")
report = model.explain("variants.csv")
print(report["backend"])
print(report["global_feature_importance"][:3])
```

Referans uygulama: [docs/examples/example_04_explainability_report.py](docs/examples/example_04_explainability_report.py).

## 6. Runtime Kwargs Override

Tum temel API metotlari opsiyonel kwargs override alir. Bu degerler sadece o cagri icin runtime YAML varsayilanlarini ezer.

```python
from pathologic import PathoLogic

model = PathoLogic("mlp")

# train() override ornegi
model.train(
    "variants.csv",
    validation_split=0.3,
    learning_rate=0.0008,
    batch_size=128,
    split={"mode": "holdout", "holdout": {"test_size": 0.2, "val_size": 0.2}},
    preprocess={"impute_strategy": "median", "scaler": "standard"},
)

# evaluate()/predict() override ornegi
eval_report = model.evaluate("variants.csv", threshold=0.6, metrics=["f1", "roc_auc"])
pred_rows = model.predict("variants.csv", threshold=0.6)

# fine_tune() override ornegi
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

Notlar:

- Bilinmeyen kwargs anahtarlari acik hata uretir.
- Oncelik sirasi: cagri kwargs > runtime config > defaults.
- Modele ozel override desteklenir (ornek: MLP icin `learning_rate`, `epochs`, `batch_size`).

## Sonraki Dokumanlar

- Veri semasi: [docs/tr/veri-formati-ve-sabitler.md](docs/tr/veri-formati-ve-sabitler.md)
- Model ekleme: [docs/tr/model-ekleme-rehberi.md](docs/tr/model-ekleme-rehberi.md)
- Egitim ve tuning: [docs/tr/egitim-tuning-rehberi.md](docs/tr/egitim-tuning-rehberi.md)
- Explainability detaylari: [docs/tr/explainability-ve-hata-analizi.md](docs/tr/explainability-ve-hata-analizi.md)
