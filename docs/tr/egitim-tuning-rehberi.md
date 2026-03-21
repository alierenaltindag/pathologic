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

## search_best_model CLI Flag Referansi

Script: [scripts/search_best_model.py](scripts/search_best_model.py)

Temel kullanim:

```bash
python scripts/search_best_model.py data.csv --output-dir results/model_search
```

### Girdi ve cikti

- `data_csv`: Zorunlu positional arguman. Giris CSV dosya yolu.
- `--output-dir`: Sonuc artefaktlarinin yazilacagi dizin. Varsayilan `results/model_search`.

### Arama kontrolu

- `--seed`: Tum surec icin rastgelelik tohumu.
- `--objective`: Aday secim metrigi (`f1`, `roc_auc`, vb.).
- `--budget-profile`: Butce profili (`quick`, `balanced`, `aggressive`).
- `--model-pool`: Denenecek model alias listesi (virgulle).
- `--exclude-models`: Havuzdan cikarilacak model alias listesi (virgulle).
- `--disable-hybrids`: Hibrit aday olusumunu kapatir.
- `--max-hybrid-combination-size`: Hibritte maksimum uye sayisi (2=pair, 3=triple).
- `--max-candidates`: Toplam degerlendirilecek aday sayisina ust sinir koyar.

### HPO ve NAS

- `--tune-engine`: HPO motoru (`optuna`, `random`, `grid`).
- `--nas-strategy`: NAS stratejisi (`low_fidelity`, `weight_sharing`).
- `--n-trials`: HPO deneme sayisini override eder.
- `--nas-candidates`: NAS aday sayisini override eder.
- `--cv-splits`: Ic CV fold sayisini override eder.

### Dis holdout split

- `--outer-test-size`: Dis test holdout orani.
- `--outer-val-size`: Dis validation/calibration holdout orani.

### Hibrit strateji ve agirliklar

- `--hybrid-strategy`: Hibrit birlestirme stratejisi (`soft_voting`, `hard_voting`, `stacking`, `blending`).
- `--hybrid-tune-strategy-and-params`: Hibritte strateji/parametre arama uzayini acik hale getirir.
- `--hybrid-weights`: Manuel uye agirliklari (ornek: `0.5,0.3,0.2`).
- `--hybrid-weighting-policy`: Agirlik politikasi (`auto`, `manual`, `equal`, `inverse_error`, `objective_proportional`).
- `--hybrid-weighting-objective`: Dinamik agirliklandirmada kullanilan hedef metrik.
- `--disable-hybrid-normalize-weights`: Verilen agirliklarin normalize edilmesini kapatir.
- `--hybrid-meta-model`: Stacking/blending icin meta model alias'i.
- `--hybrid-stacking-cv`: Stacking icin CV fold sayisi.
- `--hybrid-blend-size`: Blending icin holdout orani.

### Regularization arama uzayi

- `--regularization-profile`: Regularization davranisi (`auto`, `off`).
- `--regularization-models`: Regularization optimizasyonu uygulanacak model listesi.
- `--optimize-regularization-in-nas`: NAS arama uzayinda regularization parametrelerini aktif eder.

### Kalibrasyon

- `--calibration-bins`: ECE/reliability histogram bin sayisi.
- `--calibration-weight-objective`: Kazanan seciminde objective metric agirligi.
- `--calibration-weight-ece`: Kazanan seciminde ECE ceza agirligi.
- `--calibration-weight-brier`: Kazanan seciminde Brier ceza agirligi.

### Aciklanabilirlik ve hata analizi

- `--disable-explainability`: Explainability artefaktlarini kapatir.
- `--disable-error-analysis`: Error analysis artefaktlarini kapatir.
- `--error-analysis-mode`: Hata analizi seviyesi (`summary`, `full`, `hybrid`).
- `--explain-top-k-features`: Explainability ozetinde tutulacak top-k ozellik.
- `--explain-top-k-samples`: Sample-level aciklamalarda tutulacak top-k ornek.
- `--explain-background-size`: Attribution backend icin background ornek sayisi.
- `--explain-fp-top-k`: False-positive hotspot top-k degeri.
- `--explain-fp-min-negative-count`: Hotspot zenginlestirmesi icin minimum negatif destek.

### Calisma davranisi ve loglama

- `--verbose-inner-search`: HPO/NAS ic loglarini ayrintili yazdirir.
- `--delete-prepared`: Kosu sonunda hazirlanmis dataset CSV dosyasini siler.

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
