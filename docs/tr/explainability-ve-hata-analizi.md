# PathoLogic Explainability ve Hata Analizi Rehberi

Bu rehber, Faz 6 ile gelen explainability akisinin nasil calistigini ve nasil kullanilacagini aciklar.

## Hedef

- Model tahminlerini feature katkisi ile aciklamak
- Biyolojik olarak yorumlanabilir aciklama metni uretmek
- False-positive hotspot gruplarini raporlamak
- Tum ciktilari denetlenebilir metadata ile saklamak

## Akis Ozeti

1. `PathoLogic.train(data)` ile model egitilir.
2. `PathoLogic.explain(data)` cagrisi yapilir.
3. Sistem su bloklari uretir:
   - `global_feature_importance`
   - `sample_explanations`
   - `false_positive_hotspots`
   - `metadata`
   - `visual_report_html`

## Konfigurasyon

Explain runtime ayarlari [pathologic/configs/runtime/explain.yaml](pathologic/configs/runtime/explain.yaml) dosyasindadir.

Ana alanlar:

- `backend`: `auto`, `tree`, `linear`, `deep`, `shap`, `proxy`
- `background_size`: SHAP background ornek sayisi
- `top_k_features`: raporlanacak en etkili feature sayisi
- `top_k_samples`: sample-level raporlanacak satir sayisi
- `group_columns`: hotspot analizi icin grup kolonlari
- `biological_mapping`: teknik feature -> biyolojik etiket map'i
- `false_positive.*`: hotspot filtreleri
- `visual_report.*`: HTML rapor ayarlari

## Backend Secim Mantigi

`backend: auto` iken sistem model tipine gore en uygun backend'i secer:

- tree tabanli modeller -> `tree_shap`
- linear modeller -> `linear_shap`
- torch tabanli modeller -> `deep_shap`
- desteklenmeyen veya hata veren durumlar -> `proxy`

Bu davranis [pathologic/explain/shap_engine.py](pathologic/explain/shap_engine.py) dosyasinda uygulanir.

## Leakage ve Determinism Notlari

- Explain background varsayilan olarak train splitten gelir (`background_source: train_split`).
- Explain hesaplamasi deterministic context ile calisir (seed, numpy, torch, cuDNN ayarlari).
- Bu sayede ayni seed ve ayni veri ile tekrar calistirilabilir rapor elde edilir.

## API Ornegi

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
model.train("data/variants.csv")
report = model.explain("data/variants.csv")

print(report["backend"])
print(report["metadata"]["background_source"])
print(report["global_feature_importance"][:3])
```

## HTML Raporu Dosyaya Yazma

`pathologic/configs/runtime/explain.yaml` icinde:

```yaml
visual_report:
  enabled: true
  save_path: reports/explain/latest_report.html
```

Bu ayarla `model.explain(...)` cagrisi sonunda HTML rapor otomatik yazilir.

## Cikti Sozlesmesi

Rapor semasi [pathologic/explain/schemas.py](pathologic/explain/schemas.py) icinde tanimlidir.

Beklenen anahtarlar:

- `backend`
- `global_feature_importance`
- `sample_explanations`
- `false_positive_hotspots`
- `metadata`
- `visual_report_html` (visual report etkinse)

## Test Kapsami

- Unit:
  - [tests/unit/test_explainability_unit.py](tests/unit/test_explainability_unit.py)
  - [tests/unit/test_shap_engine_unit.py](tests/unit/test_shap_engine_unit.py)
- Integration:
  - [tests/integration/test_phase6_explainability_integration.py](tests/integration/test_phase6_explainability_integration.py)

## Operasyonel Oneri

- Klinik/audit senaryolarinda `save_path` aktif tutun.
- `group_columns` icine kurumunuzdaki biyolojik gruplari ekleyin.
- `biological_mapping` map'ini proje veri sozlugune gore doldurun.
