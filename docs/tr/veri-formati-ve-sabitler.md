# PathoLogic Veri Formati ve Sabitler

## Zorunlu Sutunlar

- `variant_id`: satir kimligi
- `gene_id`: leakage-safe bolme icin grup kimligi
- `label`: ikili hedef (0/1)
- `data.required_features` altinda listelenen feature sutunlari

Varsayilan config referansi: [pathologic/configs/defaults.yaml](pathologic/configs/defaults.yaml).

## Minimal Ornek

```csv
variant_id,gene_id,label,feat_a,feat_b
v1,G1,1,0.1,1.0
v2,G1,0,0.2,1.1
v3,G2,1,0.3,0.9
v4,G2,0,0.4,1.2
```

## Opsiyonel Sutunlar

- `domain_id`
- `protein_family`

Bu sutunlar grup bazli hata analizi ve hotspot kalitesini artirir.

## Split ve Leakage Kurallari

- `gene_id` varsa varsayilan olarak grouped split kullanilir.
- Ayni gen train ve validation foldlarinda birlikte bulunmamalidir.
- Preprocess artifact'lari yalniz train fold uzerinde fit edilir.

Uygulama referanslari:

- [pathologic/data/loader.py](pathologic/data/loader.py)
- [pathologic/data/preprocessor.py](pathologic/data/preprocessor.py)

Test referanslari:

- [tests/unit/test_loader_unit.py](tests/unit/test_loader_unit.py)
- [tests/integration/test_phase2_leakage_integration.py](tests/integration/test_phase2_leakage_integration.py)
- [tests/integration/test_phase2_preprocess_artifact_integration.py](tests/integration/test_phase2_preprocess_artifact_integration.py)

## Config Ornegi

```yaml
data:
  label_column: label
  gene_column: gene_id
  required_features:
    - feat_a
    - feat_b
```
