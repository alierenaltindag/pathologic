# PathoLogic Veri Formati ve Sabitler

## Zorunlu Sutunlar

- `Target`: ikili hedef (0/1)
- `Gene(s)`: leakage-safe bolme icin grup kimligi
- `data.required_features` altinda listelenen engineered feature sutunlari

Varsayilan config referansi: [pathologic/configs/defaults.yaml](pathologic/configs/defaults.yaml).

## Minimal Ornek

```csv
VariationID,Gene(s),Target,REVEL_Score,cadd.phred,gnomAD_is_zero,gnomAD_log,cpg_flag,proline_intro,cysteine_intro,proline_remove
v1,G1,1,0.91,28.4,1,-8.0,1,0,0,0
v2,G1,0,0.07,7.2,0,-2.1,0,1,0,1
v3,G2,1,0.84,24.8,0,-3.2,1,0,1,0
v4,G2,0,0.15,10.1,0,-1.4,0,0,0,0
```

## Metadata Sutunlari (Train'de Kullanilmaz)

- `VariationID`
- `Gene(s)`
- `Protein change`
- `Veri_Kaynagi_Paneli`

Bu sutunlar explainability ve hata analizi raporlarinda korunur.

## Split ve Leakage Kurallari

- `Gene(s)` varsa varsayilan olarak grouped split kullanilir.
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
  label_column: Target
  gene_column: Gene(s)
  required_features:
    - REVEL_Score
    - cadd.phred
    - gnomAD_is_zero
    - gnomAD_log
    - cpg_flag
  excluded_columns:
    - VariationID
    - Protein change
    - Gene(s)
    - Veri_Kaynagi_Paneli
```
