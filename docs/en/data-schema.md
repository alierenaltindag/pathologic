# PathoLogic Data Schema

## Required Columns

- `Target`: binary target (0/1)
- `Gene(s)`: group identifier for leakage-safe splitting
- engineered feature columns listed under `data.required_features`

Default config reference: [pathologic/configs/defaults.yaml](pathologic/configs/defaults.yaml).

## Minimal Example

```csv
VariationID,Gene(s),Target,REVEL_Score,cadd.phred,gnomAD_is_zero,gnomAD_log,cpg_flag,proline_intro,cysteine_intro,proline_remove
v1,G1,1,0.91,28.4,1,-8.0,1,0,0,0
v2,G1,0,0.07,7.2,0,-2.1,0,1,0,1
v3,G2,1,0.84,24.8,0,-3.2,1,0,1,0
v4,G2,0,0.15,10.1,0,-1.4,0,0,0,0
```

## Metadata Columns (Not Used For Training)

- `VariationID`
- `Gene(s)`
- `Protein change`
- `Veri_Kaynagi_Paneli`

These are retained for explainability and error analysis reporting.

## Split and Leakage Rules

- If `Gene(s)` exists, grouped split is used by default.
- The same gene must not appear in both train and validation folds.
- Preprocessing artifacts are fit on train fold only.

Implementation references:

- [pathologic/data/loader.py](pathologic/data/loader.py)
- [pathologic/data/preprocessor.py](pathologic/data/preprocessor.py)

Test references:

- [tests/unit/test_loader_unit.py](tests/unit/test_loader_unit.py)
- [tests/integration/test_phase2_leakage_integration.py](tests/integration/test_phase2_leakage_integration.py)
- [tests/integration/test_phase2_preprocess_artifact_integration.py](tests/integration/test_phase2_preprocess_artifact_integration.py)

## Config Snippet

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
