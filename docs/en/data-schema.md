# PathoLogic Data Schema

## Required Columns

- `variant_id`: unique row identifier
- `gene_id`: group identifier for leakage-safe splitting
- `label`: binary target (0/1)
- feature columns listed under `data.required_features`

Default config reference: [pathologic/configs/defaults.yaml](pathologic/configs/defaults.yaml).

## Minimal Example

```csv
variant_id,gene_id,label,feat_a,feat_b
v1,G1,1,0.1,1.0
v2,G1,0,0.2,1.1
v3,G2,1,0.3,0.9
v4,G2,0,0.4,1.2
```

## Optional Columns

- `domain_id`
- `protein_family`

These improve grouped error analysis and explainability hotspots.

## Split and Leakage Rules

- If `gene_id` exists, grouped split is used by default.
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
  label_column: label
  gene_column: gene_id
  required_features:
    - feat_a
    - feat_b
```
