# PathoLogic Explainability and Error Analysis Guide

This guide documents the Phase 6 explainability workflow and how to use it in practice.

## Goal

- Explain model predictions with feature contributions
- Generate biologically interpretable narratives
- Report false-positive hotspots by groups
- Keep outputs auditable with explicit metadata

## Flow Summary

1. Train with `PathoLogic.train(data)`.
2. Call `PathoLogic.explain(data)`.
3. The system returns:
   - `global_feature_importance`
   - `sample_explanations`
   - `false_positive_hotspots`
   - `metadata`
   - `visual_report_html`

## Configuration

Explainability runtime settings are in [pathologic/configs/runtime/explain.yaml](pathologic/configs/runtime/explain.yaml).

Key fields:

- `backend`: `auto`, `tree`, `linear`, `deep`, `shap`, `proxy`
- `background_size`: SHAP background sample size
- `top_k_features`: top features to report
- `top_k_samples`: sample-level explanations to include
- `group_columns`: candidate columns for hotspot analysis
- `biological_mapping`: technical feature -> biological label mapping
- `false_positive.*`: hotspot filtering options
- `visual_report.*`: HTML report options

## Backend Routing Policy

With `backend: auto`, the system routes by model type:

- tree-like models -> `tree_shap`
- linear models -> `linear_shap`
- torch models -> `deep_shap`
- unsupported/failing path -> `proxy`

Implementation reference: [pathologic/explain/shap_engine.py](pathologic/explain/shap_engine.py).

## Leakage and Determinism

- Background data defaults to the training split (`background_source: train_split`).
- Explain computation runs in a deterministic context (seed, numpy, torch, cuDNN settings).
- This improves reproducibility for audits and regression checks.

## API Example

```python
from pathologic import PathoLogic

model = PathoLogic("logreg")
model.train("data/variants.csv")
report = model.explain("data/variants.csv")

print(report["backend"])
print(report["metadata"]["background_source"])
print(report["global_feature_importance"][:3])
```

## Persisting the HTML Report

In [pathologic/configs/runtime/explain.yaml](pathologic/configs/runtime/explain.yaml):

```yaml
visual_report:
  enabled: true
  save_path: reports/explain/latest_report.html
```

With this setting, `model.explain(...)` writes an HTML report to disk.

## Output Contract

The report schema is defined in [pathologic/explain/schemas.py](pathologic/explain/schemas.py).

Expected top-level keys:

- `backend`
- `global_feature_importance`
- `sample_explanations`
- `false_positive_hotspots`
- `metadata`
- `visual_report_html` (when visual reports are enabled)

## Test Coverage

- Unit:
  - [tests/unit/test_explainability_unit.py](tests/unit/test_explainability_unit.py)
  - [tests/unit/test_shap_engine_unit.py](tests/unit/test_shap_engine_unit.py)
- Integration:
  - [tests/integration/test_phase6_explainability_integration.py](tests/integration/test_phase6_explainability_integration.py)

## Operational Recommendations

- Enable `save_path` for clinical/audit workflows.
- Fill `group_columns` with your domain-relevant biological groups.
- Maintain `biological_mapping` based on your data dictionary.
