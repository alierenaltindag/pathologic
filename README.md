# PathoLogic

PathoLogic is a robust, high-level machine learning framework designed for tabular data, specifically optimized for biological and pathological datasets. It provides a unified interface for state-of-the-art Gradient Boosting Machines (GBMs), Deep Learning models, and automated optimization strategies.

## 🚀 Key Features

* **Unified API:** Single entry point for training, tuning, and explaining complex models via the `PathoLogic` core.
* **Model Zoo:** Native support for `TabNet`, `LightGBM`, `CatBoost`, `XGBoost`, and specialized `MLP` architectures.
* **Meta Modeling:** Create powerful ensembles or stacking architectures by combining multiple base models.
* **AutoML Capabilities:** Integrated Hyperparameter Optimization (HPO) and Neural Architecture Search (NAS).
* **Deep Interpretability:** Built-in `SHAP` integration, biological mapping, and automated error analysis.
* **Robust Evaluation:** Advanced cross-validation strategies and distribution diagnostics.

## 🛠️ High-Level APIs

The core of the library is the `PathoLogic` class, which orchestrates the entire workflow.

### 1. Training and Fine-tuning

The `Trainer` engine supports early stopping, checkpointing, and hardware acceleration (CPU/GPU).

```python
from pathologic import PathoLogic

# Initialize for a specific model family (e.g., 'tabnet', 'mlp', 'lightgbm')
pl = PathoLogic("tabnet")

# Standard training from CSV
pl.train("data.csv", label_column="label")

# Fine-tune on a new domain-specific dataset
report = pl.fine_tune("new_data.csv", epochs=10)
```

### 2. Hyperparameter Tuning & NAS

`PathoLogic` leverages `Optuna-backed` strategies to search for optimal neural architectures (NAS) and hyperparameters simultaneously.

```python
# Automatically run HPO (and NAS for neural models like TabNet/MLP)
result = pl.tune("data.csv")
print(result["best_params"])
```

### 3. Evaluation

Built-in support for Stratified K-Fold and Group K-Fold evaluation to ensure model stability across different data splits.

```python
# Evaluate on a holdout set or perform cross-validation
results = pl.evaluate("test_data.csv")
print(results["metrics"])
```

### 4. Explainability (XAI)

The `Explain` engine provides global and local feature importance using `SHAP`, alongside specialized biological mappers for `interpretability` in clinical contexts.

```python
# Generate comprehensive SHAP report
report = pl.explain("test_data.csv")
```

### 5. Error & False Positive Analysis

Advanced diagnostics to understand model failure modes, including distribution drift detection between errors and correct predictions.

```python
# FP and Error analysis are integrated into the explainability report
report = pl.explain("test_data.csv", false_positive={"enabled": True})
```

## 🧠 Meta Models (Hybrid Architectures)

PathoLogic allows you to create **Meta Models** by combining multiple heterogeneous architectures. You can build ensembles that leverage the strengths of both Gradient Boosting and Deep Learning simultaneously.

```python
# Create a hybrid ensemble via the fluent builder API
builder = PathoLogic.builder() \
    .add_model("tabnet") \
    .add_model("lightgbm") \
    .strategy("stacking")

pl = PathoLogic.from_builder(builder)
pl.train("data.csv")
```

## 🦁 Model Pool

PathoLogic integrates a diverse set of models optimized for tabular performance:

| Category | Models |
| :--- | :--- |
| **Deep Learning** | `TabNet`, `MLP` (Multi-Layer Perceptron) |
| **GBMs** | `LightGBM`, `CatBoost`, `XGBoost`, `HistGBDT` |
| **Traditional** | `RandomForest`, `LogisticRegression`, `SVM` |
| **Meta** | `HybridModel` (Ensemble/Stacking support for multiple models) |

## 📂 Project Structure

* `pathologic/core.py`: The main high-level API.
* `pathologic/engine/`: Core logic for `Trainer`, `Tuner`, and `Evaluator`.
* `pathologic/nas/`: Strategies and search logic for Neural Architecture Search.
* `pathologic/models/hybrid.py`: Implementation of Meta/Hybrid model orchestration.
* `pathologic/models/zoo/`: Implementation wrappers for all supported models.
* `pathologic/explain/`: `SHAP` engines, Error Analysis, and Biological Mapping.
* `pathologic/configs/`: `YAML-based` configuration management (Hydra compatible).

## 🍲 Installation

```bash
git clone https://github.com/alierenaltindag/pathologic.git
cd pathologic
pip install -e .
```

## 📖 Documentation

For detailed guides, check the `docs/` folder:

* [Quickstart Guide](docs/en/quickstart.md)
* [Training & Tuning](docs/en/training-and-tuning.md)
