# 🧬 PathoLogic

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-magenta)](https://github.com/astral-sh/uv)
[![Quarto](https://img.shields.io/badge/Reporting-Quarto-blueviolet)](https://quarto.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PathoLogic** is an automated, high-performance Machine Learning framework designed for predicting genomic variant pathogenicity. It integrates diverse model architectures, automated Neural Architecture Search (NAS) / Hyperparameter Optimization (HPO), and dynamic performance reporting.

Read the documentation in your preferred language:
- [📖 English Documentation](docs/en/quickstart.md)
- [🇹🇷 Türkçe Dokümantasyon](docs/tr/hizli-baslangic.md)

---

## 🎯 Purpose and Scope
The interpretation of genomic variants is a critical bottleneck in precision medicine. **PathoLogic** aims to streamline this by providing an end-to-end pipeline that:
1. Intake variant datasets.
2. Trains and optimizes a diverse zoo of models (GBDTs, TabNet, Hybrid approaches).
3. Evaluates model performance using rigorous metrics.
4. Generates dynamic, statistically sound HTML reports for researchers.

## 🧠 Methods and Architectures
PathoLogic encapsulates state-of-the-art tabular learning models and orchestrates their tuning:
- **Gradient Boosted Decision Trees (GBDT):** XGBoost, LightGBM, and CatBoost.
- **Deep Learning:** PyTorch-based TabNet implementation.
- **Hybrid Ensembles:** Automated stacked/voting ensembles for robust predictions.
- **Auto-HPO (NAS):** Powered by **Optuna**, the framework dynamically allocates computational budgets to find the theoretical limits of your models.

## 🚀 Hardware & Operating System Support
PathoLogic is strictly engineered for high performance and scales seamlessly across different platforms.

### 🍎 macOS (Apple Silicon - M1/M2/M3/M4)
- Native optimization for **Unified Memory architecture**.
- Data downcasting (Float64 $\rightarrow$ Float32) to double RAM efficiency and maximize CPU L-cache hits.
- Graceful fallbacks for unsupported PyTorch MPS (`device="cpu"`) combined with thread-count saturation (100% P-Core utilization) for models like LightGBM and CatBoost.
- XGBoost leverages Apple `hist` tree methods seamlessly.

### 🐧 Linux & 🪟 Windows (NVIDIA CUDA)
- Full support for hardware-accelerated training using CUDA (`device="cuda"`).
- Ideal for High-Performance Computing (HPC) environments scaling across multiple GPUs.
- Windows users should prefer **WSL2** for maximum networking and execution stability.

---

## ⚙️ Setup and Reproducibility

We use [`uv`](https://github.com/astral-sh/uv) to guarantee 100% deterministic package resolution and reproducibility across all operating systems.

### 1. Prerequisites
- [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [Install `Quarto`](https://quarto.org/docs/get-started/) (for automated HTML reporting)

### 2. Installation
Clone the repository and sync the environment:
```bash
git clone https://github.com/alierenaltindag/pathologic.git
cd pathologic
uv sync 
```

### 3. Running the E2E Pipeline
You can trigger the entire workflow (Training $\rightarrow$ NAS $\rightarrow$ CSV Export $\rightarrow$ Quarto Report) with a single command:

```bash
# Syntax: bash scripts/run_full_pipeline.sh <BUDGET> <MODELS>
bash scripts/run_full_pipeline.sh aggressive "rf,xgboost,lightgbm,catboost,tabnet,hybrid"
```
*Available budget profiles:* `quick`, `standard`, `aggressive`.

### 📊 Dynamic Reporting
Once the pipeline finishes, PathoLogic automatically processes the nested JSON trial logs, extracts the best metrics (F1, AUC, Recall, etc.), and renders an interactive **Quarto HTML dashboard** complete with Plotly radar charts and scatter plots. The output will be inside the `results/` directory.

---