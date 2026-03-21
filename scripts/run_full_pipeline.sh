#!/usr/bin/env bash
set -e

# ==============================================================================
# PathoLogic - Full Model Search & Reporting Pipeline
# 
# Usage: 
#   bash scripts/run_full_pipeline.sh [BUDGET_PROFILE] [MODELS]
#
# Examples:
#   bash scripts/run_full_pipeline.sh                 -> Runs with "aggressive" (default) and all models
#   bash scripts/run_full_pipeline.sh quick           -> Runs with "quick" budget profile
#   bash scripts/run_full_pipeline.sh balanced "rf"   -> Runs with "balanced" budget and ONLY Random Forest
# ==============================================================================

# 1. PARAMETER HANDLING
# Get the budget profile from the first argument. If not provided, default to "aggressive".
BUDGET=${1:-aggressive}
# Get the model list from the second argument. If not provided, default to all supported models.
MODELS=${2:-"rf,xgboost,lightgbm,catboost,tabnet,hybrid"}

echo "=========================================================="
echo "🚀 Pathologic - Model Search & Reporting Pipeline"
echo "=========================================================="
echo "🎯 Budget Profile  : $BUDGET"
echo "🧩 Selected Models : $MODELS"
echo "=========================================================="

# 2. RUN MODEL SEARCH
# Execute the main search script with the provided parameters. 
# Output is dynamically saved into a folder named after the budget profile.
echo "[1/4] Starting $BUDGET model search..."
echo "This may take a while depending on your hardware."
uv run python scripts/search_best_model.py data/processed/variants_for_examples.csv \
    --budget-profile "$BUDGET" \
    --models "$MODELS" \
    --output-dir "results/${BUDGET}_pipeline"

echo ""
# 3. FIND LATEST RUN
echo "[2/4] Finding the latest search results..."
# Dynamically find the most recently created search directory for the chosen budget.
# This prevents hardcoding timestamps or paths.
LATEST_DIR=$(ls -td results/${BUDGET}_pipeline/search_* | head -1)
if [ -z "$LATEST_DIR" ]; then
    echo "Error: No search directory found. Search may have failed."
    exit 1
fi
echo "Latest run found at: $LATEST_DIR"

echo ""
# 4. EXPORT TO CSV
echo "[3/4] Exporting leaderboard to CSV..."
# Run the python script to parse the nested JSON leaderboard into a flat CSV format.
LEADERBOARD_JSON="$LATEST_DIR/leaderboard.json"
uv run python scripts/export_leaderboard_csv.py "$LEADERBOARD_JSON"
CSV_PATH="$LATEST_DIR/leaderboard_summary.csv"
echo "CSV created at: $CSV_PATH"

echo ""
# 5. RENDER DYNAMIC REPORT (QUARTO)
echo "[4/4] Generating Quarto HTML Report..."
# Note: Quarto commands execute from the directory of the .qmd file (which is in `reports/`)
# Therefore, we must convert the absolute/cwd-based path to a relative path for the renderer.
RELATIVE_CSV_PATH="../$CSV_PATH"

# First, try system quarto if available
if command -v quarto &> /dev/null; then
    QUARTO_BIN="quarto"
elif [ -f "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto" ]; then
    QUARTO_BIN="/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto"
elif [ -f "/Applications/Positron.app/Contents/Resources/app/quarto/bin/quarto" ]; then
    QUARTO_BIN="/Applications/Positron.app/Contents/Resources/app/quarto/bin/quarto"
else
    QUARTO_BIN="quarto"
fi

# Execute Quarto with the found binary
export QUARTO_PYTHON=$(which python)
$QUARTO_BIN render reports/model_performance_report.qmd -P csv_path:"$RELATIVE_CSV_PATH"

echo ""
echo "=========================================================="
echo "✅ Pipeline Completed Successfully!"
echo "📊 Your dynamic report is ready at: reports/model_performance_report.html"
echo "=========================================================="
