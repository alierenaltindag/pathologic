"""Phase 8 example: builder-based ensemble training workflow."""

from __future__ import annotations

import argparse
import json
from typing import Any

from pathologic import PathoLogic


def run_ensemble_workflow(data_path: str) -> dict[str, Any]:
    builder = (
        PathoLogic.builder()
        .add_model("logreg", c=1.0)
        .add_model("random_forest", n_estimators=100)
        .add_model("hist_gbdt", learning_rate=0.1)
        .strategy("stacking", cv=3)
        .meta_model("logreg", c=0.7)
    )

    model = PathoLogic.from_builder(builder)
    model.train(data_path)
    predictions = model.predict(data_path)

    return {
        "model_name": model.model_name,
        "prediction_count": len(predictions),
        "first_prediction": predictions[0] if predictions else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ensemble builder example.")
    parser.add_argument("data_path", help="Path to CSV/Parquet dataset")
    args = parser.parse_args()

    result = run_ensemble_workflow(args.data_path)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
