"""Phase 8 example: basic train, predict, evaluate workflow."""

from __future__ import annotations

import argparse
import json
from typing import Any

from pathologic import PathoLogic


def run_basic_workflow(data_path: str) -> dict[str, Any]:
    model = PathoLogic("logreg")
    model.train(data_path)
    predictions = model.predict(data_path)
    report = model.evaluate(data_path)

    return {
        "model_name": "logreg",
        "prediction_count": len(predictions),
        "first_prediction": predictions[0] if predictions else None,
        "metrics": report.get("metrics", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic PathoLogic workflow example.")
    parser.add_argument("data_path", help="Path to CSV/Parquet dataset")
    args = parser.parse_args()

    result = run_basic_workflow(args.data_path)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
