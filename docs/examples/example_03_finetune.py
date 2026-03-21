"""Phase 8 example: fine-tune workflow with metric deltas."""

from __future__ import annotations

import argparse
import json
from typing import Any

from pathologic import PathoLogic


def run_finetune_workflow(data_path: str) -> dict[str, Any]:
    model = PathoLogic("mlp")
    model.train(data_path)

    report = model.fine_tune(
        data_path,
        freeze_layers="backbone_last2",
        learning_rate=0.0005,
        epochs=5,
    )

    return {
        "model_name": report["model_name"],
        "freeze_layers": report["freeze_layers"],
        "metric_delta": report["metric_delta"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fine-tune example.")
    parser.add_argument("data_path", help="Path to CSV/Parquet dataset")
    args = parser.parse_args()

    result = run_finetune_workflow(args.data_path)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
