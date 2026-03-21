"""Train an XGBoost model with PathoLogic from a raw CSV dataset.

This script adapts common external column names to the default PathoLogic
training schema (`gene_id`, `label`, dynamic features) and then runs
train/predict/evaluate with `PathoLogic("xgboost")`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _colorize(text: str, color: str, *, bold: bool = False) -> str:
    from pathologic.utils.colorstr import colorstr

    return colorstr(text, color, bold=bold)


def _pick_first_existing(columns: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not find required column. Tried: " + ", ".join(candidates)
    )


def _encode_feature_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    if numeric_ratio >= 0.8:
        return numeric.astype("float64")

    categorical = series.astype(str).fillna("__missing__")
    codes = pd.Categorical(categorical).codes
    return pd.Series(codes, index=series.index, dtype="float64")


def prepare_dataset_for_pathologic(
    input_csv: str,
    output_csv: str,
) -> tuple[str, list[str], dict[str, int]]:
    df = pd.read_csv(input_csv)
    cols = [str(c) for c in df.columns]
    input_rows = len(df)

    gene_col = _pick_first_existing(cols, ["gene_id", "Gene(s)"])
    label_col = _pick_first_existing(cols, ["label", "Target"])

    raw_feature_columns = [
        column
        for column in cols
        if column not in {gene_col, label_col}
    ]
    if not raw_feature_columns:
        raise ValueError("No feature columns found after removing gene/label columns.")

    encoded_features: dict[str, pd.Series] = {}
    for column in raw_feature_columns:
        encoded_name = f"feature__{column.strip().replace(' ', '_')}"
        encoded_features[encoded_name] = _encode_feature_column(df[column])

    prepared = pd.DataFrame({"gene_id": df[gene_col].astype(str)})
    prepared["label"] = pd.to_numeric(df[label_col], errors="coerce")
    for encoded_name, encoded_values in encoded_features.items():
        prepared[encoded_name] = encoded_values

    # Only gene/label are mandatory at this stage; feature NaNs are handled by preprocess.
    prepared = prepared.dropna(subset=["gene_id", "label"])
    prepared["label"] = prepared["label"].astype(int)
    output_rows = len(prepared)
    feature_columns = [column for column in prepared.columns if column.startswith("feature__")]

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out_path, index=False)
    stats = {
        "input_rows": int(input_rows),
        "prepared_rows": int(output_rows),
        "dropped_rows": int(input_rows - output_rows),
    }
    return str(out_path), feature_columns, stats


def train_xgboost(data_csv: str, feature_columns: list[str]) -> dict[str, Any]:
    from pathologic import PathoLogic

    model = PathoLogic("xgboost")
    model.defaults.setdefault("data", {})["required_features"] = list(feature_columns)
    model.train(data_csv)
    predictions = model.predict(data_csv)
    report = model.evaluate(data_csv)

    return {
        "model_name": "xgboost",
        "train_source": data_csv,
        "prediction_count": len(predictions),
        "first_prediction": predictions[0] if predictions else None,
        "metrics": report.get("metrics", {}),
        "feature_count": len(feature_columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt a CSV schema and train PathoLogic XGBoost model"
    )
    parser.add_argument("data_csv", help="Input CSV path (e.g. data/raw/data.csv)")
    parser.add_argument(
        "--prepared-output",
        default="data/processed/prepared_data_for_xgboost.csv",
        help="Where to write adapted training CSV",
    )
    parser.add_argument(
        "--delete-prepared",
        action="store_true",
        help="Delete prepared CSV after training completes",
    )
    args = parser.parse_args()

    print(_colorize("[1/3] Preparing dataset for PathoLogic schema...", "cyan", bold=True))
    prepared_path, feature_columns, prep_stats = prepare_dataset_for_pathologic(
        args.data_csv,
        args.prepared_output,
    )
    print(
        _colorize(
            "Prepared dataset written: "
            + prepared_path
            + " | input_rows="
            + str(prep_stats["input_rows"])
            + " prepared_rows="
            + str(prep_stats["prepared_rows"])
            + " dropped_rows="
            + str(prep_stats["dropped_rows"])
            + " feature_count="
            + str(len(feature_columns)),
            "green",
        )
    )

    print(_colorize("[2/3] Training + evaluating XGBoost model...", "cyan", bold=True))
    payload = train_xgboost(prepared_path, feature_columns)
    payload["prepared_data_path"] = prepared_path
    payload["preprocess_stats"] = prep_stats

    if args.delete_prepared:
        Path(prepared_path).unlink(missing_ok=True)
        payload["prepared_data_path"] = "deleted"

    print(_colorize("[3/3] Done.", "green", bold=True))
    print(_colorize(json.dumps(payload, ensure_ascii=True), "magenta"))


if __name__ == "__main__":
    main()
