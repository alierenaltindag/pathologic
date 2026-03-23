"""Dataset preparation helpers for search workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from pathologic import PathoLogic


def pick_first_existing(columns: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError("Could not find required column. Tried: " + ", ".join(candidates))


def encode_feature_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    if numeric_ratio >= 0.8:
        return numeric.astype("float64")

    categorical = series.astype(str).fillna("__missing__")
    codes = pd.Categorical(categorical).codes
    return pd.Series(codes, index=series.index, dtype="float64")


def normalize_column_token(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column_name.lower())


def is_identifier_like_column(column_name: str) -> bool:
    token = normalize_column_token(column_name)
    if token in {"geneid", "label", "target"}:
        return False
    if "context" in token:
        return True
    return any(
        marker in token
        for marker in (
            "variationid",
            "variantid",
            "sampleid",
            "recordid",
            "patientid",
            "subjectid",
            "rowid",
            "index",
        )
    )


def select_feature_columns(
    columns: list[str],
    *,
    gene_column: str,
    label_column: str,
    excluded_columns: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    dropped_excluded: list[str] = []
    excluded_set = set(excluded_columns or set())
    for column in columns:
        if column in {gene_column, label_column}:
            continue
        if column in excluded_set:
            dropped_excluded.append(column)
            continue
        if is_identifier_like_column(column):
            dropped.append(column)
            continue
        kept.append(column)
    return kept, dropped, dropped_excluded


def resolve_excluded_columns_from_defaults() -> list[str]:
    probe = PathoLogic("logreg")
    data_config_raw = probe.defaults.get("data")
    data_config = data_config_raw if isinstance(data_config_raw, dict) else {}
    excluded_raw = data_config.get("excluded_columns", [])
    if excluded_raw is None:
        return []
    if not isinstance(excluded_raw, list):
        raise ValueError("Config field 'data.excluded_columns' must be a list.")
    return [str(column) for column in excluded_raw if str(column)]


def resolve_error_analysis_columns_from_defaults() -> list[str]:
    probe = PathoLogic("logreg")
    data_config_raw = probe.defaults.get("data")
    data_config = data_config_raw if isinstance(data_config_raw, dict) else {}
    raw_columns = data_config.get("error_analysis_columns", [])
    if raw_columns is None:
        return []
    if not isinstance(raw_columns, list):
        raise ValueError("Config field 'data.error_analysis_columns' must be a list.")
    return [str(column) for column in raw_columns if str(column)]


def resolve_search_defaults_from_defaults() -> dict[str, Any]:
    probe = PathoLogic("logreg")
    search_config_raw = probe.defaults.get("search")
    search_config = search_config_raw if isinstance(search_config_raw, dict) else {}
    return dict(search_config)


def prepare_dataset_for_pathologic(
    input_csv: str,
    output_csv: str,
    *,
    excluded_columns: list[str] | None = None,
    error_analysis_columns: list[str] | None = None,
) -> tuple[str, list[str], dict[str, Any]]:
    """Adapt external schema to PathoLogic-ready gene/label/feature columns."""
    df = pd.read_csv(input_csv)
    cols = [str(c) for c in df.columns]
    input_rows = len(df)

    gene_col = pick_first_existing(cols, ["gene_id", "Gene(s)"])
    label_col = pick_first_existing(cols, ["label", "Target"])

    passthrough_candidates = [
        str(column) for column in (error_analysis_columns or []) if str(column)
    ]
    passthrough_columns = [
        column for column in passthrough_candidates if column in cols and column != label_col
    ]
    passthrough_columns = list(dict.fromkeys(passthrough_columns))

    excluded_set = {str(column) for column in (excluded_columns or []) if str(column)}
    excluded_set.update(passthrough_columns)

    raw_feature_columns, dropped_identifier_columns, dropped_excluded_columns = select_feature_columns(
        cols,
        gene_column=gene_col,
        label_column=label_col,
        excluded_columns=excluded_set,
    )
    if not raw_feature_columns:
        raise ValueError("No feature columns found after removing gene/label columns.")

    encoded_features: dict[str, pd.Series] = {}
    for column in raw_feature_columns:
        encoded_name = f"feature__{column.strip().replace(' ', '_')}"
        encoded_features[encoded_name] = encode_feature_column(df[column])

    prepared = pd.DataFrame({"gene_id": df[gene_col].astype(str)})
    prepared["label"] = pd.to_numeric(df[label_col], errors="coerce")
    for encoded_name, encoded_values in encoded_features.items():
        prepared[encoded_name] = encoded_values
    for column in passthrough_columns:
        prepared[column] = df[column]

    prepared = prepared.dropna(subset=["gene_id", "label"])
    prepared["label"] = prepared["label"].astype(int)
    feature_columns = [column for column in prepared.columns if column.startswith("feature__")]

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out_path, index=False)

    stats = {
        "input_rows": int(input_rows),
        "prepared_rows": int(len(prepared)),
        "dropped_rows": int(input_rows - len(prepared)),
        "dropped_identifier_columns": dropped_identifier_columns,
        "dropped_identifier_column_count": int(len(dropped_identifier_columns)),
        "dropped_excluded_columns": dropped_excluded_columns,
        "dropped_excluded_column_count": int(len(dropped_excluded_columns)),
        "retained_error_analysis_columns": passthrough_columns,
        "retained_error_analysis_column_count": int(len(passthrough_columns)),
    }
    return str(out_path), feature_columns, stats
