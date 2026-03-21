"""Train-fold-only preprocessing utilities with artifact persistence."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ScalerName = Literal["standard", "minmax"]
ImputeName = Literal["none", "mean", "median", "most_frequent"]
MissingValuePolicy = Literal["impute", "drop_rows"]


@dataclass
class _GeneStats:
    first: np.ndarray
    second: np.ndarray


class FoldPreprocessor:
    """Fold-aware preprocessing that fits only on train data."""

    def __init__(
        self,
        *,
        numeric_features: list[str],
        gene_column: str = "gene_id",
        missing_value_policy: MissingValuePolicy = "impute",
        impute_strategy: ImputeName = "median",
        scaler: ScalerName = "standard",
        per_gene: bool = False,
        per_gene_features: list[str] | None = None,
        scaler_features: list[str] | None = None,
        add_missing_indicators: bool = False,
        missing_indicator_features: list[str] | None = None,
    ) -> None:
        if not numeric_features:
            raise ValueError("numeric_features cannot be empty.")
        self.numeric_features = numeric_features
        self.gene_column = gene_column
        self.missing_value_policy = missing_value_policy
        self.impute_strategy = impute_strategy
        self.scaler_name = scaler
        self.per_gene = per_gene
        self.per_gene_features = per_gene_features
        self.scaler_features = scaler_features
        self.add_missing_indicators = add_missing_indicators
        self.missing_indicator_features = missing_indicator_features

        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self._per_gene_stats: dict[str, _GeneStats] = {}
        self._global_stats: _GeneStats | None = None
        self._per_gene_feature_names: list[str] = []
        self._global_scaler_features: list[str] = []
        self._scaled_features: list[str] = []
        self._missing_indicator_source_features: list[str] = []
        self._missing_indicator_features: list[str] = []
        self.fitted = False

        self._validate_feature_subsets()
        self._resolve_feature_scopes()
        self._validate_missing_value_policy()

    def fit(self, train_df: pd.DataFrame) -> FoldPreprocessor:
        fit_df = self._prepare_input_frame(train_df, require_non_empty=True)
        self._fit_missing_indicators(fit_df)

        if self.impute_strategy == "none":
            self.imputer = None
            train_imputed = fit_df[self.numeric_features].copy()
        else:
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
            train_values = self.imputer.fit_transform(fit_df[self.numeric_features])
            train_imputed = pd.DataFrame(
                train_values,
                columns=self.numeric_features,
                index=fit_df.index,
            )

        self._validate_missing_for_scaling(train_imputed)

        if self.per_gene:
            self._fit_per_gene(fit_df, train_imputed)
            if self._global_scaler_features:
                if self.scaler_name == "standard":
                    self.scaler = StandardScaler()
                else:
                    self.scaler = MinMaxScaler()
                self.scaler.fit(train_imputed[self._global_scaler_features])
            else:
                self.scaler = None
        else:
            if self._scaled_features:
                if self.scaler_name == "standard":
                    self.scaler = StandardScaler()
                else:
                    self.scaler = MinMaxScaler()
                self.scaler.fit(train_imputed[self._scaled_features])
            else:
                self.scaler = None

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Call fit(...) before transform(...).")
        transform_df = self._prepare_input_frame(df, require_non_empty=False)

        if self.impute_strategy == "none":
            imputed = transform_df[self.numeric_features].copy()
        else:
            if self.imputer is None:
                raise RuntimeError("Imputer is not fitted.")
            values = self.imputer.transform(transform_df[self.numeric_features])
            imputed = pd.DataFrame(
                values,
                columns=self.numeric_features,
                index=transform_df.index,
            )

        if self.per_gene:
            scaled = imputed.copy()
            if self._per_gene_feature_names:
                per_gene_scaled = self._transform_per_gene(
                    transform_df,
                    imputed[self._per_gene_feature_names],
                )
                scaled.loc[:, self._per_gene_feature_names] = per_gene_scaled
            if self._global_scaler_features:
                if self.scaler is None:
                    raise RuntimeError("Scaler is not fitted.")
                global_scaled = self.scaler.transform(imputed[self._global_scaler_features])
                scaled.loc[:, self._global_scaler_features] = global_scaled
        else:
            scaled = imputed.copy()
            if self._scaled_features:
                if self.scaler is None:
                    raise RuntimeError("Scaler is not fitted.")
                subset_scaled = self.scaler.transform(imputed[self._scaled_features])
                scaled.loc[:, self._scaled_features] = subset_scaled

        transformed = transform_df.copy()
        transformed.loc[:, self.numeric_features] = scaled[self.numeric_features]
        if self._missing_indicator_features:
            indicator_frame = self._build_missing_indicator_frame(transform_df)
            for feature in self._missing_indicator_features:
                transformed.loc[:, feature] = indicator_frame[feature]
        return transformed

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def save_artifacts(self, path: str) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save artifacts before fitting preprocessor.")

        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "numeric_features": self.numeric_features,
            "gene_column": self.gene_column,
            "missing_value_policy": self.missing_value_policy,
            "impute_strategy": self.impute_strategy,
            "scaler_name": self.scaler_name,
            "per_gene": self.per_gene,
            "per_gene_features": self.per_gene_features,
            "scaler_features": self.scaler_features,
            "add_missing_indicators": self.add_missing_indicators,
            "missing_indicator_features": self.missing_indicator_features,
            "resolved_missing_indicator_source_features": self._missing_indicator_source_features,
            "resolved_missing_indicator_features": self._missing_indicator_features,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "per_gene_stats": self._per_gene_stats,
            "global_stats": self._global_stats,
            "fitted": self.fitted,
        }
        with artifact_path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load_artifacts(cls, path: str) -> FoldPreprocessor:
        artifact_path = Path(path)
        with artifact_path.open("rb") as handle:
            payload = pickle.load(handle)

        obj = cls(
            numeric_features=payload["numeric_features"],
            gene_column=payload["gene_column"],
            missing_value_policy=payload.get("missing_value_policy", "impute"),
            impute_strategy=payload["impute_strategy"],
            scaler=payload["scaler_name"],
            per_gene=payload["per_gene"],
            per_gene_features=payload.get("per_gene_features"),
            scaler_features=payload.get("scaler_features"),
            add_missing_indicators=bool(payload.get("add_missing_indicators", False)),
            missing_indicator_features=payload.get("missing_indicator_features"),
        )
        obj.imputer = payload["imputer"]
        obj.scaler = payload["scaler"]
        obj._per_gene_stats = payload["per_gene_stats"]
        obj._global_stats = payload["global_stats"]
        obj._missing_indicator_source_features = list(
            payload.get("resolved_missing_indicator_source_features", [])
        )
        obj._missing_indicator_features = list(
            payload.get("resolved_missing_indicator_features", [])
        )
        obj.fitted = payload["fitted"]
        return obj

    def _fit_per_gene(self, train_df: pd.DataFrame, train_imputed: pd.DataFrame) -> None:
        if self.gene_column not in train_df.columns:
            raise ValueError(f"Missing gene column: {self.gene_column}")

        if not self._per_gene_feature_names:
            self._global_stats = None
            return

        first, second = self._stat_arrays(train_imputed[self._per_gene_feature_names])
        self._global_stats = _GeneStats(first=first, second=second)

        for gene, index in train_df.groupby(self.gene_column).groups.items():
            group_matrix = train_imputed.loc[index, self._per_gene_feature_names]
            g_first, g_second = self._stat_arrays(group_matrix)
            self._per_gene_stats[str(gene)] = _GeneStats(first=g_first, second=g_second)

    def _transform_per_gene(self, df: pd.DataFrame, per_gene_imputed: pd.DataFrame) -> np.ndarray:
        if self._global_stats is None:
            raise RuntimeError("Global per-gene statistics are missing.")

        result = np.empty_like(per_gene_imputed.to_numpy(dtype=float), dtype=float)
        genes = df[self.gene_column].astype(str)
        raw_values = per_gene_imputed.to_numpy(dtype=float)

        for row_index, gene in enumerate(genes):
            stats = self._per_gene_stats.get(gene, self._global_stats)
            result[row_index] = (raw_values[row_index] - stats.first) / stats.second
        return result

    def _stat_arrays(self, matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = matrix.to_numpy(dtype=float)
        if self.scaler_name == "standard":
            means = values.mean(axis=0)
            stds = values.std(axis=0)
            stds[stds == 0.0] = 1.0
            return means, stds

        mins = values.min(axis=0)
        ranges = values.max(axis=0) - mins
        ranges[ranges == 0.0] = 1.0
        return mins, ranges

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [column for column in self.numeric_features if column not in df.columns]
        if missing:
            raise ValueError(f"Missing numeric feature columns: {', '.join(missing)}")
        if self.per_gene and self.gene_column not in df.columns:
            raise ValueError(f"Missing gene column: {self.gene_column}")

    def _validate_feature_subsets(self) -> None:
        numeric_set = set(self.numeric_features)
        if self.per_gene_features is not None:
            invalid = [feature for feature in self.per_gene_features if feature not in numeric_set]
            if invalid:
                raise ValueError(
                    "per_gene_features must be a subset of numeric_features. "
                    f"Invalid: {', '.join(invalid)}"
                )
        if self.scaler_features is not None:
            invalid = [feature for feature in self.scaler_features if feature not in numeric_set]
            if invalid:
                raise ValueError(
                    "scaler_features must be a subset of numeric_features. "
                    f"Invalid: {', '.join(invalid)}"
                )
        if self.missing_indicator_features is not None:
            invalid = [
                feature
                for feature in self.missing_indicator_features
                if feature not in numeric_set
            ]
            if invalid:
                raise ValueError(
                    "missing_indicator_features must be a subset of numeric_features. "
                    f"Invalid: {', '.join(invalid)}"
                )

    def _resolve_feature_scopes(self) -> None:
        if self.per_gene:
            if self.per_gene_features is None:
                self._per_gene_feature_names = list(self.numeric_features)
            else:
                self._per_gene_feature_names = list(dict.fromkeys(self.per_gene_features))

            per_gene_set = set(self._per_gene_feature_names)
            self._global_scaler_features = [
                feature for feature in self.numeric_features if feature not in per_gene_set
            ]
            self._scaled_features = []
            return

        if self.scaler_features is None:
            self._scaled_features = list(self.numeric_features)
        else:
            self._scaled_features = list(dict.fromkeys(self.scaler_features))

        self._per_gene_feature_names = []
        self._global_scaler_features = []

    def _validate_missing_for_scaling(self, frame: pd.DataFrame) -> None:
        """Disallow scaler/per-gene normalization with NaN when imputation is disabled."""
        if self.impute_strategy != "none":
            return

        checked_features: list[str]
        if self.per_gene:
            checked_features = [
                *self._per_gene_feature_names,
                *self._global_scaler_features,
            ]
        else:
            checked_features = list(self._scaled_features)

        if not checked_features:
            return

        subset = frame[checked_features]
        if subset.isna().any().any():
            feature_list = ", ".join(dict.fromkeys(checked_features))
            raise ValueError(
                "Missing values detected while preprocess.impute_strategy='none'. "
                "Set impute_strategy to one of mean/median/most_frequent or disable "
                "scaling on features with missing values. "
                f"Affected features: {feature_list}"
            )

    def _fit_missing_indicators(self, train_df: pd.DataFrame) -> None:
        if not self.add_missing_indicators:
            self._missing_indicator_source_features = []
            self._missing_indicator_features = []
            return

        if self.missing_indicator_features is None:
            candidates = list(self.numeric_features)
        else:
            candidates = list(dict.fromkeys(self.missing_indicator_features))

        selected = [
            feature
            for feature in candidates
            if feature in train_df.columns and bool(train_df[feature].isna().any())
        ]
        self._missing_indicator_source_features = selected
        self._missing_indicator_features = [
            self._missing_indicator_column_name(feature) for feature in selected
        ]

    def _build_missing_indicator_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._missing_indicator_source_features:
            return pd.DataFrame(index=df.index)
        frame = pd.DataFrame(index=df.index)
        for feature in self._missing_indicator_source_features:
            frame[self._missing_indicator_column_name(feature)] = (
                df[feature].isna().astype("float64")
            )
        return frame

    def _missing_indicator_column_name(self, feature: str) -> str:
        return f"{feature}__is_missing"

    def _validate_missing_value_policy(self) -> None:
        if self.missing_value_policy not in {"impute", "drop_rows"}:
            raise ValueError(
                "missing_value_policy must be one of: impute, drop_rows"
            )

    def _prepare_input_frame(self, df: pd.DataFrame, *, require_non_empty: bool) -> pd.DataFrame:
        self._validate_columns(df)
        if self.missing_value_policy != "drop_rows":
            return df

        keep_mask = ~df[self.numeric_features].isna().any(axis=1)
        filtered = df.loc[keep_mask].copy()
        if require_non_empty and filtered.empty:
            raise ValueError(
                "All rows were dropped by preprocess.missing_value_policy='drop_rows'. "
                "Use imputation or provide data with fewer missing numeric values."
            )
        return filtered
