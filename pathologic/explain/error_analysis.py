"""Multi-dimensional error analysis utilities for search workflows.

This module intentionally avoids external biological databases and excludes
positional AA features from all analyses.
"""

from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


@dataclass(frozen=True)
class ErrorAnalysisResult:
    """Serializable summary payload for candidate-level error analysis."""

    status: str
    summary: dict[str, Any]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": dict(self.summary),
            "artifacts": dict(self.artifacts),
        }


class MultiDimensionalErrorAnalyzer:
    """Analyze model errors in feature-space without spatial annotations."""

    _DISALLOWED_TOKENS = {
        "aaposition",
        "aa_position",
        "feature__aa_position",
    }

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = int(random_state)

    @classmethod
    def _is_disallowed_column(cls, column: str) -> bool:
        token = "".join(ch for ch in column.lower() if ch.isalnum() or ch == "_")
        return any(flag in token for flag in cls._DISALLOWED_TOKENS)

    @staticmethod
    def _resolve_feature_column(dataset: pd.DataFrame, base_name: str) -> str | None:
        candidates = [
            base_name,
            f"feature__{base_name}",
            f"feature__{base_name.replace(' ', '_')}",
        ]
        for candidate in candidates:
            if candidate in dataset.columns:
                return candidate
        return None

    def _resolve_numeric_columns(self, dataset: pd.DataFrame) -> list[str]:
        preferred = [
            "cadd.phred",
            "REVEL_Score",
            "dbnsfp.sift.score",
            "GERP_Score",
            "gnomAD_AF",
            "Grantham_Score",
            "Shannon_Entropy",
            "Local_Hydrophobicity",
            "BLOSUM_Score",
            "Charge_Change",
            "Polarity_Change",
            "Hyd_Delta",
            "MW_Delta",
            "Prolin_Cysteine_Count",
        ]
        resolved: list[str] = []
        for feature in preferred:
            column = self._resolve_feature_column(dataset, feature)
            if column is None:
                continue
            if self._is_disallowed_column(column):
                continue
            resolved.append(column)

        # Fall back to all numeric feature columns if preferred set is missing.
        if resolved:
            return list(dict.fromkeys(resolved))

        numeric_like = [
            column
            for column in dataset.columns
            if column.startswith("feature__") and not self._is_disallowed_column(column)
        ]
        return numeric_like

    @staticmethod
    def _resolve_gene_column(dataset: pd.DataFrame) -> str | None:
        for column in ("gene_id", "Gene(s)"):
            if column in dataset.columns:
                return column
        return None

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

    def _build_error_frame(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
        dataset: pd.DataFrame,
        numeric_columns: list[str],
    ) -> pd.DataFrame:
        error_mask = (y_true != y_pred)
        if int(error_mask.sum()) == 0:
            return pd.DataFrame()

        error_type = np.where((y_true == 0) & (y_pred == 1), "FP", "FN")
        frame = dataset.loc[error_mask].copy()
        frame["y_true"] = y_true[error_mask]
        frame["y_pred"] = y_pred[error_mask]
        frame["y_score"] = y_score[error_mask]
        frame["error_type"] = error_type[error_mask]

        existing_numeric = [column for column in numeric_columns if column in frame.columns]
        for column in existing_numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def _fit_surrogate_tree(
        self,
        *,
        error_frame: pd.DataFrame,
        numeric_columns: list[str],
    ) -> tuple[dict[str, Any], DecisionTreeClassifier | None, np.ndarray | None]:
        feature_columns = [column for column in numeric_columns if column in error_frame.columns]
        if len(feature_columns) < 2:
            return {"status": "skipped", "reason": "insufficient_numeric_features"}, None, None

        x_raw = error_frame[feature_columns]
        x_imputed = SimpleImputer(strategy="median").fit_transform(x_raw)
        # y mapping: 1 for FP (predicted Pathogenic when Benign), 0 for FN (predicted Benign when Pathogenic)
        y = (error_frame["error_type"] == "FP").to_numpy(dtype=int)
        unique_labels = np.unique(y)

        if unique_labels.size < 2:
            return {
                "status": "skipped",
                "reason": "single_error_type",
                "error_type": str(error_frame["error_type"].iloc[0]),
            }, None, None

        model = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            random_state=self.random_state,
            class_weight="balanced",
        )
        model.fit(x_imputed, y)

        rules = export_text(model, feature_names=feature_columns)
        importances = [
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in sorted(
                zip(feature_columns, model.feature_importances_, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
            if float(importance) > 0.0
        ]

        summary = {
            "status": "ok",
            "sample_count": int(len(error_frame)),
            "class_balance": {
                "fp": int((y == 1).sum()),
                "fn": int((y == 0).sum()),
            },
            "feature_importances": importances,
            "rules_text": rules,
        }
        return summary, model, x_imputed

    def _gene_proxy_analysis(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset: pd.DataFrame,
        numeric_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        gene_column = self._resolve_gene_column(dataset)
        if gene_column is None:
            return pd.DataFrame(), pd.DataFrame()

        work = dataset.copy()
        work["y_true"] = y_true
        work["y_pred"] = y_pred
        work["is_fp"] = ((y_true == 0) & (y_pred == 1)).astype(int)
        work["is_fn"] = ((y_true == 1) & (y_pred == 0)).astype(int)

        numeric_existing = [column for column in numeric_columns if column in work.columns]
        for column in numeric_existing:
            work[column] = pd.to_numeric(work[column], errors="coerce")

        grouped = work.groupby(gene_column, dropna=False)
        stats_rows: list[dict[str, Any]] = []
        for gene, frame in grouped:
            row: dict[str, Any] = {
                "gene": str(gene),
                "variant_count": int(len(frame)),
                "fpr": self._safe_divide(float(frame["is_fp"].sum()), float((frame["y_true"] == 0).sum())),
                "fnr": self._safe_divide(float(frame["is_fn"].sum()), float((frame["y_true"] == 1).sum())),
            }
            for column in numeric_existing:
                row[f"mean__{column}"] = float(frame[column].mean()) if len(frame) else float("nan")
                row[f"median__{column}"] = float(frame[column].median()) if len(frame) else float("nan")
            stats_rows.append(row)

        stats_df = pd.DataFrame(stats_rows)
        if stats_df.empty:
            return stats_df, pd.DataFrame()

        corr_rows: list[dict[str, Any]] = []
        metric_columns = [column for column in stats_df.columns if column != "gene"]
        for column in metric_columns:
            series = pd.to_numeric(stats_df[column], errors="coerce")
            if series.notna().sum() < 3:
                continue
            for target in ("fpr", "fnr"):
                target_series = pd.to_numeric(stats_df[target], errors="coerce")
                mask = series.notna() & target_series.notna()
                if int(mask.sum()) < 3:
                    continue
                if pd.Series(series[mask]).nunique(dropna=True) < 2:
                    continue
                if pd.Series(target_series[mask]).nunique(dropna=True) < 2:
                    continue
                coefficient, p_value = spearmanr(series[mask], target_series[mask])
                if np.isnan(coefficient):
                    continue
                corr_rows.append(
                    {
                        "metric": column,
                        "target": target,
                        "spearman_r": float(coefficient),
                        "p_value": float(p_value),
                        "n": int(mask.sum()),
                    }
                )

        corr_df = pd.DataFrame(corr_rows).sort_values(
            by="spearman_r", key=lambda s: s.abs(), ascending=False
        ) if corr_rows else pd.DataFrame()
        return stats_df, corr_df

    def _cluster_errors(
        self,
        *,
        error_frame: pd.DataFrame,
        numeric_columns: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        numeric_existing = [column for column in numeric_columns if column in error_frame.columns]
        if len(numeric_existing) < 2 or len(error_frame) < 8:
            return error_frame.copy(), {"status": "skipped", "reason": "insufficient_error_samples"}

        x_raw = error_frame[numeric_existing]
        x_imputed = SimpleImputer(strategy="median").fit_transform(x_raw)
        scaler = StandardScaler()
        x = scaler.fit_transform(x_imputed)

        cluster_frame = error_frame.copy()
        summary: dict[str, Any] = {"status": "ok", "numeric_feature_count": len(numeric_existing)}

        n_samples = len(cluster_frame)
        n_clusters = max(2, min(6, int(np.sqrt(n_samples))))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(x)
        cluster_frame["kmeans_cluster"] = kmeans_labels
        summary["kmeans"] = {
            "n_clusters": int(n_clusters),
            "silhouette": float(silhouette_score(x, kmeans_labels)) if n_clusters > 1 else 0.0,
        }

        min_samples = max(4, int(np.sqrt(n_samples) / 1.5))
        dbscan = DBSCAN(eps=1.1, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(x)
        cluster_frame["dbscan_cluster"] = dbscan_labels
        n_dbscan_clusters = int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))
        summary["dbscan"] = {
            "eps": 1.1,
            "min_samples": int(min_samples),
            "n_clusters": n_dbscan_clusters,
            "noise_count": int((dbscan_labels == -1).sum()),
        }

        tsne_perplexity = max(5, min(30, n_samples // 3))
        tsne = TSNE(
            n_components=2,
            random_state=self.random_state,
            perplexity=float(tsne_perplexity),
            init="pca",
            learning_rate="auto",
        )
        tsne_xy = tsne.fit_transform(x)
        cluster_frame["tsne_x"] = tsne_xy[:, 0]
        cluster_frame["tsne_y"] = tsne_xy[:, 1]

        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=self.random_state)
            umap_xy = reducer.fit_transform(x)
            cluster_frame["umap_x"] = umap_xy[:, 0]
            cluster_frame["umap_y"] = umap_xy[:, 1]
            summary["umap"] = {"status": "ok"}
        except Exception as exc:
            summary["umap"] = {"status": "failed", "reason": str(exc)}

        return cluster_frame, summary

    @staticmethod
    def _cluster_profiles(
        frame: pd.DataFrame,
        cluster_column: str,
        numeric_columns: list[str],
    ) -> list[dict[str, Any]]:
        if cluster_column not in frame.columns:
            return []

        profiles: list[dict[str, Any]] = []
        numeric_existing = [column for column in numeric_columns if column in frame.columns]
        for cluster_value, subset in frame.groupby(cluster_column, dropna=False):
            row: dict[str, Any] = {
                "cluster": str(cluster_value),
                "sample_count": int(len(subset)),
                "fp_ratio": float((subset["error_type"] == "FP").mean()),
                "fn_ratio": float((subset["error_type"] == "FN").mean()),
            }
            means: list[tuple[str, float]] = []
            for column in numeric_existing:
                mean_value = float(pd.to_numeric(subset[column], errors="coerce").mean())
                row[f"mean__{column}"] = mean_value
                if not np.isnan(mean_value):
                    means.append((column, mean_value))

            means_sorted = sorted(means, key=lambda item: abs(item[1]), reverse=True)
            top_items = means_sorted[:3]
            if top_items:
                top_text = ", ".join(f"{name}={value:.3f}" for name, value in top_items)
                narrative = (
                    f"cluster={cluster_value} n={len(subset)} "
                    f"FP%={(subset['error_type'] == 'FP').mean():.2f} top:{top_text}"
                )
            else:
                narrative = f"cluster={cluster_value} n={len(subset)}"
            row["profile_narrative"] = narrative
            profiles.append(row)

        profiles.sort(key=lambda item: int(item.get("sample_count", 0)), reverse=True)
        return profiles

    def _map_biochemical_patterns(self, error_frame: pd.DataFrame) -> pd.DataFrame:
        """Map raw biochemical deltas to categorical labels."""
        work = error_frame.copy()

        # Charge Pattern
        col_charge = self._resolve_feature_column(work, "Charge_Change")
        if col_charge:
            work["pattern__charge"] = "No Change"
            work.loc[work[col_charge] > 0, "pattern__charge"] = "Gain of Positive"
            work.loc[work[col_charge] < 0, "pattern__charge"] = "Loss of Positive"

        # Polarity Pattern
        col_polarity = self._resolve_feature_column(work, "Polarity_Change")
        if col_polarity:
            work["pattern__polarity"] = "Neutral"
            work.loc[work[col_polarity] > 0, "pattern__polarity"] = "Increase"
            work.loc[work[col_polarity] < 0, "pattern__polarity"] = "Decrease"

        # Hydropathy Pattern
        col_hyd = self._resolve_feature_column(work, "Hyd_Delta")
        if col_hyd:
            work["pattern__hydropathy"] = "Stable"
            work.loc[work[col_hyd] > 0.5, "pattern__hydropathy"] = "More Hydrophobic"
            work.loc[work[col_hyd] < -0.5, "pattern__hydropathy"] = "More Hydrophilic"

        return work

    def _analyze_pattern_concentration(self, error_frame: pd.DataFrame) -> dict[str, Any]:
        """Analyze concentration of errors in specific feature bins."""
        results: dict[str, Any] = {}

        # 0. Global Error Statistics
        counts = error_frame["error_type"].value_counts().to_dict()
        results["error_type_distribution"] = {
            "fp": int(counts.get("FP", 0)),
            "fn": int(counts.get("FN", 0)),
            "total": int(len(error_frame))
        }

        # 1. Population Frequency (gnomAD_AF)
        col_af = self._resolve_feature_column(error_frame, "gnomAD_AF")
        if col_af:
            af_vals = pd.to_numeric(error_frame[col_af], errors="coerce")
            non_null = af_vals.dropna()
            freq_stats: dict[str, dict[str, int]] = {}

            def _append_bucket(mask: pd.Series, label: str) -> None:
                subset = error_frame[mask]
                if subset.empty:
                    freq_stats[label] = {"fp": 0, "fn": 0, "total": 0}
                    return
                s_counts = subset["error_type"].value_counts().to_dict()
                freq_stats[label] = {
                    "fp": int(s_counts.get("FP", 0)),
                    "fn": int(s_counts.get("FN", 0)),
                    "total": int(len(subset)),
                }

            if non_null.empty:
                freq_stats = {
                    "Missing/Unusable": {
                        "fp": int((error_frame["error_type"] == "FP").sum()),
                        "fn": int((error_frame["error_type"] == "FN").sum()),
                        "total": int(len(error_frame)),
                    }
                }
            else:
                # If AF is already probability-like [0, 1], use biological bins.
                is_probability_like = bool(((non_null >= 0.0) & (non_null <= 1.0)).all())
                if is_probability_like:
                    _append_bucket((af_vals >= 0.0) & (af_vals < 0.0001), "Rare (<0.01%)")
                    _append_bucket((af_vals >= 0.0001) & (af_vals < 0.01), "Low Freq (0.01-1%)")
                    _append_bucket(af_vals >= 0.01, "Common (>1%)")
                    _append_bucket(af_vals.isna() | (af_vals < 0.0), "Missing/Invalid")
                else:
                    # Fallback for transformed/scaled AF values.
                    ranks = non_null.rank(method="average", pct=True)
                    low_index = ranks[ranks <= 1.0 / 3.0].index
                    mid_index = ranks[(ranks > 1.0 / 3.0) & (ranks <= 2.0 / 3.0)].index
                    high_index = ranks[ranks > 2.0 / 3.0].index

                    _append_bucket(error_frame.index.isin(low_index), "Low (Relative AF)")
                    _append_bucket(error_frame.index.isin(mid_index), "Mid (Relative AF)")
                    _append_bucket(error_frame.index.isin(high_index), "High (Relative AF)")
                    _append_bucket(af_vals.isna(), "Missing/Invalid")

            results["population_frequency"] = freq_stats

        # 2. In-Silico Conflicts (REVEL vs CADD)
        col_revel = self._resolve_feature_column(error_frame, "REVEL_Score")
        col_cadd = self._resolve_feature_column(error_frame, "cadd.phred")
        if col_revel and col_cadd:
            revel = pd.to_numeric(error_frame[col_revel], errors="coerce")
            cadd = pd.to_numeric(error_frame[col_cadd], errors="coerce")

            # Definitions: REVEL Pathogenic > 0.5, CADD Pathogenic > 20
            mask_conflict_1 = (revel > 0.5) & (cadd < 15)  # REVEL says Path, CADD says Benign
            mask_conflict_2 = (revel < 0.25) & (cadd > 25) # REVEL says Benign, CADD says Path

            conflict_stats = {}
            for scenario, mask in [("revel_high_cadd_low", mask_conflict_1), ("revel_low_cadd_high", mask_conflict_2)]:
                subset = error_frame[mask]
                if subset.empty:
                    conflict_stats[scenario] = {"fp": 0, "fn": 0, "total": 0}
                    continue
                s_counts = subset["error_type"].value_counts().to_dict()
                conflict_stats[scenario] = {
                    "fp": int(s_counts.get("FP", 0)),
                    "fn": int(s_counts.get("FN", 0)),
                    "total": int(len(subset))
                }
            results["insilico_conflicts"] = conflict_stats

        # 3. Biochemical Concentrations
        pattern_frame = self._map_biochemical_patterns(error_frame)
        bio_results: dict[str, Any] = {}
        for pat_col in ["pattern__charge", "pattern__polarity", "pattern__hydropathy"]:
            if pat_col in pattern_frame.columns:
                cat_name = pat_col.split("__")[1]
                counts_by_pattern = {}
                for pattern_label, subset in pattern_frame.groupby(pat_col):
                    s_counts = subset["error_type"].value_counts().to_dict()
                    counts_by_pattern[str(pattern_label)] = {
                        "fp": int(s_counts.get("FP", 0)),
                        "fn": int(s_counts.get("FN", 0)),
                        "total": int(len(subset))
                    }
                bio_results[cat_name] = counts_by_pattern

        if bio_results:
            results["biochemical_patterns"] = bio_results

        return results

    @staticmethod
    def _summarize_biological_context(error_frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Aggregate FP/FN counts by available biological grouping columns."""
        candidate_columns = [
            "gene_id",
            "Gene(s)",
            "protein_family",
            "domain_id",
            "Domain",
        ]
        available = [column for column in candidate_columns if column in error_frame.columns]
        if not available:
            return []

        grouped = (
            error_frame.groupby(available + ["error_type"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        pivot = grouped.pivot_table(
            index=available,
            columns="error_type",
            values="count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()

        rows: list[dict[str, Any]] = []
        for _, item in pivot.iterrows():
            fp_count = int(item.get("FP", 0))
            fn_count = int(item.get("FN", 0))
            total_errors = fp_count + fn_count
            if total_errors <= 0:
                continue

            row: dict[str, Any] = {
                "fp_count": fp_count,
                "fn_count": fn_count,
                "total_errors": total_errors,
                "fp_ratio": float(fp_count / total_errors),
                "fn_ratio": float(fn_count / total_errors),
            }
            for column in available:
                value = item.get(column, "")
                row[column] = "" if pd.isna(value) else str(value)
            rows.append(row)

        rows.sort(key=lambda value: int(value.get("total_errors", 0)), reverse=True)
        return rows[:30]

    @staticmethod
    def _save_json(payload: dict[str, Any], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @staticmethod
    def _save_tree_plot(
        model: DecisionTreeClassifier, 
        feature_names: list[str], 
        output_path: Path,
        class_names: list[str] | None = None
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(16, 8))
        
        if class_names is None:
            class_names = ["FN", "FP"]
            
        plot_tree(
            model, 
            feature_names=feature_names, 
            class_names=class_names, 
            filled=True, 
            ax=ax,
            rounded=True,
            proportion=True
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _save_cluster_scatter(
        frame: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        hue_col: str,
        title: str,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        try:
            sns = importlib.import_module("seaborn")

            sns.scatterplot(
                data=frame,
                x=x_col,
                y=y_col,
                hue=hue_col,
                style="error_type",
                palette="tab10",
                ax=ax,
                s=42,
            )
        except Exception:
            for label, sub in frame.groupby(hue_col):
                ax.scatter(sub[x_col], sub[y_col], label=str(label), s=28)
            ax.legend(loc="best", fontsize=8)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _save_surrogate_shap_plot(
        *,
        surrogate_model: DecisionTreeClassifier,
        x: np.ndarray,
        feature_names: list[str],
        output_path: Path,
    ) -> tuple[bool, str | None]:
        try:
            shap = importlib.import_module("shap")

            explainer = shap.TreeExplainer(surrogate_model)
            shap_values = explainer.shap_values(x)
            if isinstance(shap_values, list):
                target_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                target_values = shap_values

            values = np.asarray(target_values)
            if values.ndim == 3:
                if values.shape[-1] > 1:
                    values = values[:, :, 1]
                else:
                    values = values[:, :, 0]
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.ndim != 2:
                return False, f"unexpected_shap_shape={values.shape}"

            feature_matrix = np.asarray(x)
            if feature_matrix.ndim == 1:
                feature_matrix = feature_matrix.reshape(-1, 1)
            if feature_matrix.ndim != 2:
                return False, f"unexpected_feature_shape={feature_matrix.shape}"

            if feature_matrix.shape[1] != values.shape[1]:
                return False, (
                    "shap_feature_mismatch: "
                    f"values_features={values.shape[1]} input_features={feature_matrix.shape[1]}"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                values,
                feature_matrix,
                feature_names=feature_names,
                plot_type="dot",
                max_display=min(len(feature_names), 20),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(output_path, dpi=160)
            plt.close()
            return True, None
        except Exception as exc:
            return False, str(exc)

    def analyze_candidate(
        self,
        *,
        candidate_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
        dataset: pd.DataFrame,
        output_dir: Path,
        detailed: bool,
    ) -> ErrorAnalysisResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        numeric_columns = self._resolve_numeric_columns(dataset)
        error_frame = self._build_error_frame(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            dataset=dataset,
            numeric_columns=numeric_columns,
        )

        artifacts: dict[str, str] = {}
        summary: dict[str, Any] = {
            "candidate": candidate_name,
            "error_count": int(len(error_frame)),
            "error_rate": float(self._safe_divide(float(len(error_frame)), float(len(dataset)))),
            "excluded_positional_feature": "AA_Position",
            "numeric_features_used": list(numeric_columns),
        }

        if error_frame.empty:
            result = ErrorAnalysisResult(status="skipped", summary={**summary, "reason": "no_errors"}, artifacts={})
            payload_path = output_dir / "error_analysis_summary.json"
            self._save_json(result.to_dict(), payload_path)
            artifacts["error_analysis_summary_json"] = str(payload_path)
            return ErrorAnalysisResult(status=result.status, summary=result.summary, artifacts=artifacts)

        # Pattern Concentration Analysis
        # -----------------------------
        patterns = self._analyze_pattern_concentration(error_frame)
        summary["pattern_concentration"] = patterns

        surrogate_summary, surrogate_model, surrogate_x = self._fit_surrogate_tree(
            error_frame=error_frame,
            numeric_columns=numeric_columns,
        )
        summary["surrogate_tree"] = surrogate_summary
        surrogate_feature_names = [column for column in numeric_columns if column in error_frame.columns]

        gene_stats_df, gene_corr_df = self._gene_proxy_analysis(
            y_true=y_true,
            y_pred=y_pred,
            dataset=dataset,
            numeric_columns=numeric_columns,
        )
        summary["gene_proxy"] = {
            "status": "ok" if not gene_stats_df.empty else "skipped",
            "gene_count": int(len(gene_stats_df)),
            "correlation_count": int(len(gene_corr_df)),
        }
        summary["biological_context"] = self._summarize_biological_context(error_frame)

        clustered_frame, clustering_summary = self._cluster_errors(
            error_frame=error_frame,
            numeric_columns=numeric_columns,
        )
        kmeans_profiles = self._cluster_profiles(
            frame=clustered_frame,
            cluster_column="kmeans_cluster",
            numeric_columns=numeric_columns,
        )
        dbscan_profiles = self._cluster_profiles(
            frame=clustered_frame,
            cluster_column="dbscan_cluster",
            numeric_columns=numeric_columns,
        )
        if clustering_summary.get("status") == "ok":
            clustering_summary["kmeans_profiles"] = kmeans_profiles
            clustering_summary["dbscan_profiles"] = dbscan_profiles
        summary["clustering"] = clustering_summary

        summary_path = output_dir / "error_analysis_summary.json"
        self._save_json(summary, summary_path)
        artifacts["error_analysis_summary_json"] = str(summary_path)

        if not gene_stats_df.empty:
            gene_stats_path = output_dir / "gene_proxy_metrics.csv"
            gene_stats_df.to_csv(gene_stats_path, index=False)
            artifacts["gene_proxy_metrics_csv"] = str(gene_stats_path)
        if not gene_corr_df.empty:
            gene_corr_path = output_dir / "gene_proxy_correlations.csv"
            gene_corr_df.to_csv(gene_corr_path, index=False)
            artifacts["gene_proxy_correlations_csv"] = str(gene_corr_path)

        errors_csv_path = output_dir / "error_samples.csv"
        clustered_frame.to_csv(errors_csv_path, index=False)
        artifacts["error_samples_csv"] = str(errors_csv_path)

        if kmeans_profiles:
            kmeans_profile_path = output_dir / "kmeans_cluster_profiles.csv"
            pd.DataFrame(kmeans_profiles).to_csv(kmeans_profile_path, index=False)
            artifacts["kmeans_cluster_profiles_csv"] = str(kmeans_profile_path)
        if dbscan_profiles:
            dbscan_profile_path = output_dir / "dbscan_cluster_profiles.csv"
            pd.DataFrame(dbscan_profiles).to_csv(dbscan_profile_path, index=False)
            artifacts["dbscan_cluster_profiles_csv"] = str(dbscan_profile_path)

        if surrogate_model is not None and surrogate_x is not None:
            feature_names = list(surrogate_feature_names)
            if feature_names:
                tree_plot_path = output_dir / "surrogate_error_tree.png"
                
                y_unique = np.unique(surrogate_model.classes_)
                tree_classes = []
                if 0 in y_unique: tree_classes.append("FN")
                if 1 in y_unique: tree_classes.append("FP")
                
                self._save_tree_plot(
                    surrogate_model, 
                    feature_names, 
                    tree_plot_path, 
                    class_names=tree_classes if tree_classes else None
                )
                artifacts["surrogate_tree_plot_png"] = str(tree_plot_path)

                summary.setdefault("surrogate_tree", {})["plot_file"] = str(tree_plot_path.name)

        if detailed and surrogate_model is not None and surrogate_x is not None:
            feature_names = list(surrogate_feature_names)
            if feature_names:
                shap_plot_path = output_dir / "surrogate_tree_shap.png"
                shap_ok, shap_reason = self._save_surrogate_shap_plot(
                    surrogate_model=surrogate_model,
                    x=surrogate_x,
                    feature_names=feature_names,
                    output_path=shap_plot_path,
                )
                if shap_ok:
                    artifacts["surrogate_tree_shap_png"] = str(shap_plot_path)
                else:
                    summary.setdefault("surrogate_tree", {})["shap_status"] = {
                        "status": "failed",
                        "reason": str(shap_reason),
                    }

        if detailed and "kmeans_cluster" in clustered_frame.columns:
            tsne_plot = output_dir / "errors_tsne_kmeans.png"
            self._save_cluster_scatter(
                clustered_frame,
                x_col="tsne_x",
                y_col="tsne_y",
                hue_col="kmeans_cluster",
                title="Error Clusters (t-SNE + K-Means)",
                output_path=tsne_plot,
            )
            artifacts["errors_tsne_kmeans_png"] = str(tsne_plot)

        if detailed and "dbscan_cluster" in clustered_frame.columns:
            tsne_dbscan_plot = output_dir / "errors_tsne_dbscan.png"
            self._save_cluster_scatter(
                clustered_frame,
                x_col="tsne_x",
                y_col="tsne_y",
                hue_col="dbscan_cluster",
                title="Error Clusters (t-SNE + DBSCAN)",
                output_path=tsne_dbscan_plot,
            )
            artifacts["errors_tsne_dbscan_png"] = str(tsne_dbscan_plot)

        if detailed and {"umap_x", "umap_y", "kmeans_cluster"}.issubset(clustered_frame.columns):
            umap_plot = output_dir / "errors_umap_kmeans.png"
            self._save_cluster_scatter(
                clustered_frame,
                x_col="umap_x",
                y_col="umap_y",
                hue_col="kmeans_cluster",
                title="Error Clusters (UMAP + K-Means)",
                output_path=umap_plot,
            )
            artifacts["errors_umap_kmeans_png"] = str(umap_plot)

        if detailed and {"umap_x", "umap_y", "dbscan_cluster"}.issubset(clustered_frame.columns):
            umap_dbscan_plot = output_dir / "errors_umap_dbscan.png"
            self._save_cluster_scatter(
                clustered_frame,
                x_col="umap_x",
                y_col="umap_y",
                hue_col="dbscan_cluster",
                title="Error Clusters (UMAP + DBSCAN)",
                output_path=umap_dbscan_plot,
            )
            artifacts["errors_umap_dbscan_png"] = str(umap_dbscan_plot)

        final = ErrorAnalysisResult(status="ok", summary=summary, artifacts=artifacts)
        self._save_json(final.to_dict(), summary_path)
        return final
