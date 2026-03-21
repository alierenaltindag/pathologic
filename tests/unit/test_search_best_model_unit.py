from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.search_best_model import (
    _configure_windows_joblib_cpu_detection,
    _build_error_analysis_run_summary,
    _build_candidate_specs,
    _build_global_importance_label,
    _build_hybrid_strategy_tuning_search_space,
    _build_hotspot_label,
    _build_pair_tuning_search_space,
    _extract_scores_from_model,
    _parse_hybrid_weights,
    _parse_model_pool,
    _resolve_hybrid_config_for_report,
    _resolve_hybrid_strategy_config,
    _suppress_known_parallel_warnings,
    build_arg_parser,
    prepare_dataset_for_pathologic,
)


def test_build_candidate_specs_includes_all_singles_and_pairs() -> None:
    candidates = _build_candidate_specs(
        include_models=["logreg", "random_forest", "xgboost"],
        exclude_models=None,
        include_hybrids=True,
        max_candidates=None,
    )
    single_names = [item.name for item in candidates if item.kind == "single"]
    hybrid_names = [item.name for item in candidates if item.kind == "hybrid_pair"]

    assert single_names == ["logreg", "random_forest", "xgboost"]
    assert hybrid_names == [
        "logreg+random_forest",
        "logreg+xgboost",
        "random_forest+xgboost",
    ]


def test_build_pair_tuning_search_space_uses_member_namespace_prefix() -> None:
    search_space = _build_pair_tuning_search_space("logreg", "xgboost")

    assert any(key.startswith("member__logreg__") for key in search_space)
    assert any(key.startswith("member__xgboost__") for key in search_space)
    assert "member__logreg__c" in search_space
    assert "member__xgboost__max_depth" in search_space


def test_build_hybrid_strategy_tuning_search_space_includes_strategy_and_params() -> None:
    search_space = _build_hybrid_strategy_tuning_search_space()

    assert search_space["strategy"]["type"] == "categorical"
    assert "stacking" in search_space["strategy"]["values"]
    assert "strategy__cv" in search_space
    assert "strategy__blend_size" in search_space
    assert "strategy__weight_ratio" in search_space


def test_build_hybrid_strategy_tuning_search_space_uses_profile_ranges() -> None:
    quick_space = _build_hybrid_strategy_tuning_search_space(budget_profile="quick")
    aggressive_space = _build_hybrid_strategy_tuning_search_space(budget_profile="aggressive")

    assert quick_space["strategy"]["values"] == ["soft_voting", "hard_voting"]
    assert quick_space["strategy__cv"]["high"] == 3
    assert aggressive_space["strategy__cv"]["high"] == 5


def test_build_hybrid_strategy_tuning_search_space_applies_yaml_profile_overrides() -> None:
    search_space = _build_hybrid_strategy_tuning_search_space(
        budget_profile="quick",
        search_defaults={
            "hybrid_tuning_space": {
                "quick": {
                    "strategy": ["stacking"],
                    "cv": {"low": 4, "high": 4},
                    "meta_model_alias": ["xgboost"],
                }
            }
        },
    )

    assert search_space["strategy"]["values"] == ["stacking"]
    assert search_space["strategy__cv"]["low"] == 4
    assert search_space["strategy__cv"]["high"] == 4
    assert search_space["meta_model_alias"]["values"] == ["xgboost"]


def test_build_candidate_specs_expands_pair_space_when_hybrid_tune_enabled() -> None:
    candidates = _build_candidate_specs(
        include_models=["logreg", "xgboost"],
        exclude_models=None,
        include_hybrids=True,
        max_candidates=None,
        hybrid_tune_strategy_and_params=True,
    )

    pair = [item for item in candidates if item.kind == "hybrid_pair"][0]
    assert "strategy" in pair.tuning_search_space
    assert "strategy__cv" in pair.tuning_search_space
    assert "strategy__weight_ratio" in pair.tuning_search_space


def test_build_candidate_specs_uses_provided_hybrid_tuning_search_space() -> None:
    candidates = _build_candidate_specs(
        include_models=["logreg", "xgboost"],
        exclude_models=None,
        include_hybrids=True,
        max_candidates=None,
        hybrid_tune_strategy_and_params=True,
        hybrid_tuning_search_space={
            "strategy": {"type": "categorical", "values": ["stacking"]},
            "strategy__cv": {"type": "int", "low": 4, "high": 4},
        },
    )

    pair = [item for item in candidates if item.kind == "hybrid_pair"][0]
    assert pair.tuning_search_space["strategy"]["values"] == ["stacking"]
    assert pair.tuning_search_space["strategy__cv"]["low"] == 4


def test_prepare_dataset_drops_identifier_like_columns(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "VariationID": ["1", "2", "3"],
            "Gene(s)": ["G1", "G2", "G3"],
            "Target": [1, 0, 1],
            "cadd.phred": [20.0, 10.0, 15.0],
        }
    )
    source = tmp_path / "raw.csv"
    output = tmp_path / "prepared.csv"
    raw.to_csv(source, index=False)

    prepared_csv, feature_columns, stats = prepare_dataset_for_pathologic(
        str(source),
        str(output),
    )

    prepared = pd.read_csv(prepared_csv)
    assert "VariationID" not in prepared.columns
    assert any(column.startswith("feature__") for column in feature_columns)
    assert stats["dropped_identifier_column_count"] == 1
    assert stats["dropped_identifier_columns"] == ["VariationID"]


def test_prepare_dataset_drops_explicit_excluded_columns(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "Gene(s)": ["G1", "G2", "G3"],
            "Target": [1, 0, 1],
            "REVEL_Score": [0.8, 0.1, 0.6],
            "Veri_Kaynagi_Paneli": ["panel_a", "panel_b", "panel_c"],
        }
    )
    source = tmp_path / "raw.csv"
    output = tmp_path / "prepared.csv"
    raw.to_csv(source, index=False)

    prepared_csv, feature_columns, stats = prepare_dataset_for_pathologic(
        str(source),
        str(output),
        excluded_columns=["Veri_Kaynagi_Paneli"],
    )

    prepared = pd.read_csv(prepared_csv)
    assert "feature__Veri_Kaynagi_Paneli" not in prepared.columns
    assert "feature__REVEL_Score" in feature_columns
    assert stats["dropped_excluded_column_count"] == 1
    assert stats["dropped_excluded_columns"] == ["Veri_Kaynagi_Paneli"]


def test_prepare_dataset_retains_error_analysis_columns_without_feature_encoding(
    tmp_path: Path,
) -> None:
    raw = pd.DataFrame(
        {
            "Gene(s)": ["G1", "G2", "G3"],
            "Target": [1, 0, 1],
            "REVEL_Score": [0.8, 0.1, 0.6],
            "Veri_Kaynagi_Paneli": ["panel_a", "panel_b", "panel_c"],
        }
    )
    source = tmp_path / "raw.csv"
    output = tmp_path / "prepared.csv"
    raw.to_csv(source, index=False)

    prepared_csv, feature_columns, stats = prepare_dataset_for_pathologic(
        str(source),
        str(output),
        excluded_columns=["Veri_Kaynagi_Paneli"],
        error_analysis_columns=["Veri_Kaynagi_Paneli"],
    )

    prepared = pd.read_csv(prepared_csv)
    assert "Veri_Kaynagi_Paneli" in prepared.columns
    assert "feature__Veri_Kaynagi_Paneli" not in prepared.columns
    assert "feature__REVEL_Score" in feature_columns
    assert stats["retained_error_analysis_columns"] == ["Veri_Kaynagi_Paneli"]
    assert stats["retained_error_analysis_column_count"] == 1


def test_arg_parser_sets_quiet_inner_search_by_default() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["data.csv"])

    assert args.verbose_inner_search is False
    assert args.model_pool == "xgboost,tabnet"
    assert args.error_analysis_mode == "hybrid"
    assert args.disable_error_analysis is False
    assert args.hybrid_strategy == "soft_voting"
    assert args.hybrid_weighting_policy == "auto"
    assert args.hybrid_tune_strategy_and_params is True


def test_arg_parser_allows_verbose_inner_search_flag() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["data.csv", "--verbose-inner-search"])

    assert args.verbose_inner_search is True


def test_parse_hybrid_weights_returns_float_list() -> None:
    assert _parse_hybrid_weights("0.7,0.3") == [0.7, 0.3]
    assert _parse_hybrid_weights(None) is None


def test_resolve_hybrid_strategy_config_includes_manual_weights() -> None:
    candidate = _build_candidate_specs(
        include_models=["tabnet", "xgboost"],
        exclude_models=None,
        include_hybrids=True,
        max_candidates=None,
    )[-1]
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "data.csv",
            "--hybrid-strategy",
            "soft_voting",
            "--hybrid-weights",
            "0.7,0.3",
            "--hybrid-weighting-policy",
            "manual",
            "--hybrid-weighting-objective",
            "f1",
        ]
    )

    strategy, params = _resolve_hybrid_strategy_config(candidate, args)

    assert strategy == "soft_voting"
    assert params["weighting_policy"] == "manual"
    assert params["weights"] == [0.7, 0.3]
    assert params["normalize_weights"] is True


def test_resolve_hybrid_config_for_report_applies_selected_params_overrides() -> None:
    candidate = _build_candidate_specs(
        include_models=["tabnet", "xgboost"],
        exclude_models=None,
        include_hybrids=True,
        max_candidates=None,
    )[-1]
    parser = build_arg_parser()
    args = parser.parse_args(["data.csv", "--hybrid-strategy", "soft_voting"])

    resolved = _resolve_hybrid_config_for_report(
        candidate=candidate,
        args=args,
        selected_params={
            "strategy": "stacking",
            "meta_model_alias": "random_forest",
            "strategy__cv": 5,
            "strategy__weighting_policy": "equal",
        },
    )

    assert resolved["strategy"] == "stacking"
    assert resolved["meta_model"] == "random_forest"
    assert resolved["params"]["cv"] == 5
    assert resolved["params"]["weighting_policy"] == "equal"


def test_parse_model_pool_normalizes_aliases() -> None:
    parsed = _parse_model_pool("xgnet,tabnet,xgb,tabnet")

    assert parsed == ["xgboost", "tabnet"]


def test_build_global_importance_label_prefers_feature_plus_biological_label() -> None:
    label = _build_global_importance_label(
        {
            "feature": "feature__REVEL_Score",
            "biological_label": "General biological feature",
        }
    )

    assert label == "feature__REVEL_Score (General biological feature)"


def test_build_hotspot_label_uses_group_column_value() -> None:
    label = _build_hotspot_label(
        {
            "group_column": "gene_id",
            "gene_id": "HNF1A",
            "false_positive_risk_ratio": 2.0,
        }
    )

    assert label == "gene_id: HNF1A"


def test_extract_scores_from_model_aligns_labels_after_row_drop() -> None:
    class _DropPreprocessor:
        def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame.iloc[1:].copy()

    class _Predictor:
        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.column_stack([1.0 - np.full(len(x), 0.6), np.full(len(x), 0.6)])

    class _ModelStub:
        _preprocessor: Any
        _trained_model: Any

        def __init__(self) -> None:
            self._preprocessor = _DropPreprocessor()
            self._trained_model = _Predictor()

    frame = pd.DataFrame(
        {
            "label": [1, 0, 1],
            "feat_a": [0.1, 0.2, 0.3],
            "feat_b": [1.1, 1.2, 1.3],
        }
    )

    y_true, scores = _extract_scores_from_model(
        model=_ModelStub(),
        dataset=frame,
        feature_columns=["feat_a", "feat_b"],
        label_column="label",
    )

    assert len(y_true) == 2
    assert len(scores) == 2


def test_build_error_analysis_run_summary_collects_candidate_rows() -> None:
    payload = _build_error_analysis_run_summary(
        leaderboard_rows=[
            {
                "candidate": "xgboost",
                "error_analysis": {
                    "status": "ok",
                    "summary": {
                        "error_count": 10,
                        "error_rate": 0.2,
                        "surrogate_tree": {"status": "ok"},
                        "clustering": {"status": "ok"},
                    },
                },
            },
            {
                "candidate": "tabnet",
                "error_analysis": {
                    "status": "skipped",
                    "summary": {
                        "error_count": 0,
                        "error_rate": 0.0,
                        "surrogate_tree": {"status": "skipped"},
                        "clustering": {"status": "skipped"},
                    },
                },
            },
        ],
        winner_candidate="xgboost",
    )

    assert payload["winner_candidate"] == "xgboost"
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["is_winner"] is True
    assert payload["rows"][0]["error_count"] == 10


def test_configure_windows_joblib_cpu_detection_sets_env(monkeypatch: Any) -> None:
    monkeypatch.setattr("scripts.search_best_model.os.name", "nt")
    monkeypatch.setattr("scripts.search_best_model.os.cpu_count", lambda: 12)
    monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)

    _configure_windows_joblib_cpu_detection()

    assert "LOKY_MAX_CPU_COUNT" in os.environ
    assert os.environ["LOKY_MAX_CPU_COUNT"] == "11"


def test_configure_windows_joblib_cpu_detection_keeps_existing_value(monkeypatch: Any) -> None:
    monkeypatch.setattr("scripts.search_best_model.os.name", "nt")
    monkeypatch.setattr("scripts.search_best_model.os.cpu_count", lambda: 24)
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "6")

    _configure_windows_joblib_cpu_detection()

    assert os.environ["LOKY_MAX_CPU_COUNT"] == "6"


def test_suppress_known_parallel_warnings_filters_loky_noise() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _suppress_known_parallel_warnings()
        warnings.warn("Could not find the number of physical cores", UserWarning)

    assert len(caught) == 0
