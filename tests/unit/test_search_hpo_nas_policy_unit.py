from __future__ import annotations

import pandas as pd

from pathologic.search import hpo_nas as _search_hpo_nas
from pathologic.search.spec import CandidateSpec


def _single(alias: str) -> CandidateSpec:
    return CandidateSpec(
        name=alias,
        kind="single",
        members=(alias,),
        tuning_search_space={},
    )


def test_should_run_nas_for_neural_single_models() -> None:
    assert _search_hpo_nas.should_run_nas_for_candidate(_single("mlp")) is True
    assert _search_hpo_nas.should_run_nas_for_candidate(_single("tabnet")) is True


def test_should_skip_nas_for_non_neural_single_models() -> None:
    assert _search_hpo_nas.should_run_nas_for_candidate(_single("xgboost")) is False
    assert _search_hpo_nas.should_run_nas_for_candidate(_single("logreg")) is False


def test_should_run_nas_for_hybrid_with_neural_member() -> None:
    hybrid = CandidateSpec(
        name="tabnet+xgboost",
        kind="hybrid_pair",
        members=("tabnet", "xgboost"),
        tuning_search_space={},
    )

    assert _search_hpo_nas.should_run_nas_for_candidate(hybrid) is True


def test_should_skip_nas_for_hybrid_without_neural_member() -> None:
    hybrid = CandidateSpec(
        name="xgboost+catboost",
        kind="hybrid_pair",
        members=("xgboost", "catboost"),
        tuning_search_space={},
    )

    assert _search_hpo_nas.should_run_nas_for_candidate(hybrid) is False


def test_candidate_stage_order_is_nas_then_hpo_for_neural_single() -> None:
    order = _search_hpo_nas.candidate_stage_order(_single("mlp"))

    assert order[0] == "nas"
    assert order[1] == "hpo"


def test_candidate_stage_order_is_hpo_then_nas_for_non_neural() -> None:
    order = _search_hpo_nas.candidate_stage_order(_single("xgboost"))

    assert order[0] == "hpo"
    assert order[1] == "nas"


def test_candidate_stage_order_for_hybrid_uses_two_level_hpo() -> None:
    hybrid = CandidateSpec(
        name="xgboost+catboost",
        kind="hybrid_pair",
        members=("xgboost", "catboost"),
        tuning_search_space={},
    )

    order = _search_hpo_nas.candidate_stage_order(hybrid)

    assert order[0] == "nas"
    assert order[1] == "hpo_level1"
    assert order[2] == "hpo_level2"


def test_split_hybrid_hpo_search_space_separates_member_and_strategy_scopes() -> None:
    level1, level2 = _search_hpo_nas.split_hybrid_hpo_search_space(
        {
            "member__xgboost__max_depth": {"type": "int", "low": 3, "high": 8},
            "member__catboost__l2_leaf_reg": {"type": "float", "low": 1.0, "high": 8.0},
            "strategy": {"type": "categorical", "values": ["soft_voting", "stacking"]},
            "strategy__cv": {"type": "int", "low": 2, "high": 5},
            "meta_model_alias": {"type": "categorical", "values": ["logreg", "xgboost"]},
        }
    )

    assert "member__xgboost__max_depth" in level1
    assert "member__catboost__l2_leaf_reg" in level1
    assert "strategy" in level2
    assert "strategy__cv" in level2
    assert "meta_model_alias" in level2


def test_merge_hpo_level_results_prefers_level2_source_when_available() -> None:
    merged = _search_hpo_nas.merge_hpo_level_results(
        level1_result={
            "status": "ok",
            "best_score": 0.71,
            "best_params": {"member__xgboost__max_depth": 6},
            "trials": 3,
        },
        level2_result={
            "status": "ok",
            "best_score": 0.75,
            "best_params": {"strategy": "stacking", "strategy__cv": 4},
            "trials": 3,
        },
    )

    assert merged["status"] == "ok"
    assert merged["selected_params_source"] == "hpo_level2_after_level1"
    assert merged["best_params"]["member__xgboost__max_depth"] == 6
    assert merged["best_params"]["strategy"] == "stacking"
    assert merged["trials"] == 6


def test_build_hybrid_neural_nas_search_space_filters_non_neural_member_keys() -> None:
    hybrid = CandidateSpec(
        name="tabnet+xgboost",
        kind="hybrid_pair",
        members=("tabnet", "xgboost"),
        tuning_search_space={},
    )
    filtered = _search_hpo_nas.build_hybrid_neural_nas_search_space(
        candidate=hybrid,
        search_space={
            "member__tabnet__n_steps": {"type": "int", "low": 3, "high": 6},
            "member__xgboost__max_depth": {"type": "int", "low": 3, "high": 8},
            "strategy": {"type": "categorical", "values": ["stacking"]},
        },
    )

    assert "member__tabnet__n_steps" in filtered
    assert "member__xgboost__max_depth" not in filtered
    assert "strategy" not in filtered


def test_build_nas_arrays_applies_tabnet_auto_missingness_policy(monkeypatch) -> None:
    train_df = pd.DataFrame(
        {
            "label": [0, 1, 0, 1, 0, 1],
            "gene_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "feature__a": [1.0, None, 3.0, 4.0, None, 6.0],
            "feature__b": [0.1, 0.2, None, 0.4, 0.5, None],
        }
    )

    class _FakePathoLogic:
        def __init__(self, _alias: str) -> None:
            self.defaults = {
                "train": {
                    "preprocess": {
                        "missing_value_policy": "none",
                        "impute_strategy": "none",
                        "tabnet_missingness_mode": "auto",
                        "tabnet_impute_strategy": "median",
                    }
                }
            }

    monkeypatch.setattr("pathologic.search.hpo_nas.PathoLogic", _FakePathoLogic)

    x_train, y_train, x_val, y_val = _search_hpo_nas.build_nas_arrays(
        train_df=train_df,
        feature_columns=["feature__a", "feature__b"],
        seed=42,
    )

    assert x_train.shape[0] > 0
    assert x_val.shape[0] > 0
    assert y_train.shape[0] == x_train.shape[0]
    assert y_val.shape[0] == x_val.shape[0]
