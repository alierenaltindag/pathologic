from __future__ import annotations

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
