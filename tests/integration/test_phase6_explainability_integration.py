"""Integration tests for Phase 6 explainability API."""

from __future__ import annotations

import pytest

from pathologic import PathoLogic


@pytest.mark.integration
def test_core_explain_returns_report_shape(variant_csv_path: str) -> None:
    model = PathoLogic("logreg")
    model.train(variant_csv_path)

    report = model.explain(variant_csv_path)

    assert "backend" in report
    assert "global_feature_importance" in report
    assert "sample_explanations" in report
    assert "false_positive_hotspots" in report
    assert "metadata" in report
    assert len(report["global_feature_importance"]) > 0
    assert len(report["sample_explanations"]) > 0
    assert report["metadata"]["model_name"] == "logreg"
    assert report["metadata"]["background_source"] == "train_split"
    assert report["metadata"]["backend_policy"] == "auto"
    assert "visual_report_html" in report
    assert "<html>" in report["visual_report_html"]


@pytest.mark.integration
def test_core_explain_handles_missing_optional_group_columns(variant_csv_path: str) -> None:
    model = PathoLogic("logreg")
    model.train(variant_csv_path)

    report = model.explain(variant_csv_path)

    assert isinstance(report["false_positive_hotspots"], list)


@pytest.mark.integration
def test_core_explain_hybrid_includes_member_explainability(variant_csv_path: str) -> None:
    model = PathoLogic("tabnet+xgboost")
    model.train(variant_csv_path)

    report = model.explain(variant_csv_path, top_k_features=3, top_k_samples=3)

    member_payload = report.get("member_explainability")
    assert isinstance(member_payload, dict)
    assert member_payload.get("status") == "ok"
    members = member_payload.get("members")
    assert isinstance(members, dict)
    assert "tabnet" in members
    assert "xgboost" in members

    tabnet_member = members["tabnet"]
    xgboost_member = members["xgboost"]
    assert isinstance(tabnet_member, dict)
    assert isinstance(xgboost_member, dict)
    assert tabnet_member.get("status") == "ok"
    assert xgboost_member.get("status") == "ok"
    assert isinstance(tabnet_member.get("global_feature_importance"), list)
    assert isinstance(xgboost_member.get("global_feature_importance"), list)
    assert "group_columns" in report.get("metadata", {})
    assert "Member Explainability" in str(report.get("visual_report_html", ""))
