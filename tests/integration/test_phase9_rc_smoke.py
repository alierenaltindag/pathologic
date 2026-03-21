"""Phase 9 release-candidate smoke tests."""

from __future__ import annotations

import importlib

import pytest

from pathologic import PathoLogic
from pathologic.models import list_registered_models


@pytest.mark.integration
@pytest.mark.rc
@pytest.mark.smoke
def test_phase9_rc_import_surface_is_available() -> None:
    modules = [
        "pathologic",
        "pathologic.core",
        "pathologic.data.loader",
        "pathologic.data.preprocessor",
        "pathologic.engine.trainer",
        "pathologic.engine.evaluator",
        "pathologic.engine.tuner",
        "pathologic.explain.service",
        "pathologic.models.registry",
        "pathologic.search.nas.search",
    ]
    for module_name in modules:
        imported = importlib.import_module(module_name)
        assert imported is not None


@pytest.mark.integration
@pytest.mark.rc
@pytest.mark.smoke
def test_phase9_rc_model_aliases_are_resolvable() -> None:
    aliases = list_registered_models()

    for alias in aliases:
        model = PathoLogic(alias)
        assert model.model_name == alias


@pytest.mark.integration
@pytest.mark.rc
@pytest.mark.smoke
def test_phase9_rc_release_candidate_core_flow(variant_csv_path: str) -> None:
    model = PathoLogic("logreg")
    model.train(variant_csv_path)

    predictions = model.predict(variant_csv_path)
    report = model.evaluate(variant_csv_path)

    assert len(predictions) > 0
    assert "metrics" in report
    assert "f1" in report["metrics"]
