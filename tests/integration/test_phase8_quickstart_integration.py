"""Integration tests for Phase 8 documentation examples and quickstart flow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pathologic import PathoLogic


def _run_example(script_name: str, *args: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "docs" / "examples" / script_name

    completed = subprocess.run(
        [sys.executable, str(script_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines, f"Example script produced no stdout: {script_name}"
    return json.loads(lines[-1])


@pytest.mark.integration
def test_phase8_quickstart_steps_in_clean_environment(variant_csv_path: str) -> None:
    model = PathoLogic("logreg")
    model.train(variant_csv_path)

    predictions = model.predict(variant_csv_path)
    report = model.evaluate(variant_csv_path)

    assert len(predictions) > 0
    assert "metrics" in report
    assert "f1" in report["metrics"]


@pytest.mark.integration
def test_phase8_example_basic_workflow_runs(variant_csv_path: str) -> None:
    payload = _run_example("example_01_basic_workflow.py", variant_csv_path)

    assert payload["model_name"] == "logreg"
    assert int(payload["prediction_count"]) > 0


@pytest.mark.integration
def test_phase8_example_ensemble_workflow_runs(variant_csv_path: str) -> None:
    payload = _run_example("example_02_ensemble_builder.py", variant_csv_path)

    assert str(payload["model_name"]).count("+") >= 1
    assert int(payload["prediction_count"]) > 0


@pytest.mark.integration
def test_phase8_example_finetune_workflow_runs(variant_csv_path: str) -> None:
    payload = _run_example("example_03_finetune.py", variant_csv_path)

    assert payload["model_name"] == "mlp"
    assert isinstance(payload["metric_delta"], dict)


@pytest.mark.integration
def test_phase8_example_explainability_workflow_runs(
    variant_csv_path: str,
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "explainability.html"
    payload = _run_example(
        "example_04_explainability_report.py",
        variant_csv_path,
        "--output-html",
        str(html_path),
    )

    assert payload["model_name"] == "logreg"
    assert int(payload["global_feature_count"]) > 0
    assert payload["html_path"] == str(html_path)
    assert html_path.exists()
