"""Phase 9 stabilization integration tests for release readiness."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import pathologic
from pathologic import PathoLogic

_CRITICAL_MARKER_PATTERN = re.compile(r"\b(TODO|FIXME|XXX|BUG)\b")


def _assert_signature_contains(fn: object, expected_params: list[str]) -> None:
    signature = inspect.signature(fn)
    assert list(signature.parameters.keys()) == expected_params


@pytest.mark.integration
@pytest.mark.regression
def test_phase9_stabilization_no_critical_markers_in_runtime_code() -> None:
    tracked_roots = [Path("pathologic"), Path("scripts"), Path("tests")]
    python_like_patterns = ("*.py", "*.yml", "*.yaml", "*.md")

    flagged: list[str] = []
    for root in tracked_roots:
        for pattern in python_like_patterns:
            for file_path in root.rglob(pattern):
                if "__pycache__" in file_path.parts:
                    continue
                content = file_path.read_text(encoding="utf-8")
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if "_CRITICAL_MARKER_PATTERN" in line:
                        continue
                    if _CRITICAL_MARKER_PATTERN.search(line):
                        flagged.append(f"{file_path}:{line_number}:{line.strip()}")

    assert flagged == []


@pytest.mark.integration
@pytest.mark.regression
def test_phase9_stabilization_public_api_is_backward_compatible() -> None:
    _assert_signature_contains(
        PathoLogic.__init__,
        ["self", "model_name", "runtime_model_config"],
    )
    _assert_signature_contains(PathoLogic.train, ["self", "data", "overrides"])
    _assert_signature_contains(PathoLogic.predict, ["self", "data", "overrides"])
    _assert_signature_contains(
        PathoLogic.evaluate,
        ["self", "data", "overrides"],
    )
    _assert_signature_contains(PathoLogic.tune, ["self", "data", "overrides"])
    _assert_signature_contains(
        PathoLogic.explain,
        ["self", "data", "overrides"],
    )
    _assert_signature_contains(
        PathoLogic.fine_tune,
        [
            "self",
            "data",
            "overrides",
        ],
    )


@pytest.mark.integration
@pytest.mark.regression
def test_phase9_stabilization_package_exports_are_stable() -> None:
    assert "PathoLogic" in pathologic.__all__
    assert "ModelBuilder" in pathologic.__all__
    assert isinstance(pathologic.__version__, str)
    assert pathologic.__version__.count(".") == 2


@pytest.mark.integration
@pytest.mark.regression
def test_phase9_stabilization_release_workflow_has_reproducible_gates() -> None:
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    required_snippets = [
        "name: release-validation",
        "workflow_dispatch",
        "tags:",
        "v*",
        "pytest -q",
        "ruff check .",
        "mypy pathologic",
        "python scripts/validate_markdown_snippets.py docs",
        "pytest -q tests/integration/test_phase9_regression.py",
        "pytest -q tests/integration/test_phase9_rc_smoke.py",
        "python -m build",
        "test -d dist",
    ]

    for snippet in required_snippets:
        assert snippet in workflow
