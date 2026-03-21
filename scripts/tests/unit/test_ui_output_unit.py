"""Unit tests for color and progress UI helpers."""

from __future__ import annotations

from pathologic import PathoLogic
from pathologic.utils.colorstr import colorstr
from pathologic.utils.progress import is_batch_progress_enabled, is_progress_enabled


def test_colorstr_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv("PATHOLOGIC_COLORED_OUTPUT", "0")

    rendered = colorstr("hello", "green", bold=True)

    assert rendered == "hello"


def test_colorstr_enabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv("PATHOLOGIC_COLORED_OUTPUT", "1")

    rendered = colorstr("hello", "green", bold=True)

    assert rendered.startswith("\033[")
    assert rendered.endswith("\033[0m")


def test_progress_flags_follow_env(monkeypatch) -> None:
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.setenv("PATHOLOGIC_SHOW_PROGRESS", "1")
    monkeypatch.setenv("PATHOLOGIC_SHOW_BATCH_PROGRESS", "0")

    assert is_progress_enabled() is True
    assert is_batch_progress_enabled() is False


def test_core_ui_config_applies_env_flags(monkeypatch) -> None:
    monkeypatch.delenv("PATHOLOGIC_COLORED_OUTPUT", raising=False)
    monkeypatch.delenv("PATHOLOGIC_SHOW_PROGRESS", raising=False)
    monkeypatch.delenv("PATHOLOGIC_SHOW_BATCH_PROGRESS", raising=False)

    PathoLogic._apply_ui_runtime_config(
        {
            "colored_output": True,
            "show_progress": False,
            "show_batch_progress": True,
        }
    )

    assert __import__("os").environ["PATHOLOGIC_COLORED_OUTPUT"] == "1"
    assert __import__("os").environ["PATHOLOGIC_SHOW_PROGRESS"] == "0"
    assert __import__("os").environ["PATHOLOGIC_SHOW_BATCH_PROGRESS"] == "1"
