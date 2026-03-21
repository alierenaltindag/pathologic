"""ANSI color helpers for terminal output."""

from __future__ import annotations

import os
import sys

_COLOR_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def _is_enabled() -> bool:
    env_value = os.getenv("PATHOLOGIC_COLORED_OUTPUT")
    if env_value is not None:
        return env_value.strip().lower() not in {"0", "false", "no", "off"}
    return sys.stdout.isatty()


def colorstr(text: str, color: str = "reset", *, bold: bool = False) -> str:
    """Return text wrapped in ANSI color codes when enabled."""
    if not _is_enabled():
        return text

    color_code = _COLOR_CODES.get(color.strip().lower(), _COLOR_CODES["reset"])
    prefix = _COLOR_CODES["bold"] + color_code if bold else color_code
    return f"{prefix}{text}{_COLOR_CODES['reset']}"


def info_text(text: str) -> str:
    return colorstr(text, "cyan")


def success_text(text: str) -> str:
    return colorstr(text, "green", bold=True)


def warning_text(text: str) -> str:
    return colorstr(text, "yellow", bold=True)


def error_text(text: str) -> str:
    return colorstr(text, "red", bold=True)
