"""Progress helpers built on top of tqdm with safe fallbacks."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from tqdm import tqdm


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def is_progress_enabled() -> bool:
    """Resolve runtime progress visibility policy."""
    return _env_flag("PATHOLOGIC_SHOW_PROGRESS", True)


def is_batch_progress_enabled() -> bool:
    """Resolve runtime batch-level progress visibility policy."""
    return _env_flag("PATHOLOGIC_SHOW_BATCH_PROGRESS", False) and is_progress_enabled()


@contextmanager
def epoch_progress(*, total: int, desc: str, enabled: bool | None = None) -> Iterator[Any]:
    """Yield a tqdm progress bar configured for epoch-level loops."""
    show = is_progress_enabled() if enabled is None else enabled
    bar = tqdm(total=total, desc=desc, disable=not show, leave=True, file=sys.stderr)
    try:
        yield bar
    finally:
        bar.close()


@contextmanager
def step_progress(
    *,
    total: int,
    desc: str,
    enabled: bool | None = None,
    leave: bool = False,
) -> Iterator[Any]:
    """Yield a tqdm progress bar configured for inner batch/step loops."""
    show = is_batch_progress_enabled() if enabled is None else enabled
    bar = tqdm(total=total, desc=desc, disable=not show, leave=leave, file=sys.stderr)
    try:
        yield bar
    finally:
        bar.close()
