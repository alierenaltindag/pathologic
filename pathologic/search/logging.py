"""Logging and runtime-noise controls for search workflows."""

from __future__ import annotations

import io
import logging
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from pathologic.utils.colorstr import colorstr


def colorize(text: str, color: str, *, bold: bool = False) -> str:
    """Apply PathoLogic color formatting for console output."""
    return colorstr(text, color, bold=bold)


def build_run_logger(run_dir: Path) -> tuple[logging.Logger, Path]:
    """Create a per-run file logger for search script outputs."""
    log_path = run_dir / "search_best_model.log"
    logger = logging.getLogger(f"pathologic.search_best_model.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, log_path


def emit(
    message: str,
    *,
    color: str,
    bold: bool = False,
    run_logger: logging.Logger | None = None,
) -> None:
    """Print colorized console output and mirror plain message to run log."""
    print(colorize(message, color, bold=bold))
    if run_logger is not None:
        run_logger.info(message)


@contextmanager
def inner_search_runtime(
    *,
    quiet: bool,
    show_inner_progress: bool = False,
    suppress_stdout: bool = True,
    suppress_stderr: bool = True,
):
    """Temporarily suppress noisy inner-search progress and logs."""
    if not quiet:
        yield
        return

    env_names = (
        "PATHOLOGIC_SHOW_PROGRESS",
        "PATHOLOGIC_SHOW_BATCH_PROGRESS",
    )
    original_env = {name: os.environ.get(name) for name in env_names}
    progress_value = "1" if show_inner_progress else "0"
    os.environ["PATHOLOGIC_SHOW_PROGRESS"] = progress_value
    os.environ["PATHOLOGIC_SHOW_BATCH_PROGRESS"] = progress_value

    logger_names = (
        "pathologic.core",
        "pathologic.engine.tuner",
        "pathologic.nas.search",
        "optuna",
        "optuna.study",
    )
    original_levels: dict[str, int] = {}
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.WARNING)

    original_disable = logging.root.manager.disable
    logging.disable(logging.INFO)

    sink_stdout = io.StringIO()
    sink_stderr = io.StringIO()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Best weights from best epoch are automatically used!",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The given NumPy array is not writable",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Could not find the number of physical cores",
        )
        try:
            if suppress_stdout and suppress_stderr:
                with redirect_stdout(sink_stdout), redirect_stderr(sink_stderr):
                    yield
            elif suppress_stdout:
                with redirect_stdout(sink_stdout):
                    yield
            elif suppress_stderr:
                with redirect_stderr(sink_stderr):
                    yield
            else:
                yield
        finally:
            for name, value in original_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            for logger_name, level in original_levels.items():
                logging.getLogger(logger_name).setLevel(level)
            logging.disable(original_disable)
