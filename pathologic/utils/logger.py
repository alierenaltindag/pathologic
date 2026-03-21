"""Central logging helpers for PathoLogic modules."""

from __future__ import annotations

import logging

from pathologic.utils.colorstr import colorstr


class _ColorFormatter(logging.Formatter):
    """Log formatter that colorizes level labels in interactive terminals."""

    _LEVEL_COLORS = {
        logging.DEBUG: "blue",
        logging.INFO: "cyan",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "red",
    }

    def format(self, record: logging.LogRecord) -> str:
        level_name = record.levelname
        colored_level = colorstr(
            level_name,
            self._LEVEL_COLORS.get(record.levelno, "white"),
            bold=True,
        )
        original_level = record.levelname
        record.levelname = colored_level
        try:
            return super().format(record)
        finally:
            record.levelname = original_level


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single stream handler.

    This prevents duplicated log lines when the helper is called multiple times
    for the same logger name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_ColorFormatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
