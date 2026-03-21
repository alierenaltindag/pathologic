"""Unit tests for central logger helper."""

from __future__ import annotations

import logging

from pathologic.utils import get_logger


def test_get_logger_is_idempotent_for_same_name() -> None:
    logger_name = "pathologic.tests.logger"
    logger_a = get_logger(logger_name)
    handler_count = len(logger_a.handlers)

    logger_b = get_logger(logger_name)

    assert logger_a is logger_b
    assert len(logger_b.handlers) == handler_count


def test_get_logger_has_stream_handler_and_no_propagation() -> None:
    logger = get_logger("pathologic.tests.stream")

    assert any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
    assert logger.propagate is False
