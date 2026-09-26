"""Tests for package logging configuration."""

import importlib
import logging


def test_package_logger_has_null_handler():
    importlib.import_module("ragframework")
    logger = logging.getLogger("ragframework")

    assert any(isinstance(handler, logging.NullHandler) for handler in logger.handlers)
