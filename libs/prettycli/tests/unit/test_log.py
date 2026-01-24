"""Tests for the logging integration."""
import logging
import pytest
from unittest.mock import patch

from prettycli import log


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up logging after each test."""
    yield
    log.teardown()


class TestPrettyCLIHandler:
    """Test PrettyCLIHandler class."""

    def test_emit_info(self):
        """Test emitting INFO level message."""
        handler = log.PrettyCLIHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test info",
            args=(),
            exc_info=None,
        )

        with patch("prettycli.ui.info") as mock_info:
            handler.emit(record)
            mock_info.assert_called_once_with("Test info")

    def test_emit_warning(self):
        """Test emitting WARNING level message."""
        handler = log.PrettyCLIHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Test warning",
            args=(),
            exc_info=None,
        )

        with patch("prettycli.ui.warn") as mock_warn:
            handler.emit(record)
            mock_warn.assert_called_once_with("Test warning")

    def test_emit_error(self):
        """Test emitting ERROR level message."""
        handler = log.PrettyCLIHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Test error",
            args=(),
            exc_info=None,
        )

        with patch("prettycli.ui.error") as mock_error:
            handler.emit(record)
            mock_error.assert_called_once_with("Test error")

    def test_emit_debug(self):
        """Test emitting DEBUG level message."""
        handler = log.PrettyCLIHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Test debug",
            args=(),
            exc_info=None,
        )

        with patch("prettycli.ui.print") as mock_print:
            handler.emit(record)
            mock_print.assert_called_once()


class TestLogSetup:
    """Test log setup and teardown."""

    def test_setup_adds_handler(self):
        """Test that setup adds handler to root logger."""
        log.setup()

        root = logging.getLogger()
        handlers = [h for h in root.handlers if isinstance(h, log.PrettyCLIHandler)]
        assert len(handlers) == 1

    def test_setup_idempotent(self):
        """Test that calling setup twice doesn't add duplicate handlers."""
        log.setup()
        log.setup()

        root = logging.getLogger()
        handlers = [h for h in root.handlers if isinstance(h, log.PrettyCLIHandler)]
        assert len(handlers) == 1

    def test_teardown_removes_handler(self):
        """Test that teardown removes the handler."""
        log.setup()
        log.teardown()

        root = logging.getLogger()
        handlers = [h for h in root.handlers if isinstance(h, log.PrettyCLIHandler)]
        assert len(handlers) == 0

    def test_logging_after_setup(self):
        """Test that logging works after setup."""
        log.setup()

        with patch("prettycli.ui.info") as mock_info:
            logging.info("Test message")
            mock_info.assert_called_once_with("Test message")
