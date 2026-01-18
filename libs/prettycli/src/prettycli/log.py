"""
Logging integration for prettycli.

This module redirects Python's standard logging to the CLI with colored output.
Commands can use standard logging calls and they will be displayed nicely.

Example:
    >>> import logging
    >>> logging.info("Installing...")      # -> blue info message
    >>> logging.warning("Already exists")  # -> yellow warning
    >>> logging.error("Failed")            # -> red error
"""
import logging
from typing import Optional

from prettycli import ui


class PrettyCLIHandler(logging.Handler):
    """Logging handler that outputs to prettycli UI with colors."""

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                ui.error(msg)
            elif record.levelno >= logging.WARNING:
                ui.warn(msg)
            elif record.levelno == logging.INFO:
                ui.info(msg)
            else:
                # DEBUG level
                ui.print(f"[dim]{msg}[/dim]")
        except Exception:
            self.handleError(record)


_handler: Optional[PrettyCLIHandler] = None


def setup(level: int = logging.INFO):
    """Set up logging to redirect to prettycli UI.

    This is called automatically when CLI starts.

    Args:
        level: Minimum logging level (default INFO)
    """
    global _handler

    if _handler is not None:
        return  # Already set up

    _handler = PrettyCLIHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))

    # Add to root logger
    root = logging.getLogger()
    root.addHandler(_handler)
    root.setLevel(level)


def teardown():
    """Remove the prettycli logging handler."""
    global _handler

    if _handler is not None:
        logging.getLogger().removeHandler(_handler)
        _handler = None
