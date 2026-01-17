"""
Execution Context Module

This module provides the Context class that holds shared state and
configuration for command execution.

Example:
    >>> ctx = Context()
    >>> ctx.set_config("debug", True)
    >>> ctx.get_config("debug")
    True
"""
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path


@dataclass
class Context:
    """
    Command execution context.

    Holds shared state, configuration, and environment information
    that commands can access during execution.

    Attributes:
        cwd: Current working directory
        verbose: Enable verbose output
        config: Configuration dictionary

    Example:
        >>> ctx = Context(verbose=True)
        >>> ctx.set_config("api_key", "secret")
        >>> ctx.get_config("api_key")
        'secret'
    """

    cwd: Path = field(default_factory=Path.cwd)
    verbose: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any):
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
