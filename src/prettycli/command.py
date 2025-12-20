"""
Base Command Module

This module provides the BaseCommand abstract class that all CLI commands
must inherit from. Commands are automatically registered when defined.

Example:
    >>> from prettycli import BaseCommand, Context
    >>>
    >>> class GreetCommand(BaseCommand):
    ...     name = "greet"
    ...     help = "Greet a user"
    ...
    ...     def run(self, ctx: Context, name: str = "World") -> int:
    ...         print(f"Hello, {name}!")
    ...         return 0
"""
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prettycli.context import Context


class BaseCommand(ABC):
    """
    Abstract base class for CLI commands.

    All commands must inherit from this class and implement the run() method.
    Commands are automatically registered in a global registry when their
    class is defined (via __init_subclass__).

    Attributes:
        name: Command name used to invoke it (e.g., "greet")
        help: Help text displayed in command listing

    Example:
        >>> class MyCommand(BaseCommand):
        ...     name = "mycommand"
        ...     help = "Do something useful"
        ...
        ...     def run(self, ctx: Context, **kwargs) -> int:
        ...         # Implementation here
        ...         return 0
    """

    name: str = ""
    help: str = ""

    _registry: Dict[str, Type["BaseCommand"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            BaseCommand._registry[cls.name] = cls

    @abstractmethod
    def run(self, ctx: "Context", **kwargs) -> int:
        """
        Execute the command.

        Args:
            ctx: Execution context with configuration and state
            **kwargs: Command arguments parsed from CLI input

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        pass

    @classmethod
    def get(cls, name: str) -> Optional[Type["BaseCommand"]]:
        """
        Get a registered command by name.

        Args:
            name: Command name

        Returns:
            Command class or None if not found
        """
        return cls._registry.get(name)

    @classmethod
    def all(cls) -> Dict[str, Type["BaseCommand"]]:
        """
        Get all registered commands.

        Returns:
            Dictionary mapping command names to command classes
        """
        return cls._registry.copy()

    @classmethod
    def clear(cls):
        """Clear the command registry (for testing)."""
        cls._registry.clear()
