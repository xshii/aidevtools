"""
Command Module

This module provides two ways to define CLI commands:

1. Function-based (recommended):
    >>> from prettycli import command
    >>> import logging
    >>>
    >>> @command("greet", help="Greet a user")
    ... def greet(name: str = "World"):
    ...     logging.info(f"Hello, {name}!")
    ...     return 0

2. Class-based (legacy):
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
from dataclasses import dataclass
from typing import Dict, Type, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from prettycli.context import Context


# =============================================================================
# Function-based command registry
# =============================================================================

@dataclass
class CommandInfo:
    """Metadata for a registered command."""
    name: str
    func: Callable
    help: str = ""


_func_registry: Dict[str, CommandInfo] = {}


def command(name: str, help: str = ""):
    """Decorator to register a function as a CLI command.

    Args:
        name: Command name (e.g., "install-vscode")
        help: Help text for the command

    Example:
        >>> @command("greet", help="Greet a user")
        ... def greet(name: str = "World"):
        ...     logging.info(f"Hello, {name}!")
        ...     return 0
    """
    def decorator(func: Callable) -> Callable:
        _func_registry[name] = CommandInfo(name=name, func=func, help=help)
        return func
    return decorator


def get_command(name: str) -> Optional[CommandInfo]:
    """Get a registered function command by name."""
    return _func_registry.get(name)


def all_commands() -> Dict[str, CommandInfo]:
    """Get all registered function commands."""
    return _func_registry.copy()


def clear_commands():
    """Clear the function command registry (for testing)."""
    _func_registry.clear()


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
