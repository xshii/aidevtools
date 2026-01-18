"""
UI Components Module

This module provides beautiful terminal UI components including:
- Styled output (info, success, error, warning)
- Interactive prompts (select, confirm, text, password, checkbox)
- Progress indicators (spinner, progress bar)
- Tables

All components use Rich for styling and InquirerPy for interactive prompts.

Example:
    >>> from prettycli import ui
    >>>
    >>> ui.info("Starting process...")
    >>> name = ui.text("Enter your name:")
    >>> if ui.confirm("Continue?"):
    ...     ui.success("Done!")
"""
from typing import List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from InquirerPy import inquirer

console = Console()


# ============ Output ============


def info(msg: str):
    """Print an info message with blue icon."""
    console.print(f"[blue]ℹ[/] {msg}")


def success(msg: str):
    """Print a success message with green checkmark."""
    console.print(f"[green]✓[/] {msg}")


def error(msg: str):
    """Print an error message with red X."""
    console.print(f"[red]✗[/] {msg}")


def warn(msg: str):
    """Print a warning message with yellow icon."""
    console.print(f"[yellow]⚠[/] {msg}")


def print(msg: str = "", **kwargs):
    """
    Print a message to the console.

    Args:
        msg: Message to print (supports Rich markup)
        **kwargs: Additional arguments passed to Rich console.print
    """
    console.print(msg, **kwargs)


def panel(content: str, title: str = ""):
    """
    Print content in a bordered panel.

    Args:
        content: Panel content
        title: Panel title
    """
    console.print(Panel(content, title=title))


# ============ Prompts ============


def select(message: str, choices: list, default=None):
    """
    Display a selection prompt.

    Args:
        message: Prompt message
        choices: List of choices
        default: Default selection

    Returns:
        Selected choice
    """
    return inquirer.select(
        message=message,
        choices=choices,
        default=default,
        pointer="❯",
    ).execute()


def confirm(message: str, default: bool = True) -> bool:
    """
    Display a yes/no confirmation prompt.

    Args:
        message: Prompt message
        default: Default value (True for yes)

    Returns:
        True if confirmed, False otherwise
    """
    return inquirer.confirm(message=message, default=default).execute()


def text(message: str, default: str = "") -> str:
    """
    Display a text input prompt.

    Args:
        message: Prompt message
        default: Default value

    Returns:
        User input string
    """
    return inquirer.text(message=message, default=default).execute()


def password(message: str) -> str:
    """
    Display a password input prompt (hidden input).

    Args:
        message: Prompt message

    Returns:
        Password string
    """
    return inquirer.secret(message=message).execute()


def checkbox(message: str, choices: list) -> list:
    """
    Display a multi-select checkbox prompt.

    Args:
        message: Prompt message
        choices: List of choices

    Returns:
        List of selected choices
    """
    return inquirer.checkbox(
        message=message,
        choices=choices,
        pointer="❯",
    ).execute()


def fuzzy(message: str, choices: list):
    """
    Display a fuzzy search selection prompt.

    Args:
        message: Prompt message
        choices: List of choices

    Returns:
        Selected choice
    """
    return inquirer.fuzzy(
        message=message,
        choices=choices,
        max_height="50%",
    ).execute()


# ============ Progress ============


def spinner(message: str = "Working..."):
    """
    Create a spinner context manager.

    Args:
        message: Status message to display

    Returns:
        Context manager for spinner display

    Example:
        >>> with ui.spinner("Loading..."):
        ...     time.sleep(2)
    """
    return console.status(message)


def progress():
    """
    Create a progress bar context manager.

    Returns:
        Progress bar context manager

    Example:
        >>> with ui.progress() as p:
        ...     task = p.add_task("Processing", total=100)
        ...     for i in range(100):
        ...         p.update(task, advance=1)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


# ============ Table ============


def table(title: str = "", columns: List[str] = None) -> Table:
    """
    Create a table for displaying data.

    Args:
        title: Table title
        columns: Column headers

    Returns:
        Rich Table object

    Example:
        >>> t = ui.table("Users", ["Name", "Age"])
        >>> t.add_row("Alice", "30")
        >>> ui.print_table(t)
    """
    t = Table(title=title)
    if columns:
        for col in columns:
            t.add_column(col)
    return t


def print_table(t: Table):
    """
    Print a table to the console.

    Args:
        t: Rich Table object to print
    """
    console.print(t)
