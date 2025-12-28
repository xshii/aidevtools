from prettycli.command import (
    BaseCommand,
    command,
    get_command,
    all_commands,
)
from prettycli.context import Context
from prettycli.cli import CLI
from prettycli.shell import ShellSession
from prettycli.subui import *
from prettycli import ui
from prettycli import vscode
from prettycli import log

__all__ = [
    # Function-based API
    "command",
    "get_command",
    "all_commands",
    # Class-based API (legacy)
    "BaseCommand",
    "Context",
    # Core
    "CLI",
    "ShellSession",
    "ui",
    "vscode",
    "log",
]
