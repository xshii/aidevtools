"""
prettycli.vscode - VS Code extension integration

Provides communication with the prettycli VS Code extension for
rendering charts, tables, diffs, and other artifacts.

Example:
    >>> from prettycli import vscode
    >>> vscode.show_chart("bar", ["A", "B"], [{"label": "Data", "data": [1, 2]}])
    >>> vscode.show_table(["Name", "Age"], [["Alice", 30]])
"""
from prettycli.vscode.client import (
    VSCodeClient,
    get_client,
    reset_client,
    get_status,
    show_chart,
    show_table,
    show_file,
    show_diff,
    show_image,
    show_markdown,
    show_json,
    show_web,
    show_csv,
    show_excel,
    open_file,
    is_extension_installed,
    install_extension,
)

__all__ = [
    "VSCodeClient",
    "get_client",
    "reset_client",
    "get_status",
    "show_chart",
    "show_table",
    "show_file",
    "show_diff",
    "show_image",
    "show_markdown",
    "show_json",
    "show_web",
    "show_csv",
    "show_excel",
    "open_file",
    "is_extension_installed",
    "install_extension",
]
