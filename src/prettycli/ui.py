from typing import List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from InquirerPy import inquirer

console = Console()


# ============ Output ============


def info(msg: str):
    console.print(f"[blue]ℹ[/] {msg}")


def success(msg: str):
    console.print(f"[green]✓[/] {msg}")


def error(msg: str):
    console.print(f"[red]✗[/] {msg}")


def warn(msg: str):
    console.print(f"[yellow]⚠[/] {msg}")


def print(msg: str = "", **kwargs):
    console.print(msg, **kwargs)


def panel(content: str, title: str = ""):
    console.print(Panel(content, title=title))


# ============ Prompts ============


def select(message: str, choices: list, default=None):
    """选择列表"""
    return inquirer.select(
        message=message,
        choices=choices,
        default=default,
        pointer="❯",
    ).execute()


def confirm(message: str, default: bool = True) -> bool:
    """确认提示"""
    return inquirer.confirm(message=message, default=default).execute()


def text(message: str, default: str = "") -> str:
    """文本输入"""
    return inquirer.text(message=message, default=default).execute()


def password(message: str) -> str:
    """密码输入"""
    return inquirer.secret(message=message).execute()


def checkbox(message: str, choices: list) -> list:
    """多选"""
    return inquirer.checkbox(
        message=message,
        choices=choices,
        pointer="❯",
    ).execute()


def fuzzy(message: str, choices: list):
    """模糊搜索选择"""
    return inquirer.fuzzy(
        message=message,
        choices=choices,
        max_height="50%",
    ).execute()


# ============ Progress ============


def spinner(message: str = "Working..."):
    """返回一个 spinner context manager"""
    return console.status(message)


def progress():
    """返回一个 progress bar context manager"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


# ============ Table ============


def table(title: str = "", columns: List[str] = None) -> Table:
    """创建表格"""
    t = Table(title=title)
    if columns:
        for col in columns:
            t.add_column(col)
    return t


def print_table(t: Table):
    console.print(t)
