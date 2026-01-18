"""Welcome Layout - startup welcome page with ASCII art, shortcuts, and what's new"""
import re
from pathlib import Path
from typing import Optional, List, Tuple

from rich.panel import Panel
from rich.table import Table

__all__ = ["WelcomeLayout"]

# Horse ASCII art
DOC_ART = """\
        ,--,
    _ _/   |
   / `     |
  /   \ , ,'
 (_,\ \ \\ |
    |\\ || `.
    \\ || `|
     \\ `' /
      `-'"""

DEFAULT_SHORTCUTS = [
    ("Ctrl+C", "Cancel"),
    ("Ctrl+D", "Quit"),
]

# CHANGELOG.md 在项目根目录
DEFAULT_CHANGELOG = Path(__file__).parent.parent.parent.parent.parent / "CHANGELOG.md"


class WelcomeLayout:
    """Welcome page layout with ASCII art, shortcuts, and what's new.

    Layout:
    ┌──────────────────────────────────────────┐
    │  [Horse Art]       │   What's New        │
    │                    │   v0.1.0 (2024-12)  │
    │  Shortcuts:        │   • feature 1       │
    │  Tab    complete   │   • feature 2       │
    └──────────────────────────────────────────┘
    """

    def __init__(
        self,
        app_name: str = "PrettyCLI",
        shortcuts: list = None,
        changelog_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
    ):
        self._app_name = app_name
        self._shortcuts = shortcuts or DEFAULT_SHORTCUTS
        self._changelog_path = changelog_path or DEFAULT_CHANGELOG
        self._project_root = project_root or Path.cwd()
        self._version, self._date, self._changes = self._parse_changelog()

    def _parse_changelog(self) -> Tuple[str, str, List[str]]:
        """Parse CHANGELOG.md to get latest version info."""
        if not self._changelog_path.exists():
            return "0.0.0", "", ["No changelog found"]

        content = self._changelog_path.read_text()

        # Match: ## v0.1.0 (2024-12-28)
        version_pattern = r"##\s+v?([\d.]+)\s*\(([^)]+)\)"
        match = re.search(version_pattern, content)

        if not match:
            return "0.0.0", "", ["No version found"]

        version = match.group(1)
        date = match.group(2)

        # Get changes after version header until next ## or end
        start = match.end()
        next_version = re.search(r"\n##\s+", content[start:])
        end = start + next_version.start() if next_version else len(content)

        changes_text = content[start:end].strip()
        changes = [
            line.strip().lstrip("-").strip()
            for line in changes_text.split("\n")
            if line.strip().startswith("-")
        ]

        return version, date, changes or ["No changes listed"]

    def _render_left(self) -> str:
        """Render left side: ASCII art + project root + shortcuts."""
        lines = []

        # ASCII art
        lines.append(f"[cyan]{DOC_ART}[/]")

        # Project root
        lines.append(f"[dim]{self._project_root.name}/[/]")
        lines.append("")

        # Shortcuts
        lines.append("[bold]Shortcuts[/]")
        for key, desc in self._shortcuts:
            lines.append(f"[yellow]{key:8}[/] [dim]{desc}[/]")

        return "\n".join(lines)

    def _render_right(self) -> str:
        """Render right side: What's new with date."""
        lines = []

        # Version header
        lines.append(f"[bold]What's New[/] [dim]({self._date})[/]")
        lines.append("")

        # Changes
        for item in self._changes[:5]:  # Limit to 5 items
            lines.append(f"[dim]•[/] {item}")

        return "\n".join(lines)

    def render(self) -> Panel:
        """Render the welcome layout as a Rich Panel."""
        # Create a table for left-right layout
        table = Table.grid(padding=(0, 3))
        table.add_column("left", justify="left", width=22)
        table.add_column("right", justify="left")

        table.add_row(
            self._render_left(),
            self._render_right(),
        )

        return Panel(
            table,
            title=f"[bold]{self._app_name}[/] [dim]v{self._version}[/]",
            subtitle="[dim]Type 'help' for commands[/]",
            border_style="blue",
        )

    def show(self):
        """Display welcome layout."""
        from prettycli import ui
        ui.print(self.render())
