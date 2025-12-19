from pathlib import Path

from prettycli import ui


# quotes.txt 在 prettycli 根目录
QUOTES_FILE = Path(__file__).parent.parent / "quotes.txt"


class SystemStatus:
    """系统状态栏"""

    def __init__(self):
        self._index = 0
        self._quotes = self._load_quotes()

    def _load_quotes(self) -> list:
        """加载 quotes"""
        if QUOTES_FILE.exists():
            return [line.strip() for line in QUOTES_FILE.read_text().splitlines() if line.strip()]
        return ["Keep coding!"]

    def next_quote(self) -> str:
        """获取下一条 quote"""
        quote = self._quotes[self._index % len(self._quotes)]
        self._index += 1
        return quote

    def show(self):
        """显示系统状态栏"""
        quote = self.next_quote()
        ui.print(f"[dim italic]{quote}[/]")
