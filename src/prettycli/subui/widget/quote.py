"""每日一句 Widget"""
from pathlib import Path
from typing import Optional, List

__all__ = ["QuoteWidget"]

# quotes.txt 在 prettycli 根目录
DEFAULT_QUOTES_FILE = Path(__file__).parent.parent.parent / "quotes.txt"


class QuoteWidget:
    """每日一句

    使用延迟加载，在首次访问时才读取quotes文件。

    Example:
        >>> quote = QuoteWidget()
        >>> print(quote.current())
    """

    def __init__(self, quotes_file: Optional[Path] = None):
        self._quotes_file = quotes_file or DEFAULT_QUOTES_FILE
        self._index = 0
        self._quotes: Optional[List[str]] = None  # 延迟加载

    @property
    def quotes(self) -> List[str]:
        """延迟加载quotes列表"""
        if self._quotes is None:
            self._quotes = self._load_quotes()
        return self._quotes

    def _load_quotes(self) -> List[str]:
        """加载 quotes 文件"""
        if self._quotes_file.exists():
            return [
                line.strip()
                for line in self._quotes_file.read_text().splitlines()
                if line.strip()
            ]
        return ["Keep coding!"]

    def next(self) -> str:
        """获取下一条 quote"""
        quote = self.quotes[self._index % len(self.quotes)]
        self._index += 1
        return quote

    def current(self) -> str:
        """获取当前 quote（不切换）"""
        return self.quotes[self._index % len(self.quotes)]

    def render(self) -> str:
        """渲染为 rich 格式"""
        return f"[dim italic]{self.current()}[/]"

    def __call__(self) -> str:
        """作为状态提供者"""
        return self.current()
