from typing import Callable, List, Union, Tuple
from rich.console import Console
from rich.text import Text


# 预定义样式
STYLES = {
    "info": "blue",
    "success": "green",
    "warning": "yellow",
    "error": "red",
}

# 返回类型：str 或 (str, style)
StatusResult = Union[str, Tuple[str, str], None]


class StatusBar:
    """状态栏管理器"""

    def __init__(self):
        self._segments: List[Callable[[], StatusResult]] = []
        self._console = Console()

    def register(self, provider: Callable[[], StatusResult]):
        """注册状态栏内容提供者

        provider 返回值:
            - str: 普通文本
            - (str, style): 带样式的文本，style 可以是 info/success/warning/error 或 rich 样式
            - None 或 "": 跳过
        """
        self._segments.append(provider)

    def clear(self):
        """清空所有注册"""
        self._segments.clear()

    def render(self) -> Text:
        """渲染状态栏"""
        parts = []
        for provider in self._segments:
            try:
                result = provider()
                if not result:
                    continue

                if isinstance(result, tuple):
                    content, style = result
                    style = STYLES.get(style, style)
                    parts.append(Text(content, style=style))
                else:
                    parts.append(Text(result))
            except Exception:
                pass

        if not parts:
            return Text("")

        return Text(" | ", style="dim").join(parts)

    def show(self):
        """显示状态栏"""
        content = self.render()
        if content:
            self._console.print(content, style="dim")


# 全局状态栏实例
statusbar = StatusBar()
