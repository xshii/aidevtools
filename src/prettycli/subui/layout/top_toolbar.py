"""顶部固定工具栏：系统状态 + VS Code 连接状态"""
import platform
from typing import Callable, List, Tuple, Union

from prompt_toolkit.formatted_text import HTML

__all__ = ["TopToolbar"]

# 样式映射：prompt_toolkit -> rich
STYLE_MAP = {
    "success": "green",
    "warning": "yellow",
    "error": "red",
}

StatusResult = Union[str, Tuple[str, str], None]


class TopToolbar:
    """顶部固定工具栏

    用于在终端顶部显示固定的系统信息和连接状态。

    Example:
        >>> toolbar = TopToolbar("myapp")
        >>> toolbar.register(vscode.get_status)
        >>> session = PromptSession(rprompt=toolbar)
    """

    def __init__(self, app_name: str = "CLI"):
        self._app_name = app_name
        self._providers: List[Callable[[], StatusResult]] = []

    def register(self, provider: Callable[[], StatusResult]):
        """注册状态提供者

        Args:
            provider: 返回 str, (str, style), 或 None 的回调函数
        """
        self._providers.append(provider)

    def render(self) -> HTML:
        """渲染工具栏内容"""
        parts = []

        # 应用名称
        parts.append(f"<b>{self._app_name}</b>")

        # Python 版本
        parts.append(f"Py{platform.python_version()}")

        # 动态状态提供者
        for provider in self._providers:
            try:
                result = provider()
                if not result:
                    continue

                if isinstance(result, tuple):
                    text, style = result
                    if style == "success":
                        parts.append(f'<style fg="ansigreen">{text}</style>')
                    elif style == "warning":
                        parts.append(f'<style fg="ansiyellow">{text}</style>')
                    elif style == "error":
                        parts.append(f'<style fg="ansired">{text}</style>')
                    else:
                        parts.append(text)
                else:
                    parts.append(result)
            except Exception:
                pass

        return HTML(" <style fg='ansibrightblack'>|</style> ".join(parts))

    def render_rich(self) -> str:
        """渲染为 rich 格式字符串"""
        parts = []

        # 应用名称
        parts.append(f"[bold]{self._app_name}[/]")

        # Python 版本
        parts.append(f"Py{platform.python_version()}")

        # 动态状态提供者
        for provider in self._providers:
            try:
                result = provider()
                if not result:
                    continue

                if isinstance(result, tuple):
                    text, style = result
                    color = STYLE_MAP.get(style, "default")
                    parts.append(f"[{color}]{text}[/]")
                else:
                    parts.append(result)
            except Exception:
                pass

        return "[dim]|[/] ".join(parts)

    def __call__(self):
        """prompt_toolkit toolbar 回调"""
        return self.render()
