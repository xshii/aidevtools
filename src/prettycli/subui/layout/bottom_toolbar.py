"""底部工具栏布局"""
from typing import Callable, List, Optional, Tuple, Union

from prompt_toolkit.formatted_text import HTML

__all__ = ["BottomToolbar"]

ContentProvider = Callable[[], Union[str, Tuple[str, str], None]]


class BottomToolbar:
    """底部工具栏布局

    纯布局组件，接受内容提供者。

    Example:
        >>> from prettycli.subui.widget import QuoteWidget
        >>> quote = QuoteWidget()
        >>> toolbar = BottomToolbar()
        >>> toolbar.add_left(quote)
        >>> toolbar.add_right(vscode.get_status)
    """

    def __init__(self):
        self._left: List[ContentProvider] = []
        self._right: List[ContentProvider] = []

    def add_left(self, provider: ContentProvider):
        """添加左侧内容"""
        self._left.append(provider)

    def add_right(self, provider: ContentProvider):
        """添加右侧内容"""
        self._right.append(provider)

    def clear(self):
        """清空所有内容"""
        self._left.clear()
        self._right.clear()

    def _render_providers(self, providers: List[ContentProvider]) -> str:
        """渲染内容提供者列表"""
        parts = []
        for provider in providers:
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

        return " <style fg='ansibrightblack'>|</style> ".join(parts)

    def render(self) -> HTML:
        """渲染为 prompt_toolkit HTML"""
        left = self._render_providers(self._left)
        right = self._render_providers(self._right)

        if left and right:
            # 左右对齐：左边内容 ... 右边内容
            return HTML(f"<i>{left}</i><right>{right}</right>")
        elif left:
            return HTML(f"<i>{left}</i>")
        elif right:
            return HTML(f"<right>{right}</right>")
        else:
            return HTML("")

    def __call__(self):
        """prompt_toolkit toolbar 回调"""
        return self.render()
