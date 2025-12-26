"""底部工具栏布局"""
import re
import shutil
from typing import Callable, List, Tuple, Union

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

    def _strip_html(self, text: str) -> str:
        """移除 HTML 标签，计算纯文本长度"""
        return re.sub(r'<[^>]+>', '', text)

    def render(self) -> HTML:
        """渲染为 prompt_toolkit HTML"""
        left = self._render_providers(self._left)
        right = self._render_providers(self._right)

        if left and right:
            # 计算需要填充的空格数
            left_len = len(self._strip_html(left))
            right_len = len(self._strip_html(right))
            width = shutil.get_terminal_size().columns
            space_count = max(1, width - left_len - right_len)
            spaces = ' ' * space_count
            return HTML(f"<i>{left}</i>{spaces}{right}")
        elif left:
            return HTML(f"<i>{left}</i>")
        elif right:
            return HTML(f"{right}")
        else:
            return HTML("")

    def __call__(self):
        """prompt_toolkit toolbar 回调"""
        return self.render()
