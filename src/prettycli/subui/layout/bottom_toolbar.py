"""底部工具栏布局"""
import shutil
from html.parser import HTMLParser
from typing import Callable, List, Tuple, Union

from prompt_toolkit.formatted_text import HTML

__all__ = ["BottomToolbar"]


class _HTMLTextExtractor(HTMLParser):
    """从HTML中提取纯文本"""

    def __init__(self):
        super().__init__()
        self._text_parts: List[str] = []

    def handle_data(self, data: str):
        self._text_parts.append(data)

    def get_text(self) -> str:
        return "".join(self._text_parts)

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
        """移除 HTML 标签，返回纯文本"""
        extractor = _HTMLTextExtractor()
        extractor.feed(text)
        return extractor.get_text()

    def _display_width(self, text: str) -> int:
        """计算文本显示宽度（中文字符占2列）"""
        width = 0
        for char in text:
            # CJK 字符范围
            if '\u4e00' <= char <= '\u9fff' or \
               '\u3000' <= char <= '\u303f' or \
               '\uff00' <= char <= '\uffef':
                width += 2
            else:
                width += 1
        return width

    def render(self) -> HTML:
        """渲染为 prompt_toolkit HTML"""
        left = self._render_providers(self._left)
        right = self._render_providers(self._right)

        # 分割线
        width = shutil.get_terminal_size().columns
        separator = f'<style fg="ansibrightblack">{"─" * width}</style>'

        if left and right:
            # 计算需要填充的空格数（中文字符占2列）
            left_len = self._display_width(self._strip_html(left))
            right_len = self._display_width(self._strip_html(right))
            space_count = max(1, width - left_len - right_len - 2)
            spaces = ' ' * space_count
            content = f"<i>{left}</i>{spaces}{right}"
        elif left:
            content = f"<i>{left}</i>"
        elif right:
            content = f"{right}"
        else:
            content = ""

        return HTML(f"{separator}\n{content}")

    def __call__(self):
        """prompt_toolkit toolbar 回调"""
        return self.render()
