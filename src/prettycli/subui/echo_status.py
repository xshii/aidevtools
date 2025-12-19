from prettycli import ui


class EchoStatus:
    """回显状态栏，支持输出压缩/展开"""

    def __init__(self, max_lines: int = 5):
        self._last_output: str = ""
        self._collapsed: bool = False
        self._max_lines = max_lines

    def set_output(self, output: str):
        """设置输出内容"""
        self._last_output = output
        self._collapsed = False

    def toggle(self):
        """切换压缩/展开"""
        if not self._last_output:
            return

        self._collapsed = not self._collapsed
        self._redraw()

    def _redraw(self):
        """重绘输出"""
        lines = self._last_output.strip().split("\n")

        ui.print()
        if self._collapsed and len(lines) > self._max_lines:
            for line in lines[:self._max_lines]:
                ui.print(line)
            hidden = len(lines) - self._max_lines
            ui.print(f"[dim]... ({hidden} more lines, Ctrl+O to expand)[/]")
        else:
            ui.print(self._last_output.strip())
            if len(lines) > self._max_lines:
                ui.print(f"[dim](Ctrl+O to collapse)[/]")

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    @property
    def has_output(self) -> bool:
        return bool(self._last_output)

    def clear(self):
        """清空输出"""
        self._last_output = ""
        self._collapsed = False
