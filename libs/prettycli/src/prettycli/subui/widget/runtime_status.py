import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text

__all__ = ["RuntimeStatus"]


class RuntimeStatus:
    """实时状态栏，显示命令执行时长等"""

    def __init__(self):
        self._console = Console()
        self._start_time: Optional[float] = None
        self._running: bool = False
        self._command: str = ""
        self._live: Optional[Live] = None
        self._duration: float = 0

    def _format_duration(self, seconds: float) -> str:
        """格式化时长"""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m{secs:.0f}s"

    def _render(self) -> Text:
        """渲染状态栏"""
        if not self._running:
            if self._duration > 0:
                return Text(f"took {self._format_duration(self._duration)}", style="dim")
            return Text("")

        elapsed = time.time() - self._start_time
        return Text(f"⏱ {self._command} running... {self._format_duration(elapsed)}", style="dim yellow")

    def start(self, command: str):
        """开始计时"""
        self._command = command
        self._start_time = time.time()
        self._running = True
        self._duration = 0

    def stop(self):
        """停止计时"""
        if self._start_time:
            self._duration = time.time() - self._start_time
        self._running = False

    def show(self):
        """显示最终状态"""
        if self._duration > 0:
            self._console.print(f"[dim]took {self._format_duration(self._duration)}[/]")

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def is_running(self) -> bool:
        return self._running
