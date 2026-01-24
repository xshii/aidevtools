"""交互式测试会话"""
import io
import sys
import threading
import queue
import time
import re
from typing import Optional, List, Callable
from dataclasses import dataclass
from contextlib import contextmanager

__all__ = ["InteractiveSession", "TestResult", "TimeoutError"]


class TimeoutError(Exception):
    """等待超时"""
    pass


@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    output: str = ""
    error: Optional[str] = None
    duration: float = 0.0


class OutputCapture:
    """捕获标准输出"""

    def __init__(self):
        self._buffer = io.StringIO()
        self._original_stdout = None
        self._original_stderr = None

    def start(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self._buffer
        sys.stderr = self._buffer

    def stop(self):
        if self._original_stdout:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    def get_output(self) -> str:
        return self._buffer.getvalue()

    def clear(self):
        self._buffer = io.StringIO()


class InputQueue:
    """模拟用户输入队列"""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._delay = 0.05  # 输入间隔

    def send(self, text: str):
        """发送文本输入"""
        for char in text:
            self._queue.put(char)
            time.sleep(self._delay)

    def send_line(self, text: str):
        """发送一行（自动加回车）"""
        self.send(text + "\n")

    def send_key(self, key: str):
        """发送特殊按键"""
        key_map = {
            "enter": "\n",
            "tab": "\t",
            "escape": "\x1b",
            "ctrl+c": "\x03",
            "ctrl+d": "\x04",
            "ctrl+o": "\x0f",
            "up": "\x1b[A",
            "down": "\x1b[B",
            "left": "\x1b[D",
            "right": "\x1b[C",
            "backspace": "\x7f",
        }
        self._queue.put(key_map.get(key, key))

    def read(self, timeout: float = 1.0) -> Optional[str]:
        """读取一个字符"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class InteractiveSession:
    """交互式测试会话

    用于测试交互式 CLI 应用。

    Example:
        >>> with InteractiveSession() as session:
        ...     session.start(my_cli_func)
        ...     session.expect("Enter name:")
        ...     session.send_line("Alice")
        ...     session.expect("Hello, Alice!")
    """

    def __init__(self, timeout: float = 5.0):
        self._timeout = timeout
        self._output_capture = OutputCapture()
        self._input_queue = InputQueue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._error: Optional[Exception] = None
        self._results: List[TestResult] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, target: Callable, *args, **kwargs):
        """启动目标函数（在后台线程运行）"""
        self._running = True
        self._output_capture.start()

        def run():
            try:
                target(*args, **kwargs)
            except Exception as e:
                self._error = e
            finally:
                self._running = False

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        time.sleep(0.1)  # 等待启动

    def stop(self):
        """停止会话"""
        self._running = False
        self._output_capture.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    @property
    def output(self) -> str:
        """获取当前输出"""
        return self._output_capture.get_output()

    def clear_output(self):
        """清空输出缓冲"""
        self._output_capture.clear()

    def send(self, text: str):
        """发送文本"""
        self._input_queue.send(text)

    def send_line(self, text: str):
        """发送一行"""
        self._input_queue.send_line(text)

    def send_key(self, key: str):
        """发送按键: enter, tab, ctrl+c, up, down, etc."""
        self._input_queue.send_key(key)

    def expect(self, pattern: str, timeout: Optional[float] = None) -> bool:
        """等待输出匹配模式

        Args:
            pattern: 正则表达式或普通字符串
            timeout: 超时时间

        Returns:
            是否匹配成功

        Raises:
            TimeoutError: 超时未匹配
        """
        timeout = timeout or self._timeout
        start = time.time()
        regex = re.compile(pattern)

        while time.time() - start < timeout:
            output = self.output
            if regex.search(output):
                return True
            time.sleep(0.05)

        raise TimeoutError(f"Timeout waiting for: {pattern}\nGot: {self.output}")

    def expect_not(self, pattern: str, timeout: float = 0.5) -> bool:
        """确保输出不包含某模式"""
        try:
            self.expect(pattern, timeout=timeout)
            return False
        except TimeoutError:
            return True

    def wait_idle(self, timeout: Optional[float] = None):
        """等待输出稳定（无新输出）"""
        timeout = timeout or self._timeout
        last_output = ""
        stable_count = 0

        start = time.time()
        while time.time() - start < timeout:
            current = self.output
            if current == last_output:
                stable_count += 1
                if stable_count >= 3:
                    return
            else:
                stable_count = 0
                last_output = current
            time.sleep(0.1)

    @contextmanager
    def test(self, name: str):
        """测试用例上下文

        Example:
            >>> with session.test("login"):
            ...     session.send_line("admin")
            ...     session.expect("Welcome")
        """
        start = time.time()
        error = None
        try:
            yield
            passed = True
        except Exception as e:
            passed = False
            error = str(e)

        result = TestResult(
            name=name,
            passed=passed,
            output=self.output,
            error=error,
            duration=time.time() - start,
        )
        self._results.append(result)

    @property
    def results(self) -> List[TestResult]:
        """获取所有测试结果"""
        return self._results

    def summary(self) -> str:
        """生成测试摘要"""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed

        lines = [f"\n{'=' * 50}"]
        lines.append(f"Tests: {total} | Passed: {passed} | Failed: {failed}")
        lines.append("=" * 50)

        for r in self._results:
            icon = "✓" if r.passed else "✗"
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"{icon} [{status}] {r.name} ({r.duration:.2f}s)")
            if r.error:
                lines.append(f"    Error: {r.error}")

        return "\n".join(lines)
