"""持久化 Shell 会话"""
import subprocess
import threading
import queue
import time
import os
import re
import signal
from typing import Optional, List
from dataclasses import dataclass

__all__ = ["ShellSession", "ShellResult"]


@dataclass
class ShellResult:
    """命令执行结果"""
    command: str
    stdout: str
    stderr: str
    exit_code: int
    duration: float


class ShellSession:
    """持久化 Shell 会话

    保持工作目录、环境变量等状态。

    Example:
        >>> with ShellSession() as sh:
        ...     sh.run("cd /tmp")
        ...     sh.run("pwd")  # 输出 /tmp
        ...     sh.run("export FOO=bar")
        ...     sh.run("echo $FOO")  # 输出 bar
    """

    # 用于标记命令结束的特殊字符串
    _END_MARKER = "__PRETTYCLI_END_MARKER__"
    _EXIT_CODE_MARKER = "__PRETTYCLI_EXIT_CODE__"

    def __init__(
        self,
        shell: str = "/bin/bash",
        cwd: str = None,
        env: dict = None,
        timeout: float = 30.0,
    ):
        self._shell = shell
        self._cwd = cwd or os.getcwd()
        self._env = {**os.environ, **(env or {})}
        self._timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._stdout_queue: queue.Queue = queue.Queue()
        self._stderr_queue: queue.Queue = queue.Queue()
        self._reader_threads: List[threading.Thread] = []
        self._history: List[ShellResult] = []
        self._data_event: threading.Event = threading.Event()  # 用于通知数据到达

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        """启动 shell 进程"""
        self._process = subprocess.Popen(
            [self._shell],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._cwd,
            env=self._env,
            text=True,
            bufsize=1,  # 行缓冲
        )

        # 启动输出读取线程
        self._start_reader(self._process.stdout, self._stdout_queue)
        self._start_reader(self._process.stderr, self._stderr_queue)

        # 等待 shell 就绪
        time.sleep(0.1)

    def _start_reader(self, stream, q: queue.Queue):
        """启动流读取线程"""
        def reader():
            try:
                for line in iter(stream.readline, ''):
                    if line:
                        q.put(line)
                        self._data_event.set()  # 通知有新数据
            except (ValueError, OSError):
                pass  # 流已关闭

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        self._reader_threads.append(thread)

    def run(self, command: str, timeout: float = None) -> ShellResult:
        """执行命令并等待完成

        Args:
            command: 要执行的命令
            timeout: 超时时间

        Returns:
            ShellResult 包含输出和退出码
        """
        if not self._process or self._process.poll() is not None:
            raise RuntimeError("Shell session not started or already closed")

        timeout = timeout or self._timeout
        start_time = time.time()

        # 清空队列和事件
        self._drain_queue(self._stdout_queue)
        self._drain_queue(self._stderr_queue)
        self._data_event.clear()

        # 发送命令，附加退出码和结束标记
        full_command = (
            f"{command}\n"
            f"echo {self._EXIT_CODE_MARKER}$?\n"
            f"echo {self._END_MARKER}\n"
        )
        self._process.stdin.write(full_command)
        self._process.stdin.flush()

        # 收集输出
        stdout_lines = []
        stderr_lines = []
        exit_code = 0

        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                raise TimeoutError(f"Command timed out after {timeout}s: {command}")

            # 使用Event等待数据，比固定轮询更高效
            self._data_event.wait(timeout=min(remaining, 0.5))
            self._data_event.clear()

            # 读取所有可用的 stdout
            while True:
                try:
                    line = self._stdout_queue.get_nowait()
                    if self._END_MARKER in line:
                        # 读取剩余 stderr
                        try:
                            while True:
                                stderr_lines.append(self._stderr_queue.get_nowait())
                        except queue.Empty:
                            pass
                        # 返回结果
                        duration = time.time() - start_time
                        result = ShellResult(
                            command=command,
                            stdout="".join(stdout_lines).rstrip("\n"),
                            stderr="".join(stderr_lines).rstrip("\n"),
                            exit_code=exit_code,
                            duration=duration,
                        )
                        self._history.append(result)
                        return result
                    elif self._EXIT_CODE_MARKER in line:
                        # 提取退出码
                        match = re.search(rf"{self._EXIT_CODE_MARKER}(\d+)", line)
                        if match:
                            exit_code = int(match.group(1))
                    else:
                        stdout_lines.append(line)
                except queue.Empty:
                    break

            # 读取所有可用的 stderr
            try:
                while True:
                    stderr_lines.append(self._stderr_queue.get_nowait())
            except queue.Empty:
                pass

    def _drain_queue(self, q: queue.Queue):
        """清空队列"""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def send(self, text: str):
        """发送文本（不等待完成）"""
        if self._process and self._process.stdin:
            self._process.stdin.write(text)
            self._process.stdin.flush()

    def send_line(self, text: str):
        """发送一行"""
        self.send(text + "\n")

    def send_signal(self, sig: int):
        """发送信号"""
        if self._process:
            self._process.send_signal(sig)

    def interrupt(self):
        """发送 Ctrl+C"""
        self.send_signal(signal.SIGINT)

    def pwd(self) -> str:
        """获取当前工作目录"""
        result = self.run("pwd")
        return result.stdout.strip()

    def cd(self, path: str) -> bool:
        """切换目录"""
        result = self.run(f"cd {path}")
        return result.exit_code == 0

    def env(self, name: str) -> str:
        """获取环境变量"""
        result = self.run(f"echo ${name}")
        return result.stdout.strip()

    def export(self, name: str, value: str):
        """设置环境变量"""
        self.run(f"export {name}={value}")

    @property
    def history(self) -> List[ShellResult]:
        """命令历史"""
        return self._history

    @property
    def last_result(self) -> Optional[ShellResult]:
        """最后一次执行结果"""
        return self._history[-1] if self._history else None

    def close(self):
        """关闭 shell 会话"""
        if self._process:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                self._process.kill()
            self._process = None

    @property
    def is_alive(self) -> bool:
        """检查进程是否存活"""
        return self._process is not None and self._process.poll() is None
