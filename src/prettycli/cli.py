import shlex
import subprocess
import io
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from prettycli.command import BaseCommand
from prettycli.context import Context
from prettycli.subui import statusbar, SystemStatus, RuntimeStatus, EchoStatus
from prettycli import ui
from prettycli import vscode


class CLI:
    """交互式 CLI"""

    def __init__(self, name: str, prompt: str = "> ", config_path: Optional[Path] = None):
        self.name = name
        self.prompt = prompt
        self.ctx = Context()
        self._commands: Dict[str, BaseCommand] = {}
        self._config = self._load_config(config_path)
        self._system_status = SystemStatus()
        self._runtime_status = RuntimeStatus()
        self._echo_status = EchoStatus()

        # Register vscode status provider
        statusbar.register(vscode.get_status)

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """加载配置"""
        default = {
            "bash_prefix": "!",
            "toggle_key": "c-o",  # ctrl+o
            "artifact_var": "@@$$",  # 当前 artifact 文件路径变量
        }

        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f) or {}
                default.update(user_config)

        return default

    def register(self, path: Path):
        """注册命令目录"""
        import importlib.util

        if not path.exists():
            return self

        for file in path.rglob("*.py"):
            if file.name.startswith("_"):
                continue

            module_name = file.stem
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

        for name, cmd_cls in BaseCommand.all().items():
            self._commands[name] = cmd_cls()

        return self

    def _expand_variables(self, line: str) -> str:
        """展开特殊变量"""
        artifact_var = self._config.get("artifact_var", "@@$$")
        if artifact_var and artifact_var in line:
            current_file = vscode.get_client().current_file or ""
            line = line.replace(artifact_var, current_file)
        return line

    def _parse_args(self, args_str: str) -> Dict[str, str]:
        """解析参数"""
        args = {}
        tokens = shlex.split(args_str) if args_str else []

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("--"):
                key = token[2:].replace("-", "_")
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    args[key] = tokens[i + 1]
                    i += 2
                else:
                    args[key] = True
                    i += 1
            else:
                i += 1

        return args

    def _run_bash(self, line: str) -> int:
        """执行 bash 命令"""
        self._runtime_status.start(line.split()[0] if line else "bash")
        try:
            result = subprocess.run(line, shell=True, capture_output=True, text=True)
            self._runtime_status.stop()
            output = result.stdout + result.stderr
            self._last_output = output
            print(output, end="")
            return result.returncode
        except KeyboardInterrupt:
            self._runtime_status.stop()
            ui.warn("Interrupted")
            return 130
        except Exception as e:
            self._runtime_status.stop()
            ui.error(f"Bash error: {e}")
            return 1

    def _execute_command(self, line: str) -> Optional[int]:
        """执行注册的命令，不存在返回 None"""
        parts = line.strip().split(maxsplit=1)
        if not parts:
            return 0

        cmd_name = parts[0]
        args_str = parts[1] if len(parts) > 1 else ""

        if cmd_name not in self._commands:
            return None

        cmd = self._commands[cmd_name]
        args = self._parse_args(args_str)

        # 捕获输出
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        self._runtime_status.start(cmd_name)
        try:
            result = cmd.run(self.ctx, **args)
            sys.stdout = old_stdout
            self._runtime_status.stop()
            self._last_output = buffer.getvalue()
            print(self._last_output, end="")
            return result
        except KeyboardInterrupt:
            sys.stdout = old_stdout
            self._runtime_status.stop()
            ui.warn("Interrupted")
            return 130
        except TypeError as e:
            sys.stdout = old_stdout
            self._runtime_status.stop()
            ui.error(f"Invalid arguments: {e}")
            return 1
        except Exception as e:
            sys.stdout = old_stdout
            self._runtime_status.stop()
            ui.error(f"Error: {e}")
            return 1

    def _toggle_output(self):
        """切换输出压缩/展开"""
        if not self._last_output:
            return

        self._collapsed = not self._collapsed
        lines = self._last_output.strip().split("\n")

        # 清屏当前输出区域（简化版：直接重新打印）
        ui.print()
        if self._collapsed and len(lines) > self._max_collapsed_lines:
            for line in lines[:self._max_collapsed_lines]:
                ui.print(line)
            hidden = len(lines) - self._max_collapsed_lines
            ui.print(f"[dim]... ({hidden} more lines, Ctrl+O to expand)[/]")
        else:
            ui.print(self._last_output.strip())
            if len(lines) > self._max_collapsed_lines:
                ui.print(f"[dim](Ctrl+O to collapse)[/]")

    def run(self):  # pragma: no cover
        """启动交互式 CLI"""
        ui.print(f"[bold]{self.name}[/] interactive CLI")
        ui.print("Type 'help' for commands, 'exit' to quit\n")

        bash_prefix = self._config.get("bash_prefix", "!")
        toggle_key = self._config.get("toggle_key", "c-o")

        # 设置快捷键
        bindings = KeyBindings()

        @bindings.add(toggle_key)
        def _(event):
            self._toggle_output()

        session = PromptSession(key_bindings=bindings)

        while True:
            try:
                statusbar.show()
                line = session.prompt(self.prompt).strip()

                if not line:
                    continue

                if line == "exit" or line == "quit":
                    break

                if line == "help":
                    self._show_help()
                    continue

                self._collapsed = False

                # 展开特殊变量
                line = self._expand_variables(line)

                # 有前缀时，需要前缀才执行 bash
                if bash_prefix:
                    if line.startswith(bash_prefix):
                        self._run_bash(line[len(bash_prefix):].strip())
                    else:
                        result = self._execute_command(line)
                        if result is None:
                            ui.error(f"Unknown command: {line.split()[0]}")
                # 无前缀时，匹配不到 command 自动执行 bash
                else:
                    result = self._execute_command(line)
                    if result is None:
                        self._run_bash(line)

                self._runtime_status.show()
                self._system_status.show()
                ui.print()

            except KeyboardInterrupt:
                ui.print("\nUse 'exit' to quit")
            except EOFError:
                break

        ui.print("Bye!")

    def _show_help(self):
        """显示帮助"""
        t = ui.table("Commands", ["Name", "Description"])
        for name, cmd in self._commands.items():
            t.add_row(name, cmd.help)
        ui.print_table(t)
