"""
Interactive CLI Module

This module provides the CLI class for building interactive command-line
shells with command registration, status bar, and VS Code integration.

Example:
    >>> from prettycli import CLI, BaseCommand
    >>>
    >>> cli = CLI("myapp")
    >>> cli.register(Path("commands/"))
    >>> cli.run()  # Starts interactive shell
"""
import inspect
import shlex
import shutil
import io
import sys
from pathlib import Path
from typing import Dict, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from prettycli.command import BaseCommand, all_commands as all_func_commands, CommandInfo
from prettycli.context import Context
from prettycli.config import load_config
from prettycli.subui import RuntimeStatus, TopToolbar, BottomToolbar, QuoteWidget, WelcomeLayout
from prettycli.shell import ShellSession
from prettycli import ui
from prettycli import vscode
from prettycli import log


class CommandCompleter(Completer):
    """
    Tab completion for CLI commands.

    Provides completions for:
    - Command names (class-based and function-based)
    - Command arguments (--name style)
    - Built-in commands (help, exit, quit)
    """

    def __init__(
        self,
        class_commands: Dict[str, BaseCommand],
        func_commands: Dict[str, CommandInfo],
    ):
        """
        Initialize the completer.

        Args:
            class_commands: Dictionary of class-based commands
            func_commands: Dictionary of function-based commands
        """
        self.class_commands = class_commands
        self.func_commands = func_commands
        self.builtins = ["help", "exit", "quit"]

    def get_completions(self, document, complete_event):  # noqa: F841
        """
        Generate completions for the current input.

        Args:
            document: Current document state
            complete_event: Completion event (unused but required by interface)

        Yields:
            Completion objects
        """
        _ = complete_event  # Required by Completer interface
        text = document.text_before_cursor
        words = text.split()

        # Complete command name
        if not words or (len(words) == 1 and not text.endswith(" ")):
            word = words[0] if words else ""
            # Built-in commands
            for cmd in self.builtins:
                if cmd.startswith(word):
                    yield Completion(
                        cmd,
                        start_position=-len(word),
                        display_meta="built-in"
                    )
            # Class-based commands
            for name, cmd in self.class_commands.items():
                if name.startswith(word):
                    yield Completion(
                        name,
                        start_position=-len(word),
                        display_meta=cmd.help[:30] if cmd.help else ""
                    )
            # Function-based commands
            for name, cmd_info in self.func_commands.items():
                if name.startswith(word):
                    yield Completion(
                        name,
                        start_position=-len(word),
                        display_meta=cmd_info.help[:30] if cmd_info.help else ""
                    )
        # Complete command arguments
        elif len(words) >= 1:
            cmd_name = words[0]
            current_word = words[-1] if not text.endswith(" ") else ""

            # Get function to inspect
            func = None
            if cmd_name in self.class_commands:
                func = self.class_commands[cmd_name].run
            elif cmd_name in self.func_commands:
                func = self.func_commands[cmd_name].func

            if func:
                try:
                    sig = inspect.signature(func)
                    for param_name, param in sig.parameters.items():
                        if param_name in ("self", "ctx", "kwargs"):
                            continue
                        arg_name = f"--{param_name.replace('_', '-')}"
                        if arg_name.startswith(current_word):
                            type_hint = ""
                            if param.annotation != inspect.Parameter.empty:
                                type_hint = getattr(param.annotation, "__name__", str(param.annotation))
                            yield Completion(
                                arg_name,
                                start_position=-len(current_word),
                                display_meta=type_hint
                            )
                except (ValueError, TypeError):
                    pass


class CLI:
    """
    Interactive CLI shell.

    Provides an interactive command-line interface with:
    - Command registration and execution
    - Bash command passthrough (with prefix)
    - Status bar with system info
    - VS Code integration
    - Output collapsing (Ctrl+O)

    Attributes:
        name: CLI application name
        prompt: Input prompt string
        ctx: Shared execution context

    Example:
        >>> cli = CLI("myapp", prompt="$ ")
        >>> cli.register(Path("commands/"))
        >>> cli.run()
    """

    def __init__(
        self,
        name: str,
        config_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize the interactive CLI.

        Args:
            name: Application name
            config_path: Path to YAML configuration file
            project_root: Project root directory (defaults to cwd)
        """
        self.name = name
        self.project_root = (project_root or Path.cwd()).resolve()
        self.ctx = Context()
        self._class_commands: Dict[str, BaseCommand] = {}
        self._func_commands: Dict[str, CommandInfo] = {}
        self._config = load_config(config_path)
        self._runtime_status = RuntimeStatus()
        self._shell: Optional[ShellSession] = None  # 持久化 shell 会话

        # 输出相关状态
        self._last_output: str = ""
        self._collapsed: bool = False
        self._max_collapsed_lines: int = self._config.get("max_collapsed_lines", 10)

        # 固定工具栏
        self._top_toolbar = TopToolbar(name)

        # 底部工具栏：左边每日一句，右边状态
        self._quote_widget = QuoteWidget()
        self._bottom_toolbar = BottomToolbar()
        self._bottom_toolbar.add_left(self._quote_widget)
        self._bottom_toolbar.add_right(vscode.get_status)

        # 欢迎页
        self._welcome_layout = WelcomeLayout(name, project_root=self.project_root)

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

        # Load class-based commands
        for name, cmd_cls in BaseCommand.all().items():
            self._class_commands[name] = cmd_cls()

        # Load function-based commands
        self._func_commands.update(all_func_commands())

        return self

    def _expand_variables(self, line: str) -> str:
        """展开特殊变量"""
        artifact_var = self._config.get("artifact_var", "@@$$")
        if artifact_var and artifact_var in line:
            current_file = vscode.get_client().current_file or ""
            line = line.replace(artifact_var, current_file)
        return line

    def _get_prompt_text(self) -> str:
        """生成 bash 风格的路径提示符"""
        try:
            if self._shell and self._shell.is_alive:
                cwd = Path(self._shell.pwd())
            else:
                cwd = self.project_root

            # 计算相对于 project_root 的路径
            try:
                rel_path = cwd.relative_to(self.project_root)
                if str(rel_path) == ".":
                    path_str = self.name
                else:
                    path_str = f"{self.name}/{rel_path}"
            except ValueError:
                # cwd 不在 project_root 下，计算相对路径用 .. 表示
                try:
                    # 计算从 project_root 到 cwd 的相对路径
                    rel_from_root = self.project_root.relative_to(cwd)
                    # 每一层用 .. 表示
                    dots = "/".join(".." for _ in rel_from_root.parts)
                    path_str = dots if dots else self.name
                except ValueError:
                    # 无法计算相对路径，使用绝对路径
                    path_str = str(cwd)

            return f"{path_str} $ "
        except Exception:
            return f"{self.name} $ "

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

    def _get_shell(self) -> ShellSession:
        """获取或创建持久化 shell 会话"""
        if self._shell is None or not self._shell.is_alive:
            self._shell = ShellSession()
            self._shell.start()
        return self._shell

    def _run_bash(self, line: str) -> int:
        """执行 bash 命令（持久化会话，支持 cd、环境变量等）"""
        self._runtime_status.start(line.split()[0] if line else "bash")
        try:
            shell = self._get_shell()
            result = shell.run(line)
            self._runtime_status.stop()
            output = result.stdout
            if result.stderr:
                output += result.stderr
            self._last_output = output
            if output:
                print(output)
            return result.exit_code
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
        args = self._parse_args(args_str)

        # Check class-based commands
        if cmd_name in self._class_commands:
            return self._run_class_command(cmd_name, args)

        # Check function-based commands
        if cmd_name in self._func_commands:
            return self._run_func_command(cmd_name, args)

        return None

    def _run_class_command(self, cmd_name: str, args: Dict) -> int:
        """执行类式命令"""
        cmd = self._class_commands[cmd_name]

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
            return result if result is not None else 0
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

    def _run_func_command(self, cmd_name: str, args: Dict) -> int:
        """执行函数式命令"""
        cmd_info = self._func_commands[cmd_name]

        self._runtime_status.start(cmd_name)
        try:
            result = cmd_info.func(**args)
            self._runtime_status.stop()
            return result if result is not None else 0
        except KeyboardInterrupt:
            self._runtime_status.stop()
            ui.warn("Interrupted")
            return 130
        except TypeError as e:
            self._runtime_status.stop()
            ui.error(f"Invalid arguments: {e}")
            return 1
        except Exception as e:
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
                ui.print("[dim](Ctrl+O to collapse)[/]")

    def run(self):  # pragma: no cover
        """
        Start the interactive CLI shell.

        Provides an interactive prompt with:
        - Tab completion for commands and arguments
        - Status bar display
        - Keyboard shortcuts (Ctrl+O to toggle output)
        """
        # Set up logging to redirect to CLI
        log.setup()

        # Load built-in commands
        self._load_builtins()

        # 自动安装 VS Code 扩展（只在新安装时提示）
        if not vscode.is_extension_installed():
            ui.info("正在安装 VS Code 扩展...")
            if vscode.install_extension():
                ui.success("VS Code 扩展已安装，请重新加载 VS Code (Cmd+Shift+P → Reload Window)")
            else:
                ui.warn("VS Code 扩展安装失败，部分功能不可用")

        # 尝试连接 VS Code
        vscode.get_client().connect()

        self._welcome_layout.show()

        bash_prefix = self._config.get("bash_prefix", "!")
        toggle_key = self._config.get("toggle_key", "c-o")

        # Set up keyboard shortcuts
        bindings = KeyBindings()

        @bindings.add(toggle_key)
        def _on_toggle(_event):
            self._toggle_output()

        # Set up tab completion
        completer = CommandCompleter(self._class_commands, self._func_commands)

        session = PromptSession(
            key_bindings=bindings,
            completer=completer,
            complete_while_typing=False,  # Only complete on Tab
            bottom_toolbar=self._bottom_toolbar,  # 底部每日一句
            style=Style.from_dict({
                'bottom-toolbar': 'noreverse',  # 去掉背景色
            }),
        )

        def get_prompt():
            """动态生成包含顶部状态栏的提示符"""
            width = shutil.get_terminal_size().columns
            separator = "─" * width
            toolbar = self._top_toolbar.render().value  # HTML字符串
            prompt_text = self._get_prompt_text()
            return HTML(f"{toolbar}\n<style fg='ansibrightblack'>{separator}</style>\n{prompt_text}")

        def clear_prompt_lines():
            """清除prompt的3行（工具栏、分割线、输入行）"""
            sys.stdout.write("\033[3A\033[J")  # 上移3行并清除到底部
            sys.stdout.flush()

        while True:
            try:
                line = session.prompt(get_prompt).strip()

                if not line:
                    clear_prompt_lines()
                    continue

                # 清除prompt行并显示命令
                clear_prompt_lines()
                ui.print(f"[dim]{self._get_prompt_text()}[/]{line}")

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
                ui.print()
                self._quote_widget.next()  # 切换下一条每日一句

            except KeyboardInterrupt:
                ui.print("\nUse 'exit' to quit")
            except EOFError:
                break

        # 清理 shell 会话
        if self._shell:
            self._shell.close()

        ui.print("Bye!")

    def _load_builtins(self):
        """加载内置命令"""
        # Import builtins module to trigger @command decorators
        from prettycli import builtins  # noqa: F401
        self._func_commands.update(all_func_commands())

    def _show_help(self):
        """显示帮助"""
        t = ui.table("Commands", ["Name", "Description"])
        # Class-based commands
        for name, cmd in self._class_commands.items():
            t.add_row(name, cmd.help)
        # Function-based commands
        for name, cmd_info in self._func_commands.items():
            t.add_row(name, cmd_info.help)
        ui.print_table(t)
