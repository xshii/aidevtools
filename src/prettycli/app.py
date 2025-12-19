import typer
import importlib.util
import inspect
from pathlib import Path
from typing import Callable, get_type_hints

from prettycli.command import BaseCommand
from prettycli.context import Context


class App:
    """CLI 应用构建器"""

    def __init__(self, name: str, help: str = ""):
        self.name = name
        self.help = help
        self.ctx = Context()

    def register(self, path: Path) -> "App":
        """递归遍历目录注册所有命令模块"""
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

        return self

    def _make_handler(self, cmd: BaseCommand) -> Callable:
        """为命令创建 typer handler"""
        run_method = cmd.run
        sig = inspect.signature(run_method)
        hints = get_type_hints(run_method) if hasattr(run_method, "__annotations__") else {}

        # 构建参数
        params = []
        for name, param in sig.parameters.items():
            if name in ("self", "ctx"):
                continue

            annotation = hints.get(name, str)
            default = param.default if param.default != inspect.Parameter.empty else ...

            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )

        def handler(**kwargs):
            return cmd.run(self.ctx, **kwargs)

        # 保留签名供 typer 解析
        handler.__signature__ = inspect.Signature(params)
        handler.__annotations__ = {p.name: p.annotation for p in params}

        return handler

    def build(self) -> typer.Typer:
        """构建 typer 应用"""
        app = typer.Typer(
            name=self.name,
            help=self.help,
            no_args_is_help=True,
        )

        # 保持子命令模式，即使只有一个命令
        @app.callback()
        def main():
            pass

        for name, cmd_cls in BaseCommand.all().items():
            cmd = cmd_cls()
            handler = self._make_handler(cmd)
            app.command(name=name, help=cmd_cls.help)(handler)

        return app

    def run(self):
        """运行 CLI"""
        app = self.build()
        app()
