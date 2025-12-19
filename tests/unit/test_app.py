from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import typer

from prettycli.app import App
from prettycli.command import BaseCommand


class TestAppInit:
    def test_default_values(self):
        app = App("myapp")
        assert app.name == "myapp"
        assert app.help == ""
        assert app.ctx is not None

    def test_with_help(self):
        app = App("myapp", help="My application")
        assert app.help == "My application"


class TestAppRegister:
    def test_register_nonexistent_path(self):
        app = App("test")
        result = app.register(Path("/nonexistent"))
        assert result is app

    def test_register_directory(self, tmp_path):
        BaseCommand._registry.clear()

        cmd_file = tmp_path / "testcmd.py"
        cmd_file.write_text('''
from prettycli.command import BaseCommand

class TestCmd(BaseCommand):
    name = "testcmd"
    help = "Test"

    def run(self, ctx, **kwargs):
        return 0
''')

        app = App("test")
        result = app.register(tmp_path)

        assert result is app
        assert "testcmd" in BaseCommand.all()

    def test_register_skips_underscore(self, tmp_path):
        BaseCommand._registry.clear()

        (tmp_path / "_skip.py").write_text("# skip")
        (tmp_path / "keep.py").write_text('''
from prettycli.command import BaseCommand

class KeepCmd(BaseCommand):
    name = "keep"
    help = "Keep"

    def run(self, ctx, **kwargs):
        return 0
''')

        app = App("test")
        app.register(tmp_path)

        assert "keep" in BaseCommand.all()


class TestAppBuild:
    def test_build_returns_typer(self):
        BaseCommand._registry.clear()

        app = App("test")
        typer_app = app.build()

        assert isinstance(typer_app, typer.Typer)

    def test_build_registers_commands(self, tmp_path):
        BaseCommand._registry.clear()

        class BuildCmd(BaseCommand):
            name = "buildcmd"
            help = "Build test"

            def run(self, ctx, **kwargs):
                return 0

        app = App("test")
        typer_app = app.build()

        # Check command is registered
        assert len(typer_app.registered_commands) > 0


class TestAppMakeHandler:
    def test_make_handler_basic(self):
        BaseCommand._registry.clear()

        class SimpleCmd(BaseCommand):
            name = "simple"
            help = "Simple"

            def run(self, ctx, **kwargs):
                return 42

        app = App("test")
        cmd = SimpleCmd()
        handler = app._make_handler(cmd)

        result = handler()
        assert result == 42

    def test_make_handler_with_args(self):
        BaseCommand._registry.clear()

        class ArgsCmd(BaseCommand):
            name = "args"
            help = "Args"

            def run(self, ctx, name: str = "default", **kwargs):
                return name

        app = App("test")
        cmd = ArgsCmd()
        handler = app._make_handler(cmd)

        result = handler(name="test")
        assert result == "test"
