import pytest
from typer.testing import CliRunner

from prettycli import BaseCommand


runner = CliRunner()


class TestCLI:
    def test_hello_command(self, app_with_fixtures):
        """测试单文件形式的 hello 命令"""
        typer_app = app_with_fixtures.build()

        result = runner.invoke(typer_app, ["hello", "--name", "test"])

        assert result.exit_code == 0
        assert "Hello, test!" in result.output

    def test_greet_command(self, app_with_fixtures):
        """测试文件夹形式的 greet 命令"""
        typer_app = app_with_fixtures.build()

        result = runner.invoke(typer_app, ["greet", "--name", "Alice"])

        assert result.exit_code == 0
        assert "Hello, Alice!" in result.output

    def test_greet_loud(self, app_with_fixtures):
        """测试 greet 命令的 loud 参数"""
        typer_app = app_with_fixtures.build()

        result = runner.invoke(typer_app, ["greet", "--name", "Bob", "--loud"])

        assert result.exit_code == 0
        assert "HELLO, BOB!" in result.output

    def test_help_shows_all_commands(self, app_with_fixtures):
        """测试 help 显示所有命令"""
        typer_app = app_with_fixtures.build()

        result = runner.invoke(typer_app, ["--help"])

        assert "hello" in result.output
        assert "greet" in result.output

    def test_command_default_args(self, app_with_fixtures):
        """测试命令默认参数"""
        typer_app = app_with_fixtures.build()

        result = runner.invoke(typer_app, ["hello"])

        assert result.exit_code == 0
        assert "Hello, world!" in result.output
