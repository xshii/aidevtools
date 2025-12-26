import io
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from prettycli.cli import CLI
from prettycli.command import BaseCommand


class TestCLIInit:
    def test_default_values(self):
        cli = CLI("test")
        assert cli.name == "test"
        assert cli.prompt == "> "
        assert cli._commands == {}

    def test_custom_prompt(self):
        cli = CLI("test", prompt=">>> ")
        assert cli.prompt == ">>> "

    def test_default_config(self):
        cli = CLI("test")
        assert cli._config["bash_prefix"] == "!"
        assert cli._config["toggle_key"] == "c-o"


class TestCLIConfigLoad:
    def test_load_config_no_file(self, tmp_path):
        cli = CLI("test", config_path=tmp_path / "nonexistent.yaml")
        assert cli._config["bash_prefix"] == "!"

    def test_load_config_with_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("bash_prefix: '$'\n")
        cli = CLI("test", config_path=config_file)
        assert cli._config["bash_prefix"] == "$"

    def test_load_config_empty_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        cli = CLI("test", config_path=config_file)
        assert cli._config["bash_prefix"] == "!"


class TestCLIParseArgs:
    def test_empty_args(self):
        cli = CLI("test")
        assert cli._parse_args("") == {}

    def test_flag_arg(self):
        cli = CLI("test")
        result = cli._parse_args("--verbose")
        assert result == {"verbose": True}

    def test_value_arg(self):
        cli = CLI("test")
        result = cli._parse_args("--name foo")
        assert result == {"name": "foo"}

    def test_multiple_args(self):
        cli = CLI("test")
        result = cli._parse_args("--name foo --count 5 --verbose")
        assert result == {"name": "foo", "count": "5", "verbose": True}

    def test_hyphen_to_underscore(self):
        cli = CLI("test")
        result = cli._parse_args("--my-option value")
        assert result == {"my_option": "value"}

    def test_quoted_value(self):
        cli = CLI("test")
        result = cli._parse_args('--msg "hello world"')
        assert result == {"msg": "hello world"}


class TestCLIRunBash:
    def test_run_bash_success(self):
        cli = CLI("test")
        result = cli._run_bash("echo hello")
        assert result == 0

    def test_run_bash_failure(self):
        cli = CLI("test")
        result = cli._run_bash("exit 1")
        assert result == 1

    def test_run_bash_captures_output(self):
        cli = CLI("test")
        cli._run_bash("echo test")
        assert "test" in cli._last_output

    def test_run_bash_empty_line(self):
        cli = CLI("test")
        result = cli._run_bash("")
        assert result == 0

    def test_run_bash_keyboard_interrupt(self):
        cli = CLI("test")

        with patch.object(cli, '_get_shell') as mock_shell:
            mock_shell.return_value.run.side_effect = KeyboardInterrupt
            with patch('prettycli.ui.warn'):
                result = cli._run_bash("sleep 10")
                assert result == 130

    def test_run_bash_exception(self):
        cli = CLI("test")

        with patch.object(cli, '_get_shell') as mock_shell:
            mock_shell.return_value.run.side_effect = Exception("test error")
            with patch('prettycli.ui.error'):
                result = cli._run_bash("bad")
                assert result == 1


class TestCLIExecuteCommand:
    def test_execute_unknown_command(self):
        cli = CLI("test")
        result = cli._execute_command("unknown")
        assert result is None

    def test_execute_empty_line(self):
        cli = CLI("test")
        result = cli._execute_command("")
        assert result == 0

    def test_execute_registered_command(self, tmp_path):
        # Clear registry first
        BaseCommand._registry.clear()

        # Create a test command
        class TestCmd(BaseCommand):
            name = "testcmd"
            help = "Test command"

            def run(self, ctx, **kwargs):
                print("executed")
                return 0

        cli = CLI("test")
        cli._commands["testcmd"] = TestCmd()

        result = cli._execute_command("testcmd")
        assert result == 0
        assert "executed" in cli._last_output

    def test_execute_command_with_args(self):
        BaseCommand._registry.clear()

        class ArgsCmd(BaseCommand):
            name = "argscmd"
            help = "Args command"

            def run(self, ctx, name="default", **kwargs):
                print(f"name={name}")
                return 0

        cli = CLI("test")
        cli._commands["argscmd"] = ArgsCmd()

        result = cli._execute_command("argscmd --name test")
        assert result == 0
        assert "name=test" in cli._last_output

    def test_execute_command_keyboard_interrupt(self):
        BaseCommand._registry.clear()

        class InterruptCmd(BaseCommand):
            name = "interrupt"
            help = "Interrupt"

            def run(self, ctx, **kwargs):
                raise KeyboardInterrupt

        cli = CLI("test")
        cli._commands["interrupt"] = InterruptCmd()

        with patch('prettycli.ui.warn'):
            result = cli._execute_command("interrupt")
            assert result == 130

    def test_execute_command_type_error(self):
        BaseCommand._registry.clear()

        class TypeErrorCmd(BaseCommand):
            name = "typeerror"
            help = "TypeError"

            def run(self, ctx, required_arg, **kwargs):
                return 0

        cli = CLI("test")
        cli._commands["typeerror"] = TypeErrorCmd()

        with patch('prettycli.ui.error'):
            result = cli._execute_command("typeerror")
            assert result == 1

    def test_execute_command_exception(self):
        BaseCommand._registry.clear()

        class ExceptionCmd(BaseCommand):
            name = "exception"
            help = "Exception"

            def run(self, ctx, **kwargs):
                raise RuntimeError("test error")

        cli = CLI("test")
        cli._commands["exception"] = ExceptionCmd()

        with patch('prettycli.ui.error'):
            result = cli._execute_command("exception")
            assert result == 1


class TestCLIRegister:
    def test_register_directory(self, tmp_path):
        BaseCommand._registry.clear()

        # Create a command file
        cmd_file = tmp_path / "mycmd.py"
        cmd_file.write_text('''
from prettycli.command import BaseCommand

class MyCmd(BaseCommand):
    name = "mycmd"
    help = "My command"

    def run(self, ctx, **kwargs):
        return 0
''')

        cli = CLI("test")
        cli.register(tmp_path)

        assert "mycmd" in cli._commands

    def test_register_nonexistent_directory(self):
        cli = CLI("test")
        result = cli.register(Path("/nonexistent"))
        assert result is cli  # returns self

    def test_register_skips_underscore_files(self, tmp_path):
        BaseCommand._registry.clear()

        # Create files
        (tmp_path / "_private.py").write_text("# private")
        (tmp_path / "public.py").write_text('''
from prettycli.command import BaseCommand

class PublicCmd(BaseCommand):
    name = "public"
    help = "Public"

    def run(self, ctx, **kwargs):
        return 0
''')

        cli = CLI("test")
        cli.register(tmp_path)

        assert "public" in cli._commands
        assert "_private" not in cli._commands


class TestCLIShowHelp:
    def test_show_help(self):
        BaseCommand._registry.clear()

        class HelpCmd(BaseCommand):
            name = "helpcmd"
            help = "Help description"

            def run(self, ctx, **kwargs):
                return 0

        cli = CLI("test")
        cli._commands["helpcmd"] = HelpCmd()

        with patch('prettycli.ui.print_table') as mock:
            cli._show_help()
            mock.assert_called_once()


class TestCLIToggleOutput:
    def test_toggle_no_output(self):
        cli = CLI("test")
        cli._last_output = ""
        cli._toggle_output()  # should not raise

    def test_toggle_with_output(self):
        cli = CLI("test")
        cli._last_output = "line1\nline2\nline3"
        cli._collapsed = False
        cli._max_collapsed_lines = 2

        with patch('prettycli.ui.print'):
            cli._toggle_output()
            assert cli._collapsed is True

    def test_toggle_expand(self):
        cli = CLI("test")
        cli._last_output = "line1\nline2\nline3\nline4\nline5"
        cli._collapsed = True
        cli._max_collapsed_lines = 2

        with patch('prettycli.ui.print'):
            cli._toggle_output()
            assert cli._collapsed is False

    def test_toggle_short_output(self):
        cli = CLI("test")
        cli._last_output = "short"
        cli._collapsed = False
        cli._max_collapsed_lines = 5

        with patch('prettycli.ui.print'):
            cli._toggle_output()
            assert cli._collapsed is True


class TestCLIParseArgsEdge:
    def test_positional_args_ignored(self):
        cli = CLI("test")
        result = cli._parse_args("positional --flag")
        assert result == {"flag": True}

    def test_consecutive_flags(self):
        cli = CLI("test")
        result = cli._parse_args("--a --b --c")
        assert result == {"a": True, "b": True, "c": True}
