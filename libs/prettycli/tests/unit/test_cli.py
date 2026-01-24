from pathlib import Path
from unittest.mock import patch, MagicMock


from prettycli.cli import CLI
from prettycli.command import BaseCommand, command, clear_commands, CommandInfo


class TestCLIInit:
    def test_default_values(self):
        cli = CLI("test")
        assert cli.name == "test"
        assert cli._class_commands == {}

    def test_default_prompt(self, tmp_path):
        cli = CLI("test", project_root=tmp_path)
        prompt = cli._get_prompt_text()
        assert prompt == "test $ "

    def test_prompt_with_subdirectory(self, tmp_path):
        subdir = tmp_path / "src" / "app"
        subdir.mkdir(parents=True)
        cli = CLI("test", project_root=tmp_path)
        # Simulate shell in subdirectory
        cli._shell = MagicMock()
        cli._shell.is_alive = True
        cli._shell.pwd.return_value = str(subdir)
        prompt = cli._get_prompt_text()
        assert prompt == "test/src/app $ "

    def test_prompt_parent_directory(self, tmp_path):
        """Test prompt when cd to parent directory of project root."""
        cli = CLI("test", project_root=tmp_path)
        cli._shell = MagicMock()
        cli._shell.is_alive = True
        # Go to parent directory
        cli._shell.pwd.return_value = str(tmp_path.parent)
        prompt = cli._get_prompt_text()
        assert prompt == ".. $ "

    def test_prompt_grandparent_directory(self, tmp_path):
        """Test prompt when cd to grandparent directory of project root."""
        cli = CLI("test", project_root=tmp_path)
        cli._shell = MagicMock()
        cli._shell.is_alive = True
        # Go to grandparent directory
        cli._shell.pwd.return_value = str(tmp_path.parent.parent)
        prompt = cli._get_prompt_text()
        assert prompt == "../.. $ "

    def test_prompt_unrelated_path(self, tmp_path):
        """Test prompt when cd to a path unrelated to project root."""
        # Create a separate directory that's not an ancestor of tmp_path
        import tempfile
        with tempfile.TemporaryDirectory() as other_dir:
            cli = CLI("test", project_root=tmp_path)
            cli._shell = MagicMock()
            cli._shell.is_alive = True
            cli._shell.pwd.return_value = other_dir
            prompt = cli._get_prompt_text()
            # Should show absolute path for unrelated directories
            assert prompt == f"{other_dir} $ "

    def test_default_config(self):
        cli = CLI("test")
        assert cli._config["bash_prefix"] == ""
        assert cli._config["toggle_key"] == "c-o"


class TestCLIConfigLoad:
    def test_load_config_no_file(self, tmp_path):
        cli = CLI("test", config_path=tmp_path / "nonexistent.yaml")
        assert cli._config["bash_prefix"] == ""

    def test_load_config_with_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("bash_prefix: '$'\n")
        cli = CLI("test", config_path=config_file)
        assert cli._config["bash_prefix"] == "$"

    def test_load_config_empty_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        cli = CLI("test", config_path=config_file)
        assert cli._config["bash_prefix"] == ""


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
        cli._class_commands["testcmd"] = TestCmd()

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
        cli._class_commands["argscmd"] = ArgsCmd()

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
        cli._class_commands["interrupt"] = InterruptCmd()

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
        cli._class_commands["typeerror"] = TypeErrorCmd()

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
        cli._class_commands["exception"] = ExceptionCmd()

        with patch('prettycli.ui.error'):
            result = cli._execute_command("exception")
            assert result == 1


class TestCLIFuncCommand:
    """Test function-based command execution."""

    def test_execute_func_command(self):
        """Test executing a function-based command."""
        clear_commands()

        @command("func-test", help="Test function command")
        def func_test():
            return 0

        cli = CLI("test")
        cli._func_commands["func-test"] = CommandInfo(
            name="func-test", func=func_test, help="Test"
        )

        result = cli._execute_command("func-test")
        assert result == 0

    def test_execute_func_command_with_args(self):
        """Test executing function command with arguments."""
        clear_commands()

        @command("func-args")
        def func_args(name="default"):
            return 0 if name == "test" else 1

        cli = CLI("test")
        cli._func_commands["func-args"] = CommandInfo(
            name="func-args", func=func_args, help=""
        )

        result = cli._execute_command("func-args --name test")
        assert result == 0

    def test_execute_func_command_exception(self):
        """Test function command that raises exception."""
        clear_commands()

        @command("func-error")
        def func_error():
            raise RuntimeError("test error")

        cli = CLI("test")
        cli._func_commands["func-error"] = CommandInfo(
            name="func-error", func=func_error, help=""
        )

        with patch('prettycli.ui.error'):
            result = cli._execute_command("func-error")
            assert result == 1

    def test_execute_func_command_keyboard_interrupt(self):
        """Test function command interrupted by user."""
        clear_commands()

        @command("func-interrupt")
        def func_interrupt():
            raise KeyboardInterrupt

        cli = CLI("test")
        cli._func_commands["func-interrupt"] = CommandInfo(
            name="func-interrupt", func=func_interrupt, help=""
        )

        with patch('prettycli.ui.warn'):
            result = cli._execute_command("func-interrupt")
            assert result == 130

    def test_execute_func_command_none_return(self):
        """Test function command returning None (should be 0)."""
        clear_commands()

        @command("func-none")
        def func_none():
            pass  # Returns None

        cli = CLI("test")
        cli._func_commands["func-none"] = CommandInfo(
            name="func-none", func=func_none, help=""
        )

        result = cli._execute_command("func-none")
        assert result == 0


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

        assert "mycmd" in cli._class_commands

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

        assert "public" in cli._class_commands
        assert "_private" not in cli._class_commands


class TestCLIShowHelp:
    def test_show_help(self):
        BaseCommand._registry.clear()

        class HelpCmd(BaseCommand):
            name = "helpcmd"
            help = "Help description"

            def run(self, ctx, **kwargs):
                return 0

        cli = CLI("test")
        cli._class_commands["helpcmd"] = HelpCmd()

        with patch('prettycli.ui.print_table'):
            cli._show_help()

    def test_show_help_with_func_commands(self):
        """Test help shows function-based commands."""
        clear_commands()

        @command("func-help", help="Function help")
        def func_help():
            return 0

        cli = CLI("test")
        cli._func_commands["func-help"] = CommandInfo(
            name="func-help", func=func_help, help="Function help"
        )

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
