"""Tests for CLI command completion."""
import pytest
from unittest.mock import MagicMock

from prettycli.cli import CommandCompleter
from prettycli.command import BaseCommand
from prettycli.context import Context


class DummyCommand(BaseCommand):
    """Test command."""

    name = "dummy"
    help = "A dummy command for testing"

    def run(self, ctx: Context, name: str = "test", count: int = 1) -> int:
        return 0


class AnotherCommand(BaseCommand):
    """Another test command."""

    name = "another"
    help = "Another test command"

    def run(self, ctx: Context) -> int:
        return 0


from prettycli.command import CommandInfo


def dummy_func(name: str = "test", verbose: bool = False):
    """Dummy function command."""
    return 0


@pytest.fixture
def completer():
    """Create a completer with test commands."""
    class_commands = {
        "dummy": DummyCommand(),
        "another": AnotherCommand(),
    }
    func_commands = {
        "func-cmd": CommandInfo(name="func-cmd", func=dummy_func, help="A function command"),
    }
    return CommandCompleter(class_commands, func_commands)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up command registry after each test."""
    yield
    BaseCommand.clear()


class TestCommandCompleter:
    """Test CommandCompleter class."""

    def test_init(self, completer):
        """Test completer initialization."""
        assert "dummy" in completer.class_commands
        assert "another" in completer.class_commands
        assert "help" in completer.builtins
        assert "exit" in completer.builtins
        assert "quit" in completer.builtins

    def test_complete_empty_input(self, completer):
        """Test completion with empty input."""
        document = MagicMock()
        document.text_before_cursor = ""

        completions = list(completer.get_completions(document, None))

        names = [c.text for c in completions]
        assert "help" in names
        assert "exit" in names
        assert "quit" in names
        assert "dummy" in names
        assert "another" in names

    def test_complete_partial_command(self, completer):
        """Test completion with partial command name."""
        document = MagicMock()
        document.text_before_cursor = "du"

        completions = list(completer.get_completions(document, None))

        names = [c.text for c in completions]
        assert "dummy" in names
        assert "another" not in names

    def test_complete_builtin(self, completer):
        """Test completion of builtin commands."""
        document = MagicMock()
        document.text_before_cursor = "he"

        completions = list(completer.get_completions(document, None))

        names = [c.text for c in completions]
        assert "help" in names

    def test_complete_arguments(self, completer):
        """Test completion of command arguments."""
        document = MagicMock()
        document.text_before_cursor = "dummy --"

        completions = list(completer.get_completions(document, None))

        names = [c.text for c in completions]
        assert "--name" in names
        assert "--count" in names

    def test_complete_partial_argument(self, completer):
        """Test completion of partial argument."""
        document = MagicMock()
        document.text_before_cursor = "dummy --na"

        completions = list(completer.get_completions(document, None))

        names = [c.text for c in completions]
        assert "--name" in names
        assert "--count" not in names

    def test_complete_after_space(self, completer):
        """Test completion after command with space."""
        document = MagicMock()
        document.text_before_cursor = "dummy "

        completions = list(completer.get_completions(document, None))

        names = [c.text for c in completions]
        assert "--name" in names
        assert "--count" in names

    def test_complete_unknown_command(self, completer):
        """Test completion for unknown command arguments."""
        document = MagicMock()
        document.text_before_cursor = "unknown --"

        completions = list(completer.get_completions(document, None))

        # Should return empty for unknown command
        assert len(completions) == 0

    def test_completion_display_meta(self, completer):
        """Test that completions include help text."""
        document = MagicMock()
        document.text_before_cursor = "dum"

        completions = list(completer.get_completions(document, None))

        dummy_completion = next(c for c in completions if c.text == "dummy")
        # display_meta is FormattedText, convert to string for check
        meta_text = str(dummy_completion.display_meta)
        assert "dummy" in meta_text.lower() or "A dummy" in meta_text

    def test_argument_type_hint(self, completer):
        """Test that argument completions show type hints."""
        document = MagicMock()
        document.text_before_cursor = "dummy --cou"

        completions = list(completer.get_completions(document, None))

        count_completion = next(c for c in completions if c.text == "--count")
        # display_meta is FormattedText, convert to string for check
        meta_text = str(count_completion.display_meta)
        assert "int" in meta_text

    def test_complete_func_command(self, completer):
        """Test completion of function-based commands."""
        document = MagicMock()
        document.text_before_cursor = "func"

        completions = list(completer.get_completions(document, None))

        names = [c.text for c in completions]
        assert "func-cmd" in names

    def test_complete_func_command_args(self, completer):
        """Test completion of function command arguments."""
        document = MagicMock()
        document.text_before_cursor = "func-cmd --"

        completions = list(completer.get_completions(document, None))

        names = [c.text for c in completions]
        assert "--name" in names
        assert "--verbose" in names

    def test_func_command_display_meta(self, completer):
        """Test function command shows help in display_meta."""
        document = MagicMock()
        document.text_before_cursor = "func"

        completions = list(completer.get_completions(document, None))

        func_completion = next(c for c in completions if c.text == "func-cmd")
        meta_text = str(func_completion.display_meta)
        assert "function" in meta_text.lower()
