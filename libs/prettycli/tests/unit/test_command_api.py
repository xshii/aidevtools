"""Tests for the function-based command API."""
import pytest

from prettycli.command import (
    command,
    get_command,
    all_commands,
    clear_commands,
    CommandInfo,
)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up command registry after each test."""
    yield
    clear_commands()


class TestCommandDecorator:
    """Test @command decorator."""

    def test_register_command(self):
        """Test that decorator registers command."""
        @command("test-cmd", help="Test command")
        def test_cmd():
            return 0

        assert "test-cmd" in all_commands()
        cmd_info = get_command("test-cmd")
        assert cmd_info is not None
        assert cmd_info.name == "test-cmd"
        assert cmd_info.help == "Test command"
        assert cmd_info.func is test_cmd

    def test_register_multiple_commands(self):
        """Test registering multiple commands."""
        @command("cmd1")
        def cmd1():
            return 0

        @command("cmd2")
        def cmd2():
            return 0

        assert "cmd1" in all_commands()
        assert "cmd2" in all_commands()

    def test_command_with_args(self):
        """Test command with arguments."""
        @command("greet", help="Greet user")
        def greet(name: str = "World"):
            return f"Hello, {name}!"

        cmd_info = get_command("greet")
        assert cmd_info is not None
        result = cmd_info.func(name="Alice")
        assert result == "Hello, Alice!"

    def test_command_returns_int(self):
        """Test command returning exit code."""
        @command("exit-code")
        def exit_code():
            return 42

        cmd_info = get_command("exit-code")
        assert cmd_info.func() == 42

    def test_get_nonexistent_command(self):
        """Test getting a command that doesn't exist."""
        assert get_command("nonexistent") is None

    def test_clear_commands(self):
        """Test clearing the command registry."""
        @command("temp")
        def temp():
            return 0

        assert "temp" in all_commands()
        clear_commands()
        assert "temp" not in all_commands()


class TestCommandInfo:
    """Test CommandInfo dataclass."""

    def test_command_info_creation(self):
        """Test creating CommandInfo."""
        def dummy():
            pass

        info = CommandInfo(name="test", func=dummy, help="Test help")
        assert info.name == "test"
        assert info.func is dummy
        assert info.help == "Test help"

    def test_command_info_default_help(self):
        """Test CommandInfo with default help."""
        def dummy():
            pass

        info = CommandInfo(name="test", func=dummy)
        assert info.help == ""
