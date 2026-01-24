"""Tests for built-in commands."""
import pytest
from unittest.mock import patch

from prettycli.command import clear_commands, all_commands


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up command registry after each test."""
    clear_commands()
    yield
    clear_commands()


class TestInstallVSCodeCommand:
    """Test install-vscode command."""

    def test_command_registered(self):
        """Test that install-vscode command is registered."""
        # Import builtins to trigger registration
        from prettycli import builtins  # noqa: F401

        assert "install-vscode" in all_commands()

    def test_already_installed(self):
        """Test when extension is already installed."""
        from prettycli.builtins import install_vscode

        with patch("prettycli.builtins.install_vscode.vscode") as mock_vscode:
            mock_vscode.is_extension_installed.return_value = True

            result = install_vscode.install_vscode()

            assert result == 0
            mock_vscode.install_extension.assert_not_called()

    def test_install_success(self):
        """Test successful installation."""
        from prettycli.builtins import install_vscode

        with patch("prettycli.builtins.install_vscode.vscode") as mock_vscode:
            mock_vscode.is_extension_installed.return_value = False
            mock_vscode.install_extension.return_value = True

            result = install_vscode.install_vscode()

            assert result == 0
            mock_vscode.install_extension.assert_called_once()

    def test_install_failure(self):
        """Test installation failure."""
        from prettycli.builtins import install_vscode

        with patch("prettycli.builtins.install_vscode.vscode") as mock_vscode:
            mock_vscode.is_extension_installed.return_value = False
            mock_vscode.install_extension.return_value = False

            result = install_vscode.install_vscode()

            assert result == 1
            mock_vscode.install_extension.assert_called_once()
