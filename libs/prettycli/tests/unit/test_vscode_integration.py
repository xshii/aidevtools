"""Tests for VS Code integration in CLI."""
from unittest.mock import patch, MagicMock

from prettycli.cli import CLI
from prettycli import vscode


class TestVSCodeConnection:
    """Test VS Code connection during CLI lifecycle."""

    def test_cli_connects_to_vscode_on_startup(self):
        """CLI should attempt to connect to VS Code on startup."""
        cli = CLI("test")

        with patch.object(vscode, 'get_client') as mock_get_client, \
             patch.object(vscode, 'is_extension_installed', return_value=True), \
             patch('prettycli.cli.log'), \
             patch.object(cli, '_load_builtins'), \
             patch.object(cli, '_welcome_layout'), \
             patch('prompt_toolkit.PromptSession') as mock_session:

            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Mock session to raise EOFError immediately to exit run loop
            mock_session.return_value.prompt.side_effect = EOFError

            with patch('prettycli.ui.print'):
                cli.run()

            # Verify connect was called
            mock_client.connect.assert_called_once()

    def test_cli_works_when_vscode_not_available(self):
        """CLI should work normally when VS Code is not available."""
        cli = CLI("test")

        with patch.object(vscode, 'get_client') as mock_get_client, \
             patch.object(vscode, 'is_extension_installed', return_value=True), \
             patch('prettycli.cli.log'), \
             patch.object(cli, '_load_builtins'), \
             patch.object(cli, '_welcome_layout'), \
             patch('prompt_toolkit.PromptSession') as mock_session:

            mock_client = MagicMock()
            mock_client.connect.return_value = False  # Connection fails
            mock_client.is_connected = False
            mock_get_client.return_value = mock_client

            # Mock session to raise EOFError immediately
            mock_session.return_value.prompt.side_effect = EOFError

            with patch('prettycli.ui.print'):
                # Should not raise exception even when VS Code unavailable
                cli.run()

            mock_client.connect.assert_called_once()

    def test_cli_installs_extension_when_missing(self):
        """CLI should attempt to install extension when not installed."""
        cli = CLI("test")

        with patch.object(vscode, 'get_client') as mock_get_client, \
             patch.object(vscode, 'is_extension_installed', return_value=False), \
             patch.object(vscode, 'install_extension', return_value=True) as mock_install, \
             patch('prettycli.cli.log'), \
             patch.object(cli, '_load_builtins'), \
             patch.object(cli, '_welcome_layout'), \
             patch('prompt_toolkit.PromptSession') as mock_session, \
             patch('prettycli.ui.info'), \
             patch('prettycli.ui.success'):

            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_session.return_value.prompt.side_effect = EOFError

            with patch('prettycli.ui.print'):
                cli.run()

            mock_install.assert_called_once()


class TestVSCodeStatus:
    """Test VS Code status display."""

    def test_status_shows_disconnected_when_not_connected(self):
        """Status should show disconnected when not connected."""
        with patch('prettycli.vscode.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.is_connected = False
            mock_get_client.return_value = mock_client

            from prettycli.vscode.client import get_status
            status, style = get_status()

            assert "未连接" in status
            assert style == "warning"

    def test_status_shows_connected_when_connected(self):
        """Status should show connected info when connected."""
        with patch('prettycli.vscode.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.is_connected = True
            mock_client.current_file = "/path/to/file.py"
            mock_get_client.return_value = mock_client

            from prettycli.vscode.client import get_status
            status, style = get_status()

            assert "file.py" in status
            assert style == "info"

    def test_status_shows_connected_no_file(self):
        """Status should show connected even without active file."""
        with patch('prettycli.vscode.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.is_connected = True
            mock_client.current_file = None
            mock_get_client.return_value = mock_client

            from prettycli.vscode.client import get_status
            status, style = get_status()

            assert "已连接" in status
            assert style == "success"


class TestVSCodeClient:
    """Test VS Code client connection behavior."""

    def test_connect_success(self):
        """Test successful connection."""
        with patch('prettycli.vscode.client.create_connection') as mock_create:
            mock_ws = MagicMock()
            mock_create.return_value = mock_ws

            client = vscode.VSCodeClient()
            result = client.connect()

            assert result is True
            assert client.is_connected is True

    def test_connect_failure(self):
        """Test connection failure."""
        with patch('prettycli.vscode.client.create_connection') as mock_create:
            mock_create.side_effect = Exception("Connection refused")

            client = vscode.VSCodeClient()
            result = client.connect()

            assert result is False
            assert client.is_connected is False

    def test_disconnect(self):
        """Test disconnection."""
        with patch('prettycli.vscode.client.create_connection') as mock_create:
            mock_ws = MagicMock()
            mock_create.return_value = mock_ws

            client = vscode.VSCodeClient()
            client.connect()
            assert client.is_connected is True

            client.disconnect()
            assert client.is_connected is False
            mock_ws.close.assert_called_once()
