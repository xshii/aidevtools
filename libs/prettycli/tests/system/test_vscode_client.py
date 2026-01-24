"""
System tests for prettycli vscode client
"""
import pytest
import time

# Check for required dependencies
pytest.importorskip("websockets")
pytest.importorskip("websocket")

from tests.system.mock_server import MockVSCodeServer
from prettycli.vscode import VSCodeClient, get_client, get_status, reset_client


class TestVSCodeClientConnection:
    """Test WebSocket connection"""

    def test_connect_success(self):
        with MockVSCodeServer(port=19971):
            client = VSCodeClient(port=19971)
            assert client.connect() is True
            assert client.is_connected is True
            client.disconnect()

    def test_connect_failure(self):
        client = VSCodeClient(port=19999)  # No server on this port
        assert client.connect() is False
        assert client.is_connected is False

    def test_disconnect(self):
        with MockVSCodeServer(port=19972):
            client = VSCodeClient(port=19972)
            client.connect()
            client.disconnect()
            assert client.is_connected is False

    def test_context_manager(self):
        with MockVSCodeServer(port=19973):
            with VSCodeClient(port=19973) as client:
                assert client.is_connected is True
            assert client.is_connected is False


class TestVSCodeClientArtifacts:
    """Test artifact rendering"""

    @pytest.fixture(autouse=True)
    def setup_server(self):
        self.server = MockVSCodeServer(port=19974)
        self.server.start()
        time.sleep(0.2)
        yield
        self.server.stop()

    def test_show_chart(self):
        client = VSCodeClient(port=19974)
        client.connect()

        panel_id = client.show_chart(
            chart_type="bar",
            labels=["A", "B", "C"],
            datasets=[{"label": "Test", "data": [1, 2, 3]}],
            title="Test Chart",
        )

        assert panel_id != ""
        assert client.current_file is not None
        assert "chart" in client.current_file

        messages = self.server.get_messages()
        assert len(messages) == 1
        assert messages[0]["action"] == "render"
        assert messages[0]["artifact"]["type"] == "chart"

        client.disconnect()

    def test_show_table(self):
        client = VSCodeClient(port=19974)
        client.connect()

        panel_id = client.show_table(
            columns=["Name", "Value"],
            rows=[["A", 1], ["B", 2]],
            title="Test Table",
        )

        assert panel_id != ""

        messages = self.server.get_messages()
        assert messages[-1]["artifact"]["type"] == "table"
        assert messages[-1]["artifact"]["data"]["columns"] == ["Name", "Value"]

        client.disconnect()

    def test_show_diff(self):
        client = VSCodeClient(port=19974)
        client.connect()

        panel_id = client.show_diff(
            original="line1\nline2",
            modified="line1\nline2\nline3",
            original_path="old.txt",
            modified_path="new.txt",
        )

        assert panel_id != ""

        messages = self.server.get_messages()
        assert messages[-1]["artifact"]["type"] == "diff"

        client.disconnect()

    def test_show_markdown(self):
        client = VSCodeClient(port=19974)
        client.connect()

        panel_id = client.show_markdown("# Hello\n**World**")

        assert panel_id != ""

        messages = self.server.get_messages()
        assert messages[-1]["artifact"]["type"] == "markdown"

        client.disconnect()

    def test_show_json(self):
        client = VSCodeClient(port=19974)
        client.connect()

        panel_id = client.show_json({"key": "value", "nested": {"a": 1}})

        assert panel_id != ""

        messages = self.server.get_messages()
        assert messages[-1]["artifact"]["type"] == "json"

        client.disconnect()

    def test_show_web(self):
        client = VSCodeClient(port=19974)
        client.connect()

        panel_id = client.show_web(html="<h1>Hello</h1>")

        assert panel_id != ""

        messages = self.server.get_messages()
        assert messages[-1]["artifact"]["type"] == "web"

        client.disconnect()

    def test_show_file(self, tmp_path):
        # Create temp file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        client = VSCodeClient(port=19974)
        client.connect()

        panel_id = client.show_file(
            path=str(test_file),
            language="python",
        )

        assert panel_id != ""

        messages = self.server.get_messages()
        assert messages[-1]["artifact"]["type"] == "file"
        assert "print" in messages[-1]["artifact"]["data"]["content"]

        client.disconnect()

    def test_show_image(self, tmp_path):
        client = VSCodeClient(port=19974)
        client.connect()

        # Test with raw bytes
        panel_id = client.show_image(
            data=b"\x89PNG\r\n\x1a\n",  # PNG header
            mime_type="image/png",
            alt="Test image",
        )

        assert panel_id != ""

        messages = self.server.get_messages()
        assert messages[-1]["artifact"]["type"] == "image"
        assert "base64" in messages[-1]["artifact"]["data"]["src"]

        client.disconnect()

    def test_show_csv(self, tmp_path):
        # Create temp CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,value\nA,1\nB,2")

        client = VSCodeClient(port=19974)
        client.connect()

        panel_id = client.show_csv(path=str(csv_file))

        assert panel_id != ""

        # CSV is converted to table
        messages = self.server.get_messages()
        assert messages[-1]["artifact"]["type"] == "table"
        assert messages[-1]["artifact"]["data"]["columns"] == ["name", "value"]

        client.disconnect()


class TestVSCodeClientPanelManagement:
    """Test panel management"""

    @pytest.fixture(autouse=True)
    def setup_server(self):
        self.server = MockVSCodeServer(port=19975)
        self.server.start()
        time.sleep(0.2)
        yield
        self.server.stop()

    def test_close_panel(self):
        client = VSCodeClient(port=19975)
        client.connect()

        panel_id = client.show_table(["A"], [[1]])
        result = client.close_panel(panel_id)

        assert result is True

        messages = self.server.get_messages()
        assert messages[-1]["action"] == "close"

        client.disconnect()

    def test_list_panels(self):
        client = VSCodeClient(port=19975)
        client.connect()

        panels = client.list_panels()

        assert isinstance(panels, list)
        assert len(panels) == 2  # Mock returns 2 panels

        client.disconnect()

    def test_reuse_panel(self):
        client = VSCodeClient(port=19975)
        client.connect()

        panel_id = client.show_table(["A"], [[1]])
        # Update same panel
        client.show_table(["B"], [[2]], panel_id=panel_id)

        messages = self.server.get_messages()
        assert len(messages) == 2
        # Second message should have panelId
        assert messages[1].get("panelId") == panel_id

        client.disconnect()


class TestVSCodeStatus:
    """Test status functions"""

    def test_get_status_disconnected(self):
        # Reset global client
        reset_client()

        status = get_status()
        assert status[0] == "VSCode 未连接"
        assert status[1] == "warning"

    def test_get_status_connected(self):
        with MockVSCodeServer(port=19976):
            reset_client()

            client = get_client(port=19976)
            client.connect()
            time.sleep(0.2)  # Wait for connection to be established

            status = get_status()
            assert "已连接" in status[0] or "未连接" not in status[0]

            client.disconnect()

    def test_get_status_with_file(self):
        with MockVSCodeServer(port=19977):
            reset_client()

            client = get_client(port=19977)
            client.connect()
            time.sleep(0.2)  # Wait for connection to be established

            # Show something to get a file path
            client.show_table(["A"], [[1]])

            status = get_status()
            # Should show filename
            assert ".html" in status[0] or "已连接" in status[0]

            client.disconnect()


class TestVSCodeClientErrors:
    """Test error handling"""

    def test_connection_error_on_send(self):
        with MockVSCodeServer(port=19978) as server:
            client = VSCodeClient(port=19978)
            client.connect()

            # Stop server to simulate connection loss
            server.stop()
            time.sleep(0.2)

            with pytest.raises(ConnectionError):
                client.show_table(["A"], [[1]])

    def test_custom_error_response(self):
        with MockVSCodeServer(port=19979) as server:
            server.set_response("render", {"success": False, "error": "Test error"})

            client = VSCodeClient(port=19979)
            client.connect()

            # Should not raise, just return empty panel_id
            panel_id = client.show_table(["A"], [[1]])
            assert panel_id == ""

            client.disconnect()


class TestVSCodeClientAutoReconnect:
    """Test auto-reconnect functionality"""

    def test_ping_success(self):
        with MockVSCodeServer(port=19980):
            client = VSCodeClient(port=19980)
            client.connect()

            assert client.ping() is True

            client.disconnect()

    def test_ping_failure(self):
        client = VSCodeClient(port=19999)  # No server
        assert client.ping() is False

    def test_ensure_connected_already_connected(self):
        with MockVSCodeServer(port=19981):
            client = VSCodeClient(port=19981)
            client.connect()

            # Should return True without reconnecting
            assert client.ensure_connected() is True
            assert client.is_connected is True

            client.disconnect()

    def test_ensure_connected_reconnect(self):
        with MockVSCodeServer(port=19982):
            client = VSCodeClient(port=19982)

            # Not connected initially
            assert client.is_connected is False

            # Should connect
            result = client.ensure_connected()
            assert result is True
            assert client.is_connected is True

            client.disconnect()

    def test_auto_reconnect_on_send(self):
        with MockVSCodeServer(port=19983):
            client = VSCodeClient(port=19983, auto_reconnect=True, max_retries=3)
            client.connect()

            # Simulate connection loss by closing websocket
            client._ws.close()
            client._ws = None
            client._connected = False

            # Should auto-reconnect and succeed
            panel_id = client.show_table(["A"], [[1]])
            assert panel_id != ""
            assert client.is_connected is True

            client.disconnect()

    def test_auto_reconnect_disabled(self):
        with MockVSCodeServer(port=19984) as server:
            client = VSCodeClient(port=19984, auto_reconnect=False)
            client.connect()

            # Stop server
            server.stop()
            time.sleep(0.2)

            # Should fail without retrying
            with pytest.raises(ConnectionError):
                client.show_table(["A"], [[1]])

    def test_auto_reconnect_max_retries(self):
        # No server on this port
        client = VSCodeClient(port=19999, auto_reconnect=True, max_retries=2, retry_delay=0.1)

        # Should fail after max retries
        with pytest.raises(ConnectionError) as excinfo:
            client.show_table(["A"], [[1]])

        # Either "Cannot connect" or "attempts" in error message
        error_msg = str(excinfo.value)
        assert "Cannot connect" in error_msg or "attempts" in error_msg


class TestVSCodeClientOpenFile:
    """Test file opening functionality"""

    def test_open_file(self):
        with MockVSCodeServer(port=19985) as server:
            client = VSCodeClient(port=19985)
            client.connect()

            result = client.open_file("/path/to/file.pdf")

            assert result is True

            messages = server.get_messages()
            assert messages[-1]["action"] == "open"
            assert messages[-1]["artifact"]["path"] == "/path/to/file.pdf"

            client.disconnect()

    def test_show_excel(self):
        with MockVSCodeServer(port=19986) as server:
            client = VSCodeClient(port=19986)
            client.connect()

            result = client.show_excel("/path/to/data.xlsx")

            assert result is True

            messages = server.get_messages()
            assert messages[-1]["action"] == "open"
            assert messages[-1]["artifact"]["path"] == "/path/to/data.xlsx"

            client.disconnect()
