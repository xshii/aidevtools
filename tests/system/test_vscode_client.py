"""
System tests for prettycli vscode client
"""
import pytest
import time

# Check for required dependencies
pytest.importorskip("websockets")
pytest.importorskip("websocket")

from tests.system.mock_server import MockVSCodeServer
from prettycli.vscode import VSCodeClient, get_client, get_status


class TestVSCodeClientConnection:
    """Test WebSocket connection"""

    def test_connect_success(self):
        with MockVSCodeServer(port=19971) as server:
            client = VSCodeClient(port=19971)
            assert client.connect() is True
            assert client.is_connected is True
            client.disconnect()

    def test_connect_failure(self):
        client = VSCodeClient(port=19999)  # No server on this port
        assert client.connect() is False
        assert client.is_connected is False

    def test_disconnect(self):
        with MockVSCodeServer(port=19972) as server:
            client = VSCodeClient(port=19972)
            client.connect()
            client.disconnect()
            assert client.is_connected is False

    def test_context_manager(self):
        with MockVSCodeServer(port=19973) as server:
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
        import prettycli.vscode as vscode_module
        vscode_module._client = None

        status = get_status()
        assert status[0] == "VSCode 未连接"
        assert status[1] == "warning"

    def test_get_status_connected(self):
        with MockVSCodeServer(port=19976) as server:
            import prettycli.vscode as vscode_module
            vscode_module._client = None

            client = get_client(port=19976)
            client.connect()

            status = get_status()
            assert "已连接" in status[0] or "未连接" not in status[0]

            client.disconnect()

    def test_get_status_with_file(self):
        with MockVSCodeServer(port=19977) as server:
            import prettycli.vscode as vscode_module
            vscode_module._client = None

            client = get_client(port=19977)
            client.connect()

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
