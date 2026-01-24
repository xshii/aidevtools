"""
VS Code extension API client for prettycli
"""
import json
import threading
import uuid
import base64
from pathlib import Path
from typing import Optional, Any, Dict, List

try:
    from websocket import create_connection, WebSocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


class VSCodeClient:
    """
    Client for communicating with prettycli VS Code extension.

    Features:
        - Auto-reconnect on connection loss
        - Connection health checking
        - Session and artifact file tracking

    Example:
        >>> client = VSCodeClient()
        >>> client.show_chart("bar", ["A", "B"], [{"label": "Data", "data": [1, 2]}])

        # Or use as context manager:
        >>> with VSCodeClient() as client:
        ...     client.show_table(["Col1"], [["Row1"]])
    """

    def __init__(
        self,
        port: int = 19960,
        auto_reconnect: bool = True,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        max_retry_delay: float = 8.0,
    ):
        """
        Initialize VS Code client.

        Args:
            port: WebSocket server port (default: 19960)
            auto_reconnect: Enable auto-reconnect on connection loss
            max_retries: Maximum reconnection attempts
            retry_delay: Initial delay between retries in seconds (exponential backoff)
            max_retry_delay: Maximum delay between retries in seconds
        """
        self.port = port
        self.auto_reconnect = auto_reconnect
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self._ws: Optional[Any] = None
        self._connected: bool = False
        self._current_file: Optional[str] = None
        self._session_path: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to VS Code"""
        return self._connected and self._ws is not None

    @property
    def current_file(self) -> Optional[str]:
        """Get current artifact file path"""
        return self._current_file

    @property
    def session_path(self) -> Optional[str]:
        """Get session folder path"""
        return self._session_path

    def connect(self) -> bool:
        """Connect to VS Code extension"""
        if not HAS_WEBSOCKET:
            raise ImportError("websocket-client is required: pip install websocket-client")

        try:
            self._ws = create_connection(f"ws://localhost:{self.port}", timeout=5)
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from VS Code extension"""
        if self._ws:
            self._ws.close()
            self._ws = None
        self._connected = False

    def _get_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.retry_delay * (2 ** attempt)
        return min(delay, self.max_retry_delay)

    def _send(self, action: str, artifact: Optional[Dict] = None, panel_id: Optional[str] = None) -> Dict:
        """
        Send message and wait for response with auto-reconnect.

        Uses exponential backoff for retries (delay doubles each attempt,
        capped at max_retry_delay).

        Args:
            action: Action type ('render', 'close', 'list', 'ping')
            artifact: Artifact data to render
            panel_id: Target panel ID

        Returns:
            Response dictionary from VS Code extension

        Raises:
            ConnectionError: If connection fails after retries
        """
        import time

        last_error: Optional[Exception] = None
        retries = self.max_retries if self.auto_reconnect else 1

        for attempt in range(retries):
            # Ensure connection
            if not self._ws or not self._connected:
                if not self.connect():
                    if attempt < retries - 1:
                        time.sleep(self._get_retry_delay(attempt))
                        continue
                    raise ConnectionError("Cannot connect to VS Code extension")

            msg_id = str(uuid.uuid4())
            message = {
                "id": msg_id,
                "action": action,
            }
            if artifact:
                message["artifact"] = artifact
            if panel_id:
                message["panelId"] = panel_id

            try:
                self._ws.send(json.dumps(message))
                response = json.loads(self._ws.recv())

                # Update current file path from response
                if response.get("success") and response.get("data"):
                    data = response["data"]
                    if "filePath" in data:
                        self._current_file = data["filePath"]
                        # Extract session path from file path
                        if self._current_file:
                            import os
                            self._session_path = os.path.dirname(self._current_file)

                return response

            except Exception as e:
                last_error = e
                self._connected = False
                if self._ws:
                    try:
                        self._ws.close()
                    except Exception:
                        pass
                self._ws = None

                # Retry if auto-reconnect enabled
                if self.auto_reconnect and attempt < retries - 1:
                    time.sleep(self._get_retry_delay(attempt))
                    continue

        raise ConnectionError(f"Lost connection to VS Code after {retries} attempts: {last_error}")

    def ping(self) -> bool:
        """
        Check if connection is alive.

        Returns:
            True if connected and responsive, False otherwise
        """
        try:
            response = self._send("ping")
            return response.get("success", False)
        except Exception:
            return False

    def ensure_connected(self) -> bool:
        """
        Ensure connection is established, reconnecting if necessary.

        Returns:
            True if connected, False otherwise
        """
        if self.is_connected and self.ping():
            return True
        return self.connect()

    # ============ Artifact Methods ============

    def show_chart(
        self,
        chart_type: str,
        labels: List[str],
        datasets: List[Dict[str, Any]],
        title: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> str:
        """
        Show a chart artifact

        Args:
            chart_type: 'bar', 'line', or 'pie'
            labels: X-axis labels
            datasets: List of {label: str, data: List[float], color?: str}
            title: Chart title
            panel_id: Reuse existing panel

        Returns:
            Panel ID
        """
        artifact = {
            "type": "chart",
            "title": title,
            "data": {
                "chartType": chart_type,
                "labels": labels,
                "datasets": datasets,
            },
        }
        response = self._send("render", artifact, panel_id)
        return response.get("data", {}).get("panelId", "")

    def show_table(
        self,
        columns: List[str],
        rows: List[List[Any]],
        title: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> str:
        """
        Show a table artifact

        Args:
            columns: Column headers
            rows: Table rows
            title: Table title
            panel_id: Reuse existing panel

        Returns:
            Panel ID
        """
        artifact = {
            "type": "table",
            "title": title,
            "data": {
                "columns": columns,
                "rows": rows,
            },
        }
        response = self._send("render", artifact, panel_id)
        return response.get("data", {}).get("panelId", "")

    def show_file(
        self,
        path: str,
        content: Optional[str] = None,
        language: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        title: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> str:
        """
        Show a file artifact

        Args:
            path: File path
            content: File content (reads from path if not provided)
            language: Syntax highlighting language
            start_line: Start line number
            end_line: End line number
            title: Panel title
            panel_id: Reuse existing panel

        Returns:
            Panel ID
        """
        if content is None:
            content = Path(path).read_text()

        artifact = {
            "type": "file",
            "title": title or path,
            "data": {
                "path": path,
                "content": content,
                "language": language,
                "startLine": start_line,
                "endLine": end_line,
            },
        }
        response = self._send("render", artifact, panel_id)
        return response.get("data", {}).get("panelId", "")

    def show_diff(
        self,
        original: str,
        modified: str,
        original_path: Optional[str] = None,
        modified_path: Optional[str] = None,
        language: Optional[str] = None,
        title: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> str:
        """
        Show a diff artifact

        Args:
            original: Original content
            modified: Modified content
            original_path: Original file path label
            modified_path: Modified file path label
            language: Syntax highlighting language
            title: Panel title
            panel_id: Reuse existing panel

        Returns:
            Panel ID
        """
        artifact = {
            "type": "diff",
            "title": title or "Diff",
            "data": {
                "original": original,
                "modified": modified,
                "originalPath": original_path,
                "modifiedPath": modified_path,
                "language": language,
            },
        }
        response = self._send("render", artifact, panel_id)
        return response.get("data", {}).get("panelId", "")

    def show_image(
        self,
        path: Optional[str] = None,
        data: Optional[bytes] = None,
        mime_type: str = "image/png",
        alt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> str:
        """
        Show an image artifact

        Args:
            path: Image file path
            data: Image binary data
            mime_type: Image MIME type
            alt: Alt text
            width: Display width
            height: Display height
            title: Panel title
            panel_id: Reuse existing panel

        Returns:
            Panel ID
        """
        if data is None and path:
            data = Path(path).read_bytes()

        if data:
            b64 = base64.b64encode(data).decode()
            src = f"data:{mime_type};base64,{b64}"
        else:
            src = path or ""

        artifact = {
            "type": "image",
            "title": title,
            "data": {
                "src": src,
                "alt": alt,
                "width": width,
                "height": height,
            },
        }
        response = self._send("render", artifact, panel_id)
        return response.get("data", {}).get("panelId", "")

    def show_markdown(
        self,
        content: str,
        title: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> str:
        """
        Show a markdown artifact

        Args:
            content: Markdown content
            title: Panel title
            panel_id: Reuse existing panel

        Returns:
            Panel ID
        """
        artifact = {
            "type": "markdown",
            "title": title,
            "data": {
                "content": content,
            },
        }
        response = self._send("render", artifact, panel_id)
        return response.get("data", {}).get("panelId", "")

    def show_json(
        self,
        content: Any,
        collapsed: bool = False,
        title: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> str:
        """
        Show a JSON artifact

        Args:
            content: JSON-serializable content
            collapsed: Start collapsed
            title: Panel title
            panel_id: Reuse existing panel

        Returns:
            Panel ID
        """
        artifact = {
            "type": "json",
            "title": title,
            "data": {
                "content": content,
                "collapsed": collapsed,
            },
        }
        response = self._send("render", artifact, panel_id)
        return response.get("data", {}).get("panelId", "")

    def show_web(
        self,
        html: Optional[str] = None,
        url: Optional[str] = None,
        title: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> str:
        """
        Show a web artifact

        Args:
            html: HTML content
            url: URL to load
            title: Panel title
            panel_id: Reuse existing panel

        Returns:
            Panel ID
        """
        artifact = {
            "type": "web",
            "title": title,
            "data": {
                "html": html,
                "url": url,
            },
        }
        response = self._send("render", artifact, panel_id)
        return response.get("data", {}).get("panelId", "")

    def show_csv(
        self,
        path: Optional[str] = None,
        content: Optional[str] = None,
        title: Optional[str] = None,
        panel_id: Optional[str] = None,
    ) -> str:
        """
        Show a CSV as a table

        Args:
            path: CSV file path
            content: CSV content string
            title: Panel title
            panel_id: Reuse existing panel

        Returns:
            Panel ID
        """
        import csv
        from io import StringIO

        if content is None and path:
            content = Path(path).read_text()

        reader = csv.reader(StringIO(content or ""))
        rows = list(reader)

        if not rows:
            return self.show_table([], [], title, panel_id)

        columns = rows[0]
        data_rows = rows[1:]

        return self.show_table(columns, data_rows, title or path, panel_id)

    def open_file(self, path: str) -> bool:
        """
        Open a file with the system default application.

        Args:
            path: File path to open

        Returns:
            True if successful
        """
        response = self._send("open", artifact={"path": path})
        return response.get("success", False)

    def show_excel(self, path: str) -> bool:
        """
        Open an Excel file with the system default application.

        Args:
            path: Excel file path (.xlsx, .xls)

        Returns:
            True if successful
        """
        return self.open_file(path)

    # ============ Panel Management ============

    def close_panel(self, panel_id: str) -> bool:
        """Close a panel"""
        response = self._send("close", panel_id=panel_id)
        return response.get("success", False)

    def list_panels(self) -> List[str]:
        """List all open panels"""
        response = self._send("list")
        return response.get("data", {}).get("panels", [])

    # ============ Context Manager ============

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# Global client instance with thread-safe initialization
_client: Optional[VSCodeClient] = None
_client_lock = threading.Lock()


def get_client(port: int = 19960) -> VSCodeClient:
    """
    Get or create global VS Code client (thread-safe).

    Note: port parameter is only used on first call.
    Subsequent calls return the existing client regardless of port.

    Args:
        port: WebSocket server port (only used on first initialization)

    Returns:
        Global VSCodeClient instance
    """
    global _client
    if _client is None:
        with _client_lock:
            # Double-check locking pattern
            if _client is None:
                _client = VSCodeClient(port)
    return _client


def reset_client():
    """
    Reset the global client (useful for testing or reconnecting to different port).

    Thread-safe operation that disconnects existing client before clearing.
    """
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.disconnect()
            except Exception:
                pass
            _client = None


def get_status() -> tuple:
    """
    Get VS Code connection status for statusbar.

    Returns:
        Tuple of (message, style) for statusbar display
    """
    client = get_client()

    if not client.is_connected:
        return ("VSCode 未连接", "warning")

    if client.current_file:
        # Show just the filename, not full path
        import os
        filename = os.path.basename(client.current_file)
        return (f"📄 {filename}", "info")

    return ("VSCode 已连接", "success")


# Convenience functions
def show_chart(*args, **kwargs) -> str:
    return get_client().show_chart(*args, **kwargs)


def show_table(*args, **kwargs) -> str:
    return get_client().show_table(*args, **kwargs)


def show_file(*args, **kwargs) -> str:
    return get_client().show_file(*args, **kwargs)


def show_diff(*args, **kwargs) -> str:
    return get_client().show_diff(*args, **kwargs)


def show_image(*args, **kwargs) -> str:
    return get_client().show_image(*args, **kwargs)


def show_markdown(*args, **kwargs) -> str:
    return get_client().show_markdown(*args, **kwargs)


def show_json(*args, **kwargs) -> str:
    return get_client().show_json(*args, **kwargs)


def show_web(*args, **kwargs) -> str:
    return get_client().show_web(*args, **kwargs)


def show_csv(*args, **kwargs) -> str:
    return get_client().show_csv(*args, **kwargs)


def show_excel(path: str) -> bool:
    """Open an Excel file with the system default application."""
    return get_client().show_excel(path)


def open_file(path: str) -> bool:
    """Open a file with the system default application."""
    return get_client().open_file(path)


def is_extension_installed() -> bool:
    """Check if VS Code extension is installed."""
    # TODO: implement actual check
    return True


def install_extension() -> bool:
    """Install VS Code extension."""
    # TODO: implement actual installation
    return True
