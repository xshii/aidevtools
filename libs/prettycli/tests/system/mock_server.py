"""
Mock WebSocket server for testing prettycli vscode client
"""
import json
import threading
import time
from typing import Dict, List
import asyncio

try:
    import websockets
    from websockets.server import serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class MockVSCodeServer:
    """Mock VS Code extension WebSocket server for testing"""

    def __init__(self, port: int = 19961):
        self.port = port
        self.messages: List[Dict] = []
        self.responses: Dict[str, Dict] = {}
        self.server = None
        self._loop = None
        self._thread = None
        self._running = False
        self._session_path = "/tmp/prettycli/test_session"
        self._file_counter = 0

    def set_response(self, action: str, response: Dict):
        """Set custom response for an action"""
        self.responses[action] = response

    def _generate_file_path(self, artifact_type: str) -> str:
        """Generate a mock file path"""
        self._file_counter += 1
        return f"{self._session_path}/{self._file_counter:03d}_{artifact_type}.html"

    async def _handle_message(self, websocket):
        """Handle incoming WebSocket messages"""
        async for message in websocket:
            data = json.loads(message)
            self.messages.append(data)

            msg_id = data.get("id", "unknown")
            action = data.get("action", "")

            # Check for custom response
            if action in self.responses:
                response = {"id": msg_id, **self.responses[action]}
            else:
                # Default responses
                if action == "render":
                    artifact = data.get("artifact", {})
                    file_path = self._generate_file_path(artifact.get("type", "unknown"))
                    response = {
                        "id": msg_id,
                        "success": True,
                        "data": {
                            "panelId": f"panel-{msg_id[:8]}",
                            "filePath": file_path,
                        },
                    }
                elif action == "update":
                    response = {
                        "id": msg_id,
                        "success": True,
                        "data": {
                            "filePath": self._generate_file_path("update"),
                        },
                    }
                elif action == "close":
                    response = {
                        "id": msg_id,
                        "success": True,
                    }
                elif action == "list":
                    response = {
                        "id": msg_id,
                        "success": True,
                        "data": {
                            "panels": ["panel-1", "panel-2"],
                        },
                    }
                elif action == "ping":
                    response = {
                        "id": msg_id,
                        "success": True,
                    }
                elif action == "open":
                    response = {
                        "id": msg_id,
                        "success": True,
                    }
                else:
                    response = {
                        "id": msg_id,
                        "success": False,
                        "error": f"Unknown action: {action}",
                    }

            await websocket.send(json.dumps(response))

    async def _run_server(self):
        """Run the WebSocket server"""
        async with serve(self._handle_message, "localhost", self.port):
            self._running = True
            while self._running:
                await asyncio.sleep(0.1)

    def _run_in_thread(self):
        """Run asyncio loop in thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_server())
        except Exception:
            pass
        finally:
            self._loop.close()

    def start(self):
        """Start the mock server in a background thread"""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets is required: pip install websockets")

        self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
        self._thread.start()

        # Wait for server to start
        for _ in range(50):
            if self._running:
                break
            time.sleep(0.1)

    def stop(self):
        """Stop the mock server"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_messages(self) -> List[Dict]:
        """Get all received messages"""
        return self.messages.copy()

    def clear_messages(self):
        """Clear received messages"""
        self.messages.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
