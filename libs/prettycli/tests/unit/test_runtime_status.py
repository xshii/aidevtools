import time

from prettycli.subui import RuntimeStatus


class TestRuntimeStatus:
    def test_start_stop(self):
        rs = RuntimeStatus()
        rs.start("test")
        time.sleep(0.05)
        rs.stop()

        assert rs.duration > 0
        assert not rs.is_running

    def test_is_running(self):
        rs = RuntimeStatus()
        assert not rs.is_running

        rs.start("cmd")
        assert rs.is_running

        rs.stop()
        assert not rs.is_running

    def test_format_duration_ms(self):
        rs = RuntimeStatus()
        assert "ms" in rs._format_duration(0.5)

    def test_format_duration_seconds(self):
        rs = RuntimeStatus()
        assert "s" in rs._format_duration(5.0)

    def test_format_duration_minutes(self):
        rs = RuntimeStatus()
        result = rs._format_duration(125)
        assert "m" in result

    def test_render_not_running(self):
        rs = RuntimeStatus()
        text = rs._render()
        assert text.plain == ""

    def test_render_with_duration(self):
        rs = RuntimeStatus()
        rs._duration = 1.5
        text = rs._render()
        assert "took" in text.plain

    def test_render_running(self):
        rs = RuntimeStatus()
        rs.start("test")
        text = rs._render()
        assert "running" in text.plain
        rs.stop()

    def test_show(self):
        from unittest.mock import patch

        rs = RuntimeStatus()
        rs.start("cmd")
        rs.stop()

        with patch.object(rs._console, 'print') as mock:
            rs.show()
            mock.assert_called_once()

    def test_show_no_duration(self):
        from unittest.mock import patch

        rs = RuntimeStatus()

        with patch.object(rs._console, 'print') as mock:
            rs.show()
            mock.assert_not_called()

    def test_duration_property(self):
        rs = RuntimeStatus()
        assert rs.duration == 0
        rs._duration = 5.0
        assert rs.duration == 5.0
