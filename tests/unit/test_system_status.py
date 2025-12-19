from prettycli.subui import SystemStatus


class TestSystemStatus:
    def test_load_quotes(self):
        ss = SystemStatus()
        assert len(ss._quotes) > 0

    def test_next_quote_returns_string(self):
        ss = SystemStatus()
        quote = ss.next_quote()
        assert isinstance(quote, str)
        assert len(quote) > 0

    def test_next_quote_cycles(self):
        ss = SystemStatus()
        first = ss.next_quote()
        # 循环一轮
        for _ in range(len(ss._quotes) - 1):
            ss.next_quote()
        # 回到第一条
        cycled = ss.next_quote()
        assert cycled == first

    def test_index_increments(self):
        ss = SystemStatus()
        assert ss._index == 0
        ss.next_quote()
        assert ss._index == 1
        ss.next_quote()
        assert ss._index == 2

    def test_show(self):
        from unittest.mock import patch
        import prettycli.ui as ui

        ss = SystemStatus()

        with patch.object(ui, 'print') as mock:
            ss.show()
            mock.assert_called_once()

    def test_fallback_quote(self, tmp_path, monkeypatch):
        # Test when quotes file doesn't exist
        import prettycli.subui.system_status as ss_module

        monkeypatch.setattr(ss_module, 'QUOTES_FILE', tmp_path / "missing.txt")

        ss = ss_module.SystemStatus()
        assert ss._quotes == ["Keep coding!"]
