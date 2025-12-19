import pytest

from prettycli.subui import StatusBar


class TestStatusBar:
    def test_register_provider(self):
        bar = StatusBar()
        bar.register(lambda: "test")

        content = bar.render()

        assert "test" in content.plain

    def test_multiple_providers(self):
        bar = StatusBar()
        bar.register(lambda: "foo")
        bar.register(lambda: "bar")

        content = bar.render()

        assert "foo" in content.plain
        assert "bar" in content.plain
        assert "|" in content.plain

    def test_empty_provider_skipped(self):
        bar = StatusBar()
        bar.register(lambda: "")
        bar.register(lambda: "valid")

        content = bar.render()

        assert "valid" in content.plain

    def test_exception_in_provider_skipped(self):
        bar = StatusBar()

        def bad_provider():
            raise Exception("error")

        bar.register(bad_provider)
        bar.register(lambda: "good")

        content = bar.render()

        assert "good" in content.plain

    def test_styled_provider(self):
        bar = StatusBar()
        bar.register(lambda: ("error msg", "error"))

        content = bar.render()
        assert "error msg" in content.plain

    def test_custom_style(self):
        bar = StatusBar()
        bar.register(lambda: ("custom", "bold red"))

        content = bar.render()
        assert "custom" in content.plain

    def test_none_provider_skipped(self):
        bar = StatusBar()
        bar.register(lambda: None)
        bar.register(lambda: "valid")

        content = bar.render()
        assert "valid" in content.plain

    def test_clear(self):
        bar = StatusBar()
        bar.register(lambda: "test")
        bar.clear()

        content = bar.render()
        assert content.plain == ""

    def test_show(self):
        from unittest.mock import patch

        bar = StatusBar()
        bar.register(lambda: "status")

        with patch.object(bar._console, 'print') as mock:
            bar.show()
            mock.assert_called_once()
