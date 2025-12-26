"""测试 QuoteWidget"""
from pathlib import Path

from prettycli.subui.widget.quote import QuoteWidget


class TestQuoteWidget:
    def test_load_quotes(self):
        widget = QuoteWidget()
        assert len(widget._quotes) > 0

    def test_current_returns_string(self):
        widget = QuoteWidget()
        quote = widget.current()
        assert isinstance(quote, str)
        assert len(quote) > 0

    def test_next_advances_index(self):
        widget = QuoteWidget()
        first = widget.current()
        widget.next()
        # index should advance
        assert widget._index == 1

    def test_next_returns_quote(self):
        widget = QuoteWidget()
        quote = widget.next()
        assert isinstance(quote, str)

    def test_render(self):
        widget = QuoteWidget()
        rendered = widget.render()
        assert "[dim italic]" in rendered
        assert "[/]" in rendered

    def test_callable(self):
        widget = QuoteWidget()
        result = widget()
        assert isinstance(result, str)

    def test_fallback_quote(self, tmp_path):
        # Test when quotes file doesn't exist
        widget = QuoteWidget(quotes_file=tmp_path / "missing.txt")
        assert widget._quotes == ["Keep coding!"]

    def test_custom_quotes_file(self, tmp_path):
        # Create custom quotes file
        quotes_file = tmp_path / "quotes.txt"
        quotes_file.write_text("Quote 1\nQuote 2\n")

        widget = QuoteWidget(quotes_file=quotes_file)
        assert len(widget._quotes) == 2
        assert "Quote 1" in widget._quotes
