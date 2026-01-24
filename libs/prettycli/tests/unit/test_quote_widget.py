"""测试 QuoteWidget"""

from prettycli.subui.widget.quote import QuoteWidget


class TestQuoteWidget:
    def test_load_quotes(self):
        widget = QuoteWidget()
        # 使用公开的 quotes 属性触发延迟加载
        assert len(widget.quotes) > 0

    def test_current_returns_string(self):
        widget = QuoteWidget()
        quote = widget.current()
        assert isinstance(quote, str)
        assert len(quote) > 0

    def test_next_advances_index(self):
        widget = QuoteWidget()
        widget.current()
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
        # 使用公开的 quotes 属性触发延迟加载
        assert widget.quotes == ["Keep coding!"]

    def test_custom_quotes_file(self, tmp_path):
        # Create custom quotes file
        quotes_file = tmp_path / "quotes.txt"
        quotes_file.write_text("Quote 1\nQuote 2\n")

        widget = QuoteWidget(quotes_file=quotes_file)
        # 使用公开的 quotes 属性触发延迟加载
        assert len(widget.quotes) == 2
        assert "Quote 1" in widget.quotes

    def test_lazy_loading(self, tmp_path):
        # Test that quotes are not loaded until accessed
        quotes_file = tmp_path / "quotes.txt"
        quotes_file.write_text("Lazy Quote\n")

        widget = QuoteWidget(quotes_file=quotes_file)
        # Before accessing, _quotes should be None
        assert widget._quotes is None
        # After accessing quotes property, it should be loaded
        _ = widget.quotes
        assert widget._quotes is not None
        assert "Lazy Quote" in widget._quotes
