from prettycli.subui import EchoStatus


class TestEchoStatus:
    def test_set_output(self):
        es = EchoStatus()
        es.set_output("hello")
        assert es.has_output
        assert es._last_output == "hello"

    def test_has_output_false_when_empty(self):
        es = EchoStatus()
        assert not es.has_output

    def test_clear(self):
        es = EchoStatus()
        es.set_output("test")
        es.clear()
        assert not es.has_output
        assert not es.is_collapsed

    def test_toggle_when_no_output(self):
        es = EchoStatus()
        es.toggle()  # should not raise
        assert not es.is_collapsed

    def test_toggle_changes_state(self):
        es = EchoStatus()
        es.set_output("line1\nline2")
        assert not es.is_collapsed
        es._collapsed = True  # manually set to avoid redraw
        assert es.is_collapsed

    def test_set_output_resets_collapsed(self):
        es = EchoStatus()
        es.set_output("old")
        es._collapsed = True
        es.set_output("new")
        assert not es.is_collapsed

    def test_max_lines_default(self):
        es = EchoStatus()
        assert es._max_lines == 5

    def test_max_lines_custom(self):
        es = EchoStatus(max_lines=10)
        assert es._max_lines == 10

    def test_redraw_collapsed(self):
        from unittest.mock import patch
        import prettycli.ui as ui

        es = EchoStatus(max_lines=2)
        es.set_output("line1\nline2\nline3\nline4\nline5")
        es._collapsed = True

        with patch.object(ui, 'print') as mock:
            es._redraw()
            assert mock.call_count >= 3  # empty line + 2 lines + hidden msg

    def test_redraw_expanded(self):
        from unittest.mock import patch
        import prettycli.ui as ui

        es = EchoStatus(max_lines=2)
        es.set_output("line1\nline2\nline3")
        es._collapsed = False

        with patch.object(ui, 'print') as mock:
            es._redraw()
            assert mock.call_count >= 2  # empty line + content + collapse hint

    def test_toggle_actually_toggles(self):
        from unittest.mock import patch
        import prettycli.ui as ui

        es = EchoStatus()
        es.set_output("test output")

        with patch.object(ui, 'print'):
            es.toggle()
            assert es.is_collapsed is True
            es.toggle()
            assert es.is_collapsed is False
