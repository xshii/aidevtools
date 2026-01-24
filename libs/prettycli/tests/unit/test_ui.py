from unittest.mock import patch

from rich.table import Table

from prettycli import ui


class TestUIOutput:
    def test_info(self, capsys):
        with patch.object(ui.console, 'print') as mock:
            ui.info("test message")
            mock.assert_called_once()
            assert "test message" in str(mock.call_args)

    def test_success(self):
        with patch.object(ui.console, 'print') as mock:
            ui.success("done")
            mock.assert_called_once()
            assert "done" in str(mock.call_args)

    def test_error(self):
        with patch.object(ui.console, 'print') as mock:
            ui.error("failed")
            mock.assert_called_once()
            assert "failed" in str(mock.call_args)

    def test_warn(self):
        with patch.object(ui.console, 'print') as mock:
            ui.warn("warning")
            mock.assert_called_once()
            assert "warning" in str(mock.call_args)

    def test_print(self):
        with patch.object(ui.console, 'print') as mock:
            ui.print("hello")
            mock.assert_called_once_with("hello")

    def test_print_empty(self):
        with patch.object(ui.console, 'print') as mock:
            ui.print()
            mock.assert_called_once_with("")

    def test_panel(self):
        with patch.object(ui.console, 'print') as mock:
            ui.panel("content", title="Title")
            mock.assert_called_once()


class TestUITable:
    def test_table_basic(self):
        t = ui.table()
        assert isinstance(t, Table)

    def test_table_with_title(self):
        t = ui.table(title="Test")
        assert t.title == "Test"

    def test_table_with_columns(self):
        t = ui.table(columns=["A", "B", "C"])
        assert len(t.columns) == 3

    def test_print_table(self):
        t = ui.table(columns=["Name"])
        with patch.object(ui.console, 'print') as mock:
            ui.print_table(t)
            mock.assert_called_once_with(t)


class TestUIProgress:
    def test_spinner(self):
        s = ui.spinner("Loading...")
        assert s is not None

    def test_progress(self):
        p = ui.progress()
        assert p is not None


class TestUIPrompts:
    def test_select(self):
        with patch('prettycli.ui.inquirer') as mock_inq:
            mock_inq.select.return_value.execute.return_value = "choice1"
            result = ui.select("Pick one", ["choice1", "choice2"])
            assert result == "choice1"

    def test_confirm(self):
        with patch('prettycli.ui.inquirer') as mock_inq:
            mock_inq.confirm.return_value.execute.return_value = True
            result = ui.confirm("Sure?")
            assert result is True

    def test_text(self):
        with patch('prettycli.ui.inquirer') as mock_inq:
            mock_inq.text.return_value.execute.return_value = "input"
            result = ui.text("Enter:")
            assert result == "input"

    def test_password(self):
        with patch('prettycli.ui.inquirer') as mock_inq:
            mock_inq.secret.return_value.execute.return_value = "secret"
            result = ui.password("Password:")
            assert result == "secret"

    def test_checkbox(self):
        with patch('prettycli.ui.inquirer') as mock_inq:
            mock_inq.checkbox.return_value.execute.return_value = ["a", "b"]
            result = ui.checkbox("Select:", ["a", "b", "c"])
            assert result == ["a", "b"]

    def test_fuzzy(self):
        with patch('prettycli.ui.inquirer') as mock_inq:
            mock_inq.fuzzy.return_value.execute.return_value = "match"
            result = ui.fuzzy("Search:", ["match", "other"])
            assert result == "match"
