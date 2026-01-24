from pathlib import Path

from prettycli import Context


class TestContext:
    def test_default_values(self):
        ctx = Context()

        assert ctx.cwd == Path.cwd()
        assert ctx.verbose is False
        assert ctx.config == {}

    def test_get_set_config(self):
        ctx = Context()

        ctx.set_config("key", "value")
        assert ctx.get_config("key") == "value"
        assert ctx.get_config("missing") is None
        assert ctx.get_config("missing", "default") == "default"
