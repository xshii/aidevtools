
from prettycli import BaseCommand, Context


class TestBaseCommand:
    def setup_method(self):
        BaseCommand.clear()

    def test_register_command(self):
        class HelloCommand(BaseCommand):
            name = "hello"
            help = "Say hello"

            def run(self, ctx: Context, **kwargs) -> int:
                return 0

        assert "hello" in BaseCommand.all()
        assert BaseCommand.get("hello") == HelloCommand

    def test_no_register_without_name(self):
        class NoNameCommand(BaseCommand):
            help = "No name"

            def run(self, ctx: Context, **kwargs) -> int:
                return 0

        assert len(BaseCommand.all()) == 0

    def test_run_command(self):
        result = []

        class TestCommand(BaseCommand):
            name = "test"
            help = "Test command"

            def run(self, ctx: Context, msg: str = "default") -> int:
                result.append(msg)
                return 0

        cmd = TestCommand()
        ctx = Context()
        cmd.run(ctx, msg="hello")

        assert result == ["hello"]

    def test_clear_registry(self):
        class Cmd1(BaseCommand):
            name = "cmd1"

            def run(self, ctx: Context, **kwargs) -> int:
                return 0

        assert len(BaseCommand.all()) == 1
        BaseCommand.clear()
        assert len(BaseCommand.all()) == 0
