from prettycli import BaseCommand, Context, ui, statusbar


class HelloCommand(BaseCommand):
    name = "hello"
    help = "Say hello to someone"

    def __init__(self):
        self.count = 0
        statusbar.register(self.status)

    def status(self):
        if self.count == 0:
            return ("hello: ready", "info")
        return (f"hello: {self.count} calls", "success")

    def run(self, ctx: Context, name: str = "world") -> int:
        self.count += 1
        ui.success(f"Hello, {name}!")
        return 0
