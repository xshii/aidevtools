from prettycli import BaseCommand, Context, ui, statusbar


class GreetCommand(BaseCommand):
    name = "greet"
    help = "Greet someone with options"

    def __init__(self):
        self.last_name = None
        statusbar.register(self.status)

    def status(self) -> str:
        if self.last_name:
            return f"greet: {self.last_name}"
        return "greet: idle"

    def run(self, ctx: Context, name: str = "world", loud: bool = False) -> int:
        self.last_name = name
        msg = f"Hello, {name}!"
        if loud:
            msg = msg.upper()

        ui.success(msg)
        return 0
