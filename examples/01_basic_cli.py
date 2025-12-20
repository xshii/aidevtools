#!/usr/bin/env python3
"""
Basic CLI Example - Shows how to create a simple command-line application.

Usage:
    python 01_basic_cli.py hello
    python 01_basic_cli.py hello --name Alice
    python 01_basic_cli.py greet Alice Bob Charlie
"""
from prettycli import App, BaseCommand, Context


class HelloCommand(BaseCommand):
    """A simple hello world command."""

    name = "hello"
    description = "Say hello to someone"

    def execute(self, ctx: Context, name: str = "World"):
        """
        Execute the hello command.

        Args:
            ctx: The execution context
            name: Name to greet (default: World)
        """
        ctx.print(f"Hello, {name}!")


class GreetCommand(BaseCommand):
    """Greet multiple people at once."""

    name = "greet"
    description = "Greet multiple people"

    def execute(self, ctx: Context, *names: str):
        """
        Greet multiple people.

        Args:
            ctx: The execution context
            names: Names to greet
        """
        if not names:
            ctx.print("Hello, everyone!")
        else:
            for name in names:
                ctx.print(f"Hello, {name}!")


if __name__ == "__main__":
    app = App("basic-cli")
    app.register(HelloCommand())
    app.register(GreetCommand())
    app.run()
