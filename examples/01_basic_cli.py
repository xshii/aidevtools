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
    help = "Say hello to someone"

    def run(self, ctx: Context, name: str = "World") -> int:
        """
        Execute the hello command.

        Args:
            ctx: The execution context
            name: Name to greet (default: World)

        Returns:
            Exit code (0 for success)
        """
        print(f"Hello, {name}!")
        return 0


class GreetCommand(BaseCommand):
    """Greet multiple people at once."""

    name = "greet"
    help = "Greet multiple people"

    def run(self, ctx: Context, *names: str) -> int:
        """
        Greet multiple people.

        Args:
            ctx: The execution context
            names: Names to greet

        Returns:
            Exit code (0 for success)
        """
        if not names:
            print("Hello, everyone!")
        else:
            for name in names:
                print(f"Hello, {name}!")
        return 0


if __name__ == "__main__":
    app = App("basic-cli")
    app.register(HelloCommand())
    app.register(GreetCommand())
    app.run()
