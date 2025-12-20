#!/usr/bin/env python3
"""
Interactive CLI Example - Shows how to create an interactive shell.

Usage:
    python 02_interactive_cli.py

Then type commands at the prompt:
    > hello
    > hello --name Alice
    > calc 1 + 2
    > exit
"""
from prettycli import CLI, BaseCommand, Context


class HelloCommand(BaseCommand):
    """Say hello."""

    name = "hello"
    description = "Say hello"

    def execute(self, ctx: Context, name: str = "World"):
        ctx.print(f"Hello, {name}!")


class CalcCommand(BaseCommand):
    """Simple calculator."""

    name = "calc"
    description = "Calculate expression (e.g., calc 1 + 2)"

    def execute(self, ctx: Context, *args: str):
        if not args:
            ctx.print("Usage: calc <number> <op> <number>")
            return

        expr = " ".join(args)
        try:
            # Safe evaluation for simple math
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expr):
                result = eval(expr)
                ctx.print(f"{expr} = {result}")
            else:
                ctx.print("Invalid expression")
        except Exception as e:
            ctx.print(f"Error: {e}")


class ExitCommand(BaseCommand):
    """Exit the shell."""

    name = "exit"
    description = "Exit the interactive shell"

    def execute(self, ctx: Context):
        ctx.print("Goodbye!")
        raise SystemExit(0)


if __name__ == "__main__":
    cli = CLI("interactive-cli")
    cli.register(HelloCommand())
    cli.register(CalcCommand())
    cli.register(ExitCommand())

    print("Interactive CLI Demo")
    print("Type 'help' for available commands, 'exit' to quit")
    print()

    cli.run()
