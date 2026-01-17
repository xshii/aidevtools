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
    help = "Say hello"

    def run(self, ctx: Context, name: str = "World") -> int:
        print(f"Hello, {name}!")
        return 0


class CalcCommand(BaseCommand):
    """Simple calculator."""

    name = "calc"
    help = "Calculate expression (e.g., calc 1 + 2)"

    def run(self, ctx: Context, *args: str) -> int:
        if not args:
            print("Usage: calc <number> <op> <number>")
            return 1

        expr = " ".join(args)
        try:
            # Safe evaluation for simple math
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expr):
                result = eval(expr)
                print(f"{expr} = {result}")
                return 0
            else:
                print("Invalid expression")
                return 1
        except Exception as e:
            print(f"Error: {e}")
            return 1


class ExitCommand(BaseCommand):
    """Exit the shell."""

    name = "exit"
    help = "Exit the interactive shell"

    def run(self, ctx: Context) -> int:
        print("Goodbye!")
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
