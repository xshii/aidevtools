#!/usr/bin/env python3
"""
UI Components Example - Shows various UI elements available in prettycli.

Usage:
    python 03_ui_components.py
"""
from prettycli import ui
import time


def demo_progress():
    """Demonstrate progress bar."""
    print("\n=== Progress Bar ===")

    items = list(range(20))
    for item in ui.progress(items, description="Processing"):
        time.sleep(0.1)  # Simulate work


def demo_spinner():
    """Demonstrate spinner."""
    print("\n=== Spinner ===")

    with ui.spinner("Loading data..."):
        time.sleep(2)  # Simulate work
    print("Done!")


def demo_table():
    """Demonstrate table display."""
    print("\n=== Table ===")

    headers = ["Name", "Age", "City"]
    rows = [
        ["Alice", "30", "New York"],
        ["Bob", "25", "Los Angeles"],
        ["Charlie", "35", "Chicago"],
    ]
    ui.table(headers, rows)


def demo_prompt():
    """Demonstrate prompts."""
    print("\n=== Prompts ===")

    # Text input
    name = ui.prompt("What's your name?", default="Anonymous")
    print(f"Hello, {name}!")

    # Confirmation
    if ui.confirm("Do you like this demo?"):
        print("Great!")
    else:
        print("Sorry to hear that!")

    # Selection
    color = ui.select("Pick a color:", choices=["Red", "Green", "Blue"])
    print(f"You picked: {color}")


def demo_styled_output():
    """Demonstrate styled output."""
    print("\n=== Styled Output ===")

    ui.success("This is a success message")
    ui.warning("This is a warning message")
    ui.error("This is an error message")
    ui.info("This is an info message")


if __name__ == "__main__":
    print("PrettyCLI UI Components Demo")
    print("=" * 40)

    demo_styled_output()
    demo_table()
    demo_progress()
    demo_spinner()

    # Interactive prompts (optional)
    try:
        demo_prompt()
    except (KeyboardInterrupt, EOFError):
        print("\nSkipped interactive prompts")
