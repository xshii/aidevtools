#!/usr/bin/env python3
"""
Status Bar Example - Shows how to use the status bar system.

Usage:
    python 05_statusbar.py
"""
import time
from prettycli import statusbar, RuntimeStatus, SystemStatus, EchoStatus


def demo_basic_status():
    """Basic status bar usage."""
    print("\n=== Basic Status Bar ===")

    # Register a simple status provider
    def my_status():
        return ("MyApp", "Running")

    statusbar.register(my_status)

    # Render the status bar
    print("Status bar:")
    print(statusbar.render())


def demo_runtime_status():
    """Runtime status for tracking execution time."""
    print("\n=== Runtime Status ===")

    runtime = RuntimeStatus()
    statusbar.register(runtime.get_status)

    print("Initial status:")
    print(statusbar.render())

    time.sleep(1)

    print("\nAfter 1 second:")
    print(statusbar.render())


def demo_system_status():
    """System status showing CPU and memory."""
    print("\n=== System Status ===")

    system = SystemStatus()
    statusbar.register(system.get_status)

    print("System status:")
    print(statusbar.render())


def demo_echo_status():
    """Echo status for custom messages."""
    print("\n=== Echo Status ===")

    echo = EchoStatus()
    statusbar.register(echo.get_status)

    echo.set("Processing", "Step 1 of 3")
    print(statusbar.render())

    echo.set("Processing", "Step 2 of 3")
    print(statusbar.render())

    echo.set("Done", "Completed!")
    print(statusbar.render())


def demo_multiple_providers():
    """Multiple status providers combined."""
    print("\n=== Multiple Providers ===")

    # Clear existing providers
    statusbar._providers.clear()

    # Add multiple providers
    runtime = RuntimeStatus()
    echo = EchoStatus()

    statusbar.register(runtime.get_status)
    statusbar.register(echo.get_status)
    statusbar.register(lambda: ("Mode", "Development"))

    echo.set("Task", "Building...")

    print("Combined status bar:")
    print(statusbar.render())


if __name__ == "__main__":
    print("PrettyCLI Status Bar Demo")
    print("=" * 40)

    demo_basic_status()
    demo_runtime_status()
    demo_system_status()
    demo_echo_status()
    demo_multiple_providers()
