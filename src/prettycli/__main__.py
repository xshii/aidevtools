"""
Entry point for running prettycli as a module.

Usage:
    python -m prettycli
    python -m prettycli --project-root /path/to/project
"""
import argparse
from pathlib import Path

from prettycli.cli import CLI


def main():
    parser = argparse.ArgumentParser(description="PrettyCLI - Interactive CLI shell")
    parser.add_argument(
        "--project-root", "-p",
        type=Path,
        default=None,
        help="Project root directory (defaults to current directory)",
    )
    args = parser.parse_args()

    cli = CLI("prettycli", project_root=args.project_root)
    cli.run()


if __name__ == "__main__":
    main()
