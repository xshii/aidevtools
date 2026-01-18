"""AI Dev Tools CLI entry point."""

from prettycli import CLI


def main():
    """Main entry point for aidev CLI."""
    cli = CLI("aidev")
    cli.run()


if __name__ == "__main__":
    main()
