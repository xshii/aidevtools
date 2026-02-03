"""AI Dev Tools CLI entry point."""

from prettycli import CLI

# 导入命令模块以触发注册


def main():
    """Main entry point for aidev CLI."""
    cli = CLI("aidev")
    cli.run()


if __name__ == "__main__":
    main()
