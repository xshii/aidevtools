"""Install VS Code extension command."""
import logging

from prettycli.command import command
from prettycli import vscode


@command("install-vscode", help="Install the prettycli VS Code extension")
def install_vscode():
    """Install the prettycli VS Code extension."""
    if vscode.is_extension_installed():
        logging.info("VS Code extension is already installed")
        return 0

    logging.info("Installing VS Code extension...")
    if vscode.install_extension():
        logging.info(
            "VS Code extension installed. "
            "Please reload VS Code (Cmd+Shift+P -> Reload Window)"
        )
        return 0
    else:
        logging.error("Failed to install VS Code extension")
        return 1
