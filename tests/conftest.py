import pytest
from pathlib import Path

from prettycli import BaseCommand, App, statusbar


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clear_registry():
    """每个测试前后清空注册"""
    BaseCommand.clear()
    statusbar.clear()
    yield
    BaseCommand.clear()
    statusbar.clear()


@pytest.fixture
def app_with_fixtures():
    """创建包含所有 fixtures 命令的 App"""
    app = App("testcli", "Test CLI")
    app.register(FIXTURES_DIR)
    return app
