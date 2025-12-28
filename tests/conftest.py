import pytest

from prettycli import BaseCommand


@pytest.fixture(autouse=True)
def clear_registry():
    """每个测试前后清空注册"""
    BaseCommand.clear()
    yield
    BaseCommand.clear()
