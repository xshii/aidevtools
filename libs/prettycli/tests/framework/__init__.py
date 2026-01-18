"""
tests.framework - 交互式测试框架

用于测试交互式 CLI 应用程序。

Example:
    >>> from tests.framework import test, mock_prompt, TestRunner
    >>>
    >>> @test("should greet user")
    ... def test_greet():
    ...     with mock_prompt(["Alice"]):
    ...         result = greet_user()
    ...         assert "Alice" in result
    >>>
    >>> runner = TestRunner()
    >>> runner.discover(Path("tests/"))
    >>> runner.run(interactive=True)
"""
from tests.framework.session import *
from tests.framework.assertions import *
from tests.framework.mock import *
from tests.framework.runner import *
