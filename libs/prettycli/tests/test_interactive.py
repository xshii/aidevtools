"""交互式测试示例"""
from pathlib import Path

from tests.framework import (
    test,
    mock_prompt,
    assert_contains,
    TestRunner,
)


# 被测试的函数
def greet():
    """简单问候函数"""
    from prompt_toolkit import PromptSession
    session = PromptSession()
    name = session.prompt("What's your name? ")
    return f"Hello, {name}!"


def calculator():
    """简单计算器"""
    from prompt_toolkit import PromptSession
    session = PromptSession()

    a = int(session.prompt("First number: "))
    op = session.prompt("Operator (+, -, *, /): ")
    b = int(session.prompt("Second number: "))

    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b


# 测试用例
@test("should greet user by name")
def test_greet():
    with mock_prompt(["Alice"]):
        result = greet()
        assert_contains(result, "Alice")
        assert_contains(result, "Hello")


@test("should add two numbers", tags=["math"])
def test_add():
    with mock_prompt(["5", "+", "3"]):
        result = calculator()
        assert result == 8


@test("should multiply two numbers", tags=["math"])
def test_multiply():
    with mock_prompt(["4", "*", "7"]):
        result = calculator()
        assert result == 28


@test("should handle division", tags=["math"])
def test_divide():
    with mock_prompt(["10", "/", "2"]):
        result = calculator()
        assert result == 5


if __name__ == "__main__":
    import sys
    runner = TestRunner()
    runner.discover(Path(__file__).parent)

    # 交互模式用 --interactive 参数
    interactive = "--interactive" in sys.argv or "-i" in sys.argv
    sys.exit(runner.run(interactive=interactive))
