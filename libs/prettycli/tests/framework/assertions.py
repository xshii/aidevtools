"""测试断言"""
import re
from typing import Any, Optional

__all__ = [
    "assert_contains",
    "assert_not_contains",
    "assert_matches",
    "assert_line_count",
    "assert_equals",
    "assert_exit_code",
    "AssertionFailed",
]


class AssertionFailed(AssertionError):
    """断言失败"""

    def __init__(self, message: str, expected: Any = None, actual: Any = None):
        self.expected = expected
        self.actual = actual
        super().__init__(message)


def assert_contains(output: str, substring: str, msg: Optional[str] = None):
    """断言输出包含子串"""
    if substring not in output:
        raise AssertionFailed(
            msg or f"Expected output to contain: {substring!r}",
            expected=substring,
            actual=output,
        )


def assert_not_contains(output: str, substring: str, msg: Optional[str] = None):
    """断言输出不包含子串"""
    if substring in output:
        raise AssertionFailed(
            msg or f"Expected output NOT to contain: {substring!r}",
            expected=f"not {substring!r}",
            actual=output,
        )


def assert_matches(output: str, pattern: str, msg: Optional[str] = None):
    """断言输出匹配正则表达式"""
    if not re.search(pattern, output):
        raise AssertionFailed(
            msg or f"Expected output to match: {pattern}",
            expected=pattern,
            actual=output,
        )


def assert_line_count(output: str, expected: int, msg: Optional[str] = None):
    """断言输出行数"""
    lines = output.strip().split("\n") if output.strip() else []
    actual = len(lines)
    if actual != expected:
        raise AssertionFailed(
            msg or f"Expected {expected} lines, got {actual}",
            expected=expected,
            actual=actual,
        )


def assert_equals(actual: Any, expected: Any, msg: Optional[str] = None):
    """断言相等"""
    if actual != expected:
        raise AssertionFailed(
            msg or f"Expected {expected!r}, got {actual!r}",
            expected=expected,
            actual=actual,
        )


def assert_exit_code(actual: int, expected: int = 0, msg: Optional[str] = None):
    """断言退出码"""
    if actual != expected:
        raise AssertionFailed(
            msg or f"Expected exit code {expected}, got {actual}",
            expected=expected,
            actual=actual,
        )
