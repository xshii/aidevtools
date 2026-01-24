"""Mock 工具，用于模拟 prompt_toolkit 输入"""
import sys
from typing import List, Iterator, Any
from unittest.mock import patch
from contextlib import contextmanager

__all__ = ["MockInput", "mock_prompt", "mock_confirm", "mock_select"]


class MockInput:
    """模拟输入流

    Example:
        >>> mock = MockInput(["hello", "world"])
        >>> with mock.patch_prompt():
        ...     result = session.prompt("> ")  # returns "hello"
        ...     result = session.prompt("> ")  # returns "world"
    """

    def __init__(self, inputs: List[str]):
        self._inputs = iter(inputs)
        self._index = 0

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        return next(self._inputs)

    def readline(self) -> str:
        try:
            return next(self._inputs) + "\n"
        except StopIteration:
            return ""

    @contextmanager
    def patch_stdin(self):
        """替换 sys.stdin"""
        original = sys.stdin
        sys.stdin = self
        try:
            yield
        finally:
            sys.stdin = original

    @contextmanager
    def patch_prompt(self):
        """替换 prompt_toolkit 的 PromptSession.prompt"""
        from prompt_toolkit import PromptSession


        def mock_prompt_method(self_session, *args, **kwargs):
            try:
                return next(self._inputs)
            except StopIteration:
                raise EOFError()

        with patch.object(PromptSession, 'prompt', mock_prompt_method):
            yield


@contextmanager
def mock_prompt(responses: List[str]):
    """快捷方式：模拟 prompt 输入

    Example:
        >>> with mock_prompt(["alice", "password123"]):
        ...     name = session.prompt("Name: ")
        ...     pwd = session.prompt("Password: ")
    """
    mock = MockInput(responses)
    with mock.patch_prompt():
        yield mock


@contextmanager
def mock_confirm(responses: List[bool]):
    """模拟确认对话框

    Example:
        >>> with mock_confirm([True, False]):
        ...     result1 = ui.confirm("Continue?")  # True
        ...     result2 = ui.confirm("Delete?")    # False
    """
    responses_iter = iter(responses)

    def mock_confirm_func(*args, **kwargs):
        return next(responses_iter)

    with patch("prettycli.ui.confirm", side_effect=mock_confirm_func):
        yield


@contextmanager
def mock_select(responses: List[Any]):
    """模拟选择对话框

    Example:
        >>> with mock_select(["option1", "option2"]):
        ...     result = ui.select("Choose:", options)
    """
    responses_iter = iter(responses)

    def mock_select_func(*args, **kwargs):
        return next(responses_iter)

    with patch("prettycli.ui.select", side_effect=mock_select_func):
        yield
