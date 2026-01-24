"""测试运行器"""
import time
import traceback
from pathlib import Path
from typing import List, Callable, Optional
from dataclasses import dataclass, field

from prettycli import ui

__all__ = ["TestRunner", "TestCase", "test", "before_each", "after_each"]

# 装饰器收集的测试用例
_test_cases: List["TestCase"] = []
_before_each: Optional[Callable] = None
_after_each: Optional[Callable] = None


@dataclass
class TestCase:
    """测试用例"""
    name: str
    func: Callable
    tags: List[str] = field(default_factory=list)
    timeout: float = 30.0


def test(name: str = None, tags: List[str] = None, timeout: float = 30.0):
    """测试用例装饰器

    Example:
        >>> @test("should greet user")
        ... def test_greet():
        ...     with mock_prompt(["Alice"]):
        ...         result = greet()
        ...         assert "Alice" in result
    """
    def decorator(func: Callable) -> Callable:
        case = TestCase(
            name=name or func.__name__,
            func=func,
            tags=tags or [],
            timeout=timeout,
        )
        _test_cases.append(case)
        return func
    return decorator


def before_each(func: Callable) -> Callable:
    """每个测试前执行"""
    global _before_each
    _before_each = func
    return func


def after_each(func: Callable) -> Callable:
    """每个测试后执行"""
    global _after_each
    _after_each = func
    return func


@dataclass
class TestResult:
    """测试结果"""
    case: TestCase
    passed: bool
    error: Optional[str] = None
    traceback: Optional[str] = None
    duration: float = 0.0


class TestRunner:
    """交互式测试运行器

    Example:
        >>> runner = TestRunner()
        >>> runner.discover(Path("tests/"))
        >>> runner.run()  # 运行所有测试
        >>> runner.run(tags=["unit"])  # 只运行带 unit 标签的测试
    """

    def __init__(self):
        self._cases: List[TestCase] = []
        self._results: List[TestResult] = []

    def add(self, case: TestCase):
        """添加测试用例"""
        self._cases.append(case)

    def discover(self, path: Path, recursive: bool = False):
        """从目录发现测试

        Args:
            path: 测试目录
            recursive: 是否递归搜索子目录
        """
        global _test_cases
        _test_cases.clear()

        import importlib.util

        pattern = "test_*.py"
        files = path.rglob(pattern) if recursive else path.glob(pattern)

        for file in files:
            spec = importlib.util.spec_from_file_location(file.stem, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

        self._cases.extend(_test_cases)
        _test_cases.clear()

    def run(
        self,
        tags: List[str] = None,
        pattern: str = None,
        interactive: bool = False,
    ) -> int:
        """运行测试

        Args:
            tags: 只运行这些标签的测试
            pattern: 测试名模式
            interactive: 交互模式（失败时暂停）

        Returns:
            失败的测试数量（0 表示全部通过）
        """
        self._results.clear()
        cases = self._filter_cases(tags, pattern)

        ui.print(f"\n[bold]Running {len(cases)} tests...[/]\n")

        for i, case in enumerate(cases, 1):
            self._run_case(case, i, len(cases), interactive)

        self._print_summary()

        # 返回失败数量作为退出码
        failed = sum(1 for r in self._results if not r.passed)
        return failed

    def _filter_cases(
        self,
        tags: List[str] = None,
        pattern: str = None,
    ) -> List[TestCase]:
        """过滤测试用例"""
        cases = self._cases

        if tags:
            cases = [c for c in cases if any(t in c.tags for t in tags)]

        if pattern:
            import re
            regex = re.compile(pattern)
            cases = [c for c in cases if regex.search(c.name)]

        return cases

    def _run_case(
        self,
        case: TestCase,
        index: int,
        total: int,
        interactive: bool,
    ):
        """运行单个测试"""
        ui.print(f"[dim][{index}/{total}][/] {case.name} ", end="")

        start = time.time()
        error = None
        tb = None

        try:
            # before_each
            if _before_each:
                _before_each()

            # 运行测试
            case.func()

            # after_each
            if _after_each:
                _after_each()

            passed = True
            ui.print("[green]✓ PASS[/]")

        except Exception as e:
            passed = False
            error = str(e)
            tb = traceback.format_exc()
            ui.print("[red]✗ FAIL[/]")
            ui.print(f"[red]  {error}[/]")

            if interactive:
                self._interactive_fail(case, error, tb)

        duration = time.time() - start
        self._results.append(TestResult(
            case=case,
            passed=passed,
            error=error,
            traceback=tb,
            duration=duration,
        ))

    def _interactive_fail(self, case: TestCase, error: str, tb: str):
        """交互式失败处理"""
        ui.print("\n[yellow]Test failed. Options:[/]")
        ui.print("  [c]ontinue  [r]etry  [d]ebug  [q]uit")

        while True:
            choice = input("> ").strip().lower()
            if choice == "c":
                break
            elif choice == "r":
                ui.print("\n[dim]Retrying...[/]")
                try:
                    case.func()
                    ui.print("[green]✓ PASS on retry[/]")
                except Exception as e:
                    ui.print(f"[red]✗ Still failing: {e}[/]")
                break
            elif choice == "d":
                ui.print("\n[dim]Traceback:[/]")
                ui.print(tb)
            elif choice == "q":
                raise KeyboardInterrupt("User quit")

    def _print_summary(self):
        """打印测试摘要"""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed
        duration = sum(r.duration for r in self._results)

        ui.print("\n" + "=" * 50)

        if failed == 0:
            ui.print(f"[green bold]All {total} tests passed![/] ({duration:.2f}s)")
        else:
            ui.print(f"[red bold]{failed} failed[/], {passed} passed ({duration:.2f}s)")

            ui.print("\n[red]Failed tests:[/]")
            for r in self._results:
                if not r.passed:
                    ui.print(f"  ✗ {r.case.name}")
                    if r.error:
                        ui.print(f"    [dim]{r.error}[/]")

        ui.print("=" * 50)
