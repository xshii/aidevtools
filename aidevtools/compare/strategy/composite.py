"""
组合策略

将多个策略组合，提供预定义的比对方案。
"""
from typing import List, Dict, Any
from .base import CompareStrategy, CompareContext
from .exact import ExactStrategy
from .fuzzy import FuzzyStrategy
from .sanity import SanityStrategy
from .blocked import BlockedStrategy
from .bit_analysis import BitAnalysisStrategy


class CompositeStrategy(CompareStrategy):
    """
    组合策略

    将多个子策略组合，按顺序执行。

    特点：
    - 灵活组装多个策略
    - 统一返回格式
    - 支持策略间共享PreparedPair
    """

    def __init__(self, strategies: List[CompareStrategy], name: str = "composite"):
        """
        Args:
            strategies: 子策略列表
            name: 组合策略名称
        """
        self.strategies = strategies
        self._name = name

    def prepare(self, ctx: CompareContext) -> None:
        """预处理所有子策略"""
        for strategy in self.strategies:
            strategy.prepare(ctx)

    def run(self, ctx: CompareContext) -> Dict[str, Any]:
        """
        执行所有子策略

        Returns:
            {strategy.name: result} 字典
        """
        results = {}
        for strategy in self.strategies:
            results[strategy.name] = strategy.run(ctx)
        return results

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        strategy_names = [s.name for s in self.strategies]
        return f"CompositeStrategy(name={self.name!r}, strategies={strategy_names})"


# ============================================================================
# 预定义组合策略
# ============================================================================


class QuickCheckStrategy(CompositeStrategy):
    """
    快速检查策略

    包含：
    - 精确比对
    - 模糊比对（纯FP32）

    适用场景：
    - 快速验证（CI/CD）
    - 初步检查
    """

    def __init__(self):
        super().__init__(
            strategies=[
                ExactStrategy(),
                FuzzyStrategy(use_golden_qnt=False),
            ],
            name="quick_check"
        )

    def run(self, ctx: CompareContext) -> Dict[str, Any]:
        """执行所有子策略并计算状态"""
        results = super().run(ctx)

        # 计算总体状态
        from ..types import CompareStatus

        # 判断 DUT 是否通过
        dut_passed = False
        if results.get('exact') and results['exact'].passed:
            dut_passed = True
        elif results.get('fuzzy_pure') and results['fuzzy_pure'].passed:
            dut_passed = True

        # 快速检查策略没有 sanity，所以 golden 总是有效
        golden_valid = True

        # 状态判定
        if dut_passed and golden_valid:
            status = CompareStatus.PASS
        else:
            status = CompareStatus.DUT_ISSUE

        results['status'] = status
        return results


class StandardStrategy(CompositeStrategy):
    """
    标准比对策略（推荐）

    包含：
    - 精确比对
    - 模糊比对（纯FP32）
    - 模糊比对（量化感知）
    - Golden自检

    适用场景：
    - 日常开发验证
    - 回归测试
    """

    def __init__(self):
        super().__init__(
            strategies=[
                ExactStrategy(),
                FuzzyStrategy(use_golden_qnt=False),
                FuzzyStrategy(use_golden_qnt=True),
                SanityStrategy(),
            ],
            name="standard"
        )

    def run(self, ctx: CompareContext) -> Dict[str, Any]:
        """执行所有子策略并计算状态"""
        results = super().run(ctx)

        # 计算总体状态
        from ..types import CompareStatus

        # 判断 DUT 是否通过 (exact 或 fuzzy_qnt 通过)
        dut_passed = False
        if results.get('exact') and results['exact'].passed:
            dut_passed = True
        elif results.get('fuzzy_qnt') and results['fuzzy_qnt'].passed:
            dut_passed = True
        elif results.get('fuzzy_pure') and results['fuzzy_pure'].passed:
            # 如果没有 fuzzy_qnt，fallback 到 fuzzy_pure
            dut_passed = True

        # 判断 Golden 是否有效
        golden_valid = True
        if results.get('sanity'):
            golden_valid = results['sanity'].valid

        # 状态判定矩阵
        if dut_passed and golden_valid:
            status = CompareStatus.PASS
        elif dut_passed and not golden_valid:
            status = CompareStatus.GOLDEN_SUSPECT
        elif not dut_passed and golden_valid:
            status = CompareStatus.DUT_ISSUE
        else:
            status = CompareStatus.BOTH_SUSPECT

        results['status'] = status
        return results


class DeepAnalysisStrategy(CompositeStrategy):
    """
    深度分析策略

    包含：
    - 精确比对
    - 模糊比对（纯FP32）
    - 模糊比对（量化感知）
    - Golden自检
    - 分块分析

    适用场景：
    - 深度调试
    - 误差定位
    - 问题排查

    注意：不包含 bit 级分析。如需 bit 级分析，请使用 BitAnalysisStrategy。
    """

    def __init__(self, block_size: int = 1024):
        super().__init__(
            strategies=[
                ExactStrategy(),
                FuzzyStrategy(use_golden_qnt=False),
                FuzzyStrategy(use_golden_qnt=True),
                SanityStrategy(),
                BlockedStrategy(block_size=block_size),
            ],
            name="deep_analysis"
        )

    def run(self, ctx: CompareContext) -> Dict[str, Any]:
        """执行所有子策略并计算状态"""
        results = super().run(ctx)

        # 计算总体状态（与 StandardStrategy 相同）
        from ..types import CompareStatus

        dut_passed = False
        if results.get('exact') and results['exact'].passed:
            dut_passed = True
        elif results.get('fuzzy_qnt') and results['fuzzy_qnt'].passed:
            dut_passed = True
        elif results.get('fuzzy_pure') and results['fuzzy_pure'].passed:
            dut_passed = True

        golden_valid = True
        if results.get('sanity'):
            golden_valid = results['sanity'].valid

        if dut_passed and golden_valid:
            status = CompareStatus.PASS
        elif dut_passed and not golden_valid:
            status = CompareStatus.GOLDEN_SUSPECT
        elif not dut_passed and golden_valid:
            status = CompareStatus.DUT_ISSUE
        else:
            status = CompareStatus.BOTH_SUSPECT

        results['status'] = status
        return results


class MinimalStrategy(CompositeStrategy):
    """
    最小比对策略

    仅包含：
    - 模糊比对（纯FP32）

    适用场景：
    - 性能优先
    - 简单算子验证
    """

    def __init__(self):
        super().__init__(
            strategies=[
                FuzzyStrategy(use_golden_qnt=False),
            ],
            name="minimal"
        )

    def run(self, ctx: CompareContext) -> Dict[str, Any]:
        """执行所有子策略并计算状态"""
        results = super().run(ctx)

        # 计算总体状态
        from ..types import CompareStatus

        # 判断 DUT 是否通过
        dut_passed = False
        if results.get('fuzzy_pure') and results['fuzzy_pure'].passed:
            dut_passed = True

        # 最小策略没有 sanity，golden 总是有效
        golden_valid = True

        # 状态判定
        if dut_passed:
            status = CompareStatus.PASS
        else:
            status = CompareStatus.DUT_ISSUE

        results['status'] = status
        return results
