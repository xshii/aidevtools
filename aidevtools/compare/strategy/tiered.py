"""
分级策略 - 渐进式分析

支持根据前一级结果决定是否执行下一级，适用于：
- 快速筛选 → 详细诊断
- 节省计算资源（大模型场景）
- 渐进式错误定位

使用场景：
    # 单算子分级
    engine = CompareEngine.progressive()
    result = engine.run(dut, golden)

    # 模型级分级（全局协调）
    analyzer = CompareEngine.model_progressive()
    results = analyzer.progressive_analyze(per_op_pairs)
"""
from typing import List, Callable, Dict, Any
from dataclasses import dataclass

from .base import CompareStrategy, CompareContext
from .exact import ExactStrategy
from .fuzzy import FuzzyStrategy
from .sanity import SanityStrategy
from .blocked import BlockedStrategy
from .bit_xor import BitXorStrategy
from .bit_analysis import BitAnalysisStrategy


# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class StrategyLevel:
    """
    策略级别定义

    Attributes:
        name: 级别名称（如 "L1_quick", "L2_medium", "L3_deep"）
        strategies: 该级别执行的策略列表
        condition: 判断函数，返回 True 表示需要执行下一级
    """
    name: str
    strategies: List[CompareStrategy]
    condition: Callable[[Dict[str, Any]], bool]


# ============================================================================
# 预定义条件函数
# ============================================================================


def always_continue(results: Dict[str, Any]) -> bool:
    """总是继续下一级（用于中间级）"""
    return True


def never_continue(results: Dict[str, Any]) -> bool:
    """从不继续（用于最后一级）"""
    return False


def stop_if_exact_passed(results: Dict[str, Any]) -> bool:
    """如果 exact 通过则停止，否则继续"""
    exact = results.get("exact")
    if exact is None:
        return True  # 未执行exact，继续
    return not exact.get("passed", False)


def stop_if_fuzzy_passed(results: Dict[str, Any]) -> bool:
    """如果 fuzzy（pure或qnt任一）通过则停止"""
    fuzzy_pure = results.get("fuzzy_pure")
    fuzzy_qnt = results.get("fuzzy_qnt")

    # 任一fuzzy通过则停止
    pure_passed = fuzzy_pure and fuzzy_pure.get("passed", False)
    qnt_passed = fuzzy_qnt and fuzzy_qnt.get("passed", False)

    return not (pure_passed or qnt_passed)


# ============================================================================
# 分级策略基类
# ============================================================================


class TieredStrategy(CompareStrategy):
    """
    分级策略 - 根据前一级结果决定是否执行下一级

    工作流程：
        1. 执行 Level 1 的所有策略
        2. 根据 Level 1 结果 + condition 判断是否继续
        3. 如果需要，执行 Level 2 → Level 3 ...
        4. 返回所有执行过的级别的结果

    示例:
        strategy = TieredStrategy([
            StrategyLevel("L1", [ExactStrategy()], stop_if_exact_passed),
            StrategyLevel("L2", [FuzzyStrategy()], stop_if_fuzzy_passed),
            StrategyLevel("L3", [BlockedStrategy()], never_continue),
        ])
    """

    def __init__(self, levels: List[StrategyLevel], name: str = "tiered"):
        """
        Args:
            levels: 策略级别列表（按执行顺序）
            name: 策略名称
        """
        self.levels = levels
        self._name = name

    def run(self, ctx: CompareContext) -> Dict[str, Any]:
        """
        执行分级比对

        Returns:
            {
                # 各策略结果
                "exact": {...},
                "fuzzy_pure": {...},
                ...
                # 元信息
                "_executed_levels": ["L1", "L2"],  # 执行了哪些级别
                "_stopped_at": "L2",  # 在哪一级停止
            }
        """
        results = {"_executed_levels": []}

        for i, level in enumerate(self.levels):
            # 执行当前级别的所有策略
            for strategy in level.strategies:
                result = strategy.run(ctx)
                results[strategy.name] = result

            results["_executed_levels"].append(level.name)

            # 检查是否需要下一级
            should_continue = level.condition(results)
            if not should_continue:
                results["_stopped_at"] = level.name
                break
        else:
            # 执行完所有级别
            results["_stopped_at"] = self.levels[-1].name if self.levels else None

        return results

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        level_names = [f"{lvl.name}({len(lvl.strategies)} strategies)"
                       for lvl in self.levels]
        return f"TieredStrategy(name={self.name!r}, levels={level_names})"


# ============================================================================
# 预定义分级策略
# ============================================================================


class QuickThenDeepStrategy(TieredStrategy):
    """
    快速检查 → 深度分析（如果失败）

    两级策略：
    - L1: Exact + Bitwise（快速判断是否有差异）
    - L2: Fuzzy + Sanity + Blocked（深度分析，如果L1失败）

    适用场景：
        - 大部分数据预期一致
        - 需要快速筛选出有问题的算子
    """

    def __init__(self, block_size: int = 64):
        """
        Args:
            block_size: 分块大小（用于 BlockedStrategy）
        """
        super().__init__(
            levels=[
                StrategyLevel(
                    name="L1_quick",
                    strategies=[ExactStrategy(), BitXorStrategy()],
                    condition=stop_if_exact_passed,
                ),
                StrategyLevel(
                    name="L2_deep",
                    strategies=[
                        FuzzyStrategy(),
                        SanityStrategy(),
                        BlockedStrategy(block_size=block_size),
                    ],
                    condition=never_continue,
                ),
            ],
            name="quick_then_deep",
        )


class ProgressiveStrategy(TieredStrategy):
    """
    渐进式三级分析

    三级策略：
    - L1: Exact + Bitwise（初步检查）
    - L2: Fuzzy + Sanity（中度诊断）
    - L3: BitwisePro + Blocked（深度定位）

    适用场景：
        - 需要逐级定位问题根因
        - 大模型调试（节省计算资源）
        - 不确定错误类型
    """

    def __init__(self, block_size: int = 64):
        """
        Args:
            block_size: 分块大小（用于 BlockedStrategy）
        """
        super().__init__(
            levels=[
                StrategyLevel(
                    name="L1_quick",
                    strategies=[ExactStrategy(), BitXorStrategy()],
                    condition=stop_if_exact_passed,
                ),
                StrategyLevel(
                    name="L2_medium",
                    strategies=[FuzzyStrategy(), SanityStrategy()],
                    condition=stop_if_fuzzy_passed,
                ),
                StrategyLevel(
                    name="L3_deep",
                    strategies=[
                        BitAnalysisStrategy(),
                        BlockedStrategy(block_size=block_size),
                    ],
                    condition=never_continue,
                ),
            ],
            name="progressive",
        )


__all__ = [
    "StrategyLevel",
    "TieredStrategy",
    "QuickThenDeepStrategy",
    "ProgressiveStrategy",
    # 条件函数
    "always_continue",
    "never_continue",
    "stop_if_exact_passed",
    "stop_if_fuzzy_passed",
]
