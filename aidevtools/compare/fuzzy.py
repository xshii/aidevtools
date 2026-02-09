"""
模糊比对便捷函数（向后兼容）

新代码请使用 FuzzyStrategy:
    from aidevtools.compare.strategy import FuzzyStrategy
    result = FuzzyStrategy.compare(golden, dut, config)
"""

from .strategy.fuzzy import FuzzyStrategy
from .types import FuzzyResult, CompareConfig


def compare_fuzzy(golden, result, config=None) -> FuzzyResult:
    """模糊比对（向后兼容）"""
    return FuzzyStrategy.compare(golden, result, config)


def compare_isclose(golden, result, atol=1e-5, rtol=1e-3, max_exceed_ratio=0.0) -> FuzzyResult:
    """IsClose 比对（向后兼容）"""
    return FuzzyStrategy.compare_isclose(golden, result, atol=atol, rtol=rtol,
                                          max_exceed_ratio=max_exceed_ratio)
