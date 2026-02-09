"""
精确比对便捷函数（向后兼容）

新代码请使用 ExactStrategy:
    from aidevtools.compare.strategy import ExactStrategy
    result = ExactStrategy.compare(golden, dut)
"""

from .strategy.exact import ExactStrategy
from .types import ExactResult


def compare_exact(golden, result, max_abs=0.0, max_count=0) -> ExactResult:
    """精确比对（向后兼容）"""
    return ExactStrategy.compare(golden, result, max_abs=max_abs, max_count=max_count)


def compare_bit(golden_bytes: bytes, result_bytes: bytes) -> bool:
    """bit 级字节比对（向后兼容）"""
    return ExactStrategy.compare_bytes(golden_bytes, result_bytes)
