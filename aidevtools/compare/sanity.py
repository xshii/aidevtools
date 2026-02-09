"""
Golden 自检便捷函数（向后兼容）

新代码请使用 SanityStrategy:
    from aidevtools.compare.strategy import SanityStrategy
    result = SanityStrategy.compare(golden_pure, golden_qnt, config)
"""

from .strategy.sanity import SanityStrategy
from .types import SanityResult, CompareConfig


def check_golden_sanity(golden_pure, golden_qnt=None, config=None) -> SanityResult:
    """Golden 自检（向后兼容）"""
    return SanityStrategy.compare(golden_pure, golden_qnt, config)


def check_data_sanity(data, name="data") -> SanityResult:
    """通用数据自检（向后兼容）"""
    return SanityStrategy.check_data(data, name)
