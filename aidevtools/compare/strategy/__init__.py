"""
比对策略模块

提供基于策略模式的比对功能。
"""

# 基础设施
from .base import CompareContext, CompareStrategy

# 具体策略
from .exact import ExactStrategy
from .fuzzy import FuzzyStrategy
from .sanity import SanityStrategy
from .blocked import BlockedStrategy, BlockResult
from .bit_xor import BitXorStrategy, BitXorResult
from .bit_analysis import (
    BitAnalysisStrategy,
    BitAnalysisResult,
    FloatFormat,
    BitLayout,
    FP32,
    FP16,
    BFLOAT16,
    BFP16,
    BFP8,
    BFP4,
)

# 组合策略
from .composite import (
    CompositeStrategy,
    QuickCheckStrategy,
    StandardStrategy,
    DeepAnalysisStrategy,
    MinimalStrategy,
)

# 分级策略
from .tiered import (
    TieredStrategy,
    ProgressiveStrategy,
    QuickThenDeepStrategy,
    StrategyLevel,
    # 条件函数
    always_continue,
    never_continue,
    stop_if_exact_passed,
    stop_if_fuzzy_passed,
)

__all__ = [
    # 基础
    "CompareContext",
    "CompareStrategy",
    # 具体策略
    "ExactStrategy",
    "FuzzyStrategy",
    "SanityStrategy",
    "BlockedStrategy",
    "BitXorStrategy",
    "BitAnalysisStrategy",  # 高级调试工具（可选）
    # 组合策略
    "CompositeStrategy",
    "QuickCheckStrategy",
    "StandardStrategy",
    "DeepAnalysisStrategy",
    "MinimalStrategy",
    # 分级策略
    "TieredStrategy",
    "ProgressiveStrategy",
    "QuickThenDeepStrategy",
    "StrategyLevel",
    "always_continue",
    "never_continue",
    "stop_if_exact_passed",
    "stop_if_fuzzy_passed",
    # 结果类型
    "BlockResult",
    "BitXorResult",
    # Bit Analysis 结果和格式
    "BitAnalysisResult",
    "FloatFormat",
    "BitLayout",
    "FP32",
    "FP16",
    "BFLOAT16",
    "BFP16",
    "BFP8",
    "BFP4",
]
