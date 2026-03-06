"""
比对策略模块
"""

# 基础设施
from .base import CompareContext, CompareStrategy

# 具体策略 + 结果类型
from .exact import ExactStrategy, ExactResult
from .fuzzy import FuzzyStrategy, FuzzyResult
from .sanity import SanityStrategy, SanityResult
from .blocked import BlockedStrategy, BlockResult
from .bit_analysis import (
    BitAnalysisStrategy,
    BitAnalysisResult,
    FloatFormat,
    BitLayout,
    FP32,
    FP16,
    BFLOAT16,
)

# 组合策略
from .composite import CompositeStrategy

# 分级策略
from .tiered import (
    TieredStrategy,
    ProgressiveStrategy,
    StrategyLevel,
    always_continue,
    never_continue,
    stop_if_exact_passed,
    stop_if_fuzzy_passed,
)

__all__ = [
    "CompareContext",
    "CompareStrategy",
    "ExactStrategy",
    "FuzzyStrategy",
    "SanityStrategy",
    "BlockedStrategy",
    "BitAnalysisStrategy",
    "CompositeStrategy",
    "TieredStrategy",
    "ProgressiveStrategy",
    "StrategyLevel",
    "always_continue",
    "never_continue",
    "stop_if_exact_passed",
    "stop_if_fuzzy_passed",
    "ExactResult",
    "FuzzyResult",
    "SanityResult",
    "BlockResult",
    "BitAnalysisResult",
    "FloatFormat",
    "BitLayout",
    "FP32",
    "FP16",
    "BFLOAT16",
]
