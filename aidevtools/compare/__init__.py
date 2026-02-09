"""
比对模块 (重构版)

基于策略模式的数据比对框架。

快速开始:
    from aidevtools.compare import CompareEngine

    # 方式1: 使用预定义策略
    engine = CompareEngine.standard()
    results = engine.run(dut, golden)

    # 方式2: 自定义策略组合
    from aidevtools.compare.strategy import ExactStrategy, FuzzyStrategy, CompositeStrategy
    engine = CompareEngine(CompositeStrategy([
        ExactStrategy(),
        FuzzyStrategy(),
    ]))
    results = engine.run(dut, golden)

架构:
    CompareEngine ← 执行引擎
        ↓
    CompareStrategy ← 策略接口
        ├─ ExactStrategy ← 精确比对
        ├─ FuzzyStrategy ← 模糊比对
        ├─ SanityStrategy ← 自检
        ├─ BitXorStrategy ← Bit XOR比对
        ├─ BitAnalysisStrategy ← Bit级分析（可选）
        ├─ BlockedStrategy ← 分块分析
        └─ CompositeStrategy ← 组合策略
            ├─ StandardStrategy ← 标准比对（推荐）
            ├─ QuickCheckStrategy ← 快速检查
            ├─ DeepAnalysisStrategy ← 深度分析
            └─ MinimalStrategy ← 最小比对
"""

# ============================================================================
# 核心类型
# ============================================================================

from .types import (
    CompareConfig,
    CompareResult,
    CompareStatus,
    ExactResult,
    FuzzyResult,
    SanityResult,
)

# ============================================================================
# 引擎
# ============================================================================

from .engine import (
    CompareEngine,
    compare_full,
    compare_quick,
    determine_status,
)

# 旧API兼容
from .exact import compare_exact, compare_bit
from .fuzzy import compare_fuzzy, compare_isclose
from .sanity import check_golden_sanity, check_data_sanity

# ============================================================================
# 策略
# ============================================================================

from .strategy import (
    # 基础设施
    CompareContext,
    CompareStrategy,
    # 具体策略
    ExactStrategy,
    FuzzyStrategy,
    SanityStrategy,
    BlockedStrategy,
    BitXorStrategy,
    BitAnalysisStrategy,  # 高级调试工具（可选）
    # 组合策略
    CompositeStrategy,
    StandardStrategy,
    QuickCheckStrategy,
    DeepAnalysisStrategy,
    MinimalStrategy,
    # 分级策略
    TieredStrategy,
    ProgressiveStrategy,
    QuickThenDeepStrategy,
    StrategyLevel,
)

# ============================================================================
# 结果类型和格式定义
# ============================================================================

from .strategy import (
    BlockResult,
    BitXorResult,
    BitAnalysisResult,
    ModelBitAnalysis,
    FloatFormat,
    BitLayout,
    FP32,
    FP16,
    BFLOAT16,
    BFP16,
    BFP8,
    BFP4,
    INT8,
    UINT8,
    # 可视化
    print_bit_analysis,
    print_bit_template,
    print_bit_heatmap,
    gen_bit_heatmap_svg,
    gen_perbit_bar_svg,
    # 模型级
    compare_model_bitwise,
    print_model_bit_analysis,
)


# ============================================================================
# 指标计算（高级用户）
# ============================================================================

from .metrics import (
    AllMetrics,
    calc_all_metrics,
    calc_qsnr,
    calc_cosine,
)

# ============================================================================
# 模型级分析
# ============================================================================

from .model import (
    ModelTieredAnalyzer,
)

# ============================================================================
# 报告生成
# ============================================================================

from .report import (
    # 新API
    print_strategy_table,
    format_strategy_results,
    generate_strategy_json,
    # 旧API（已废弃）
    print_compare_table,
    generate_text_report,
    generate_json_report,
)

# ============================================================================
# 导出列表
# ============================================================================

__all__ = [
    # 核心类型
    "CompareConfig",
    "CompareResult",
    "CompareStatus",
    "ExactResult",
    "FuzzyResult",
    "SanityResult",
    # 引擎
    "CompareEngine",
    "compare_full",
    "compare_quick",
    # 策略基础设施
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
    "StandardStrategy",
    "QuickCheckStrategy",
    "DeepAnalysisStrategy",
    "MinimalStrategy",
    # 分级策略
    "TieredStrategy",
    "ProgressiveStrategy",
    "QuickThenDeepStrategy",
    "StrategyLevel",
    # 结果类型
    "BlockResult",
    "BitXorResult",
    "BitAnalysisResult",
    # 格式定义
    "FloatFormat",
    "BitLayout",
    "FP32",
    "FP16",
    "BFLOAT16",
    "BFP16",
    "BFP8",
    "BFP4",
    # 指标计算
    "AllMetrics",
    "calc_all_metrics",
    "calc_qsnr",
    "calc_cosine",
    # 模型级分析
    "ModelTieredAnalyzer",
    # 报告生成（新API）
    "print_strategy_table",
    "format_strategy_results",
    "generate_strategy_json",
    # 报告生成（旧API，已废弃）
    "print_compare_table",
    "generate_text_report",
    "generate_json_report",
]
