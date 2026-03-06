"""
比对模块

基于策略模式的数据比对框架。

使用:
    from aidevtools.compare import CompareEngine

    # 渐进式分级 (默认，早停)
    engine = CompareEngine.progressive()
    results = engine.run(dut, golden)

    # 深度模式 (三级全执行)
    engine = CompareEngine.progressive(deep=True)
    results = engine.run(dut, golden)

架构:
    CompareEngine ← 执行引擎
        ↓
    CompareStrategy ← 策略接口
        ├─ ExactStrategy ← 精确比对 + bit 统计
        ├─ FuzzyStrategy ← 模糊比对
        ├─ SanityStrategy ← 自检
        ├─ BitAnalysisStrategy ← Bit级语义分析
        ├─ BlockedStrategy ← 分块分析
        └─ CompositeStrategy ← 自定义组合
    ProgressiveStrategy ← 渐进式三级分析 (L1→L2→L3)
"""

# 配置
from .types import CompareConfig

# 引擎
from .engine import CompareEngine

# 策略 + 结果类型
from .strategy import (
    CompareContext,
    CompareStrategy,
    ExactStrategy,
    ExactResult,
    FuzzyStrategy,
    FuzzyResult,
    SanityStrategy,
    SanityResult,
    BlockedStrategy,
    BlockResult,
    BitAnalysisStrategy,
    BitAnalysisResult,
    CompositeStrategy,
    TieredStrategy,
    ProgressiveStrategy,
    StrategyLevel,
    FloatFormat,
    BitLayout,
    FP32,
    FP16,
    BFLOAT16,
)

# 指标计算
from .metrics import (
    AllMetrics,
    calc_all_metrics,
    calc_qsnr,
    calc_cosine,
)

# 模型级分析
from .model import ModelTieredAnalyzer

# 报告生成
from .report.text_report import (
    print_strategy_table,
    format_strategy_results,
    generate_strategy_json,
    print_joint_report,
    visualize_joint_report,
)
from .report.visualizer import Visualizer
from .report.model_visualizer import (
    ModelVisualizer,
    ModelCompareResult,
    OpCompareResult,
    OpStatus,
)

__all__ = [
    # 核心类型
    "CompareConfig",
    "ExactResult",
    "FuzzyResult",
    "SanityResult",
    # 引擎
    "CompareEngine",
    # 策略
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
    # 结果类型
    "BlockResult",
    "BitAnalysisResult",
    # 格式定义
    "FloatFormat",
    "BitLayout",
    "FP32",
    "FP16",
    "BFLOAT16",
    # 指标计算
    "AllMetrics",
    "calc_all_metrics",
    "calc_qsnr",
    "calc_cosine",
    # 模型级分析
    "ModelTieredAnalyzer",
    # 报告生成
    "print_strategy_table",
    "format_strategy_results",
    "generate_strategy_json",
    "print_joint_report",
    "visualize_joint_report",
    # 可视化
    "Visualizer",
    "ModelVisualizer",
    "ModelCompareResult",
    "OpCompareResult",
    "OpStatus",
]
