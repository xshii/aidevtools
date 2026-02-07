"""
比对模块

提供精确比对、模糊比对、Golden 自检功能，支持 4 种状态判定。

状态判定矩阵:
    DUT vs Golden | Golden 自检 | 判定状态
    --------------|-------------|---------------
    PASS          | PASS        | PASS
    PASS          | FAIL        | GOLDEN_SUSPECT
    FAIL          | PASS        | DUT_ISSUE
    FAIL          | FAIL        | BOTH_SUSPECT

基本使用:
    from aidevtools.compare import compare_full, CompareConfig, FP32

    # 执行完整比对
    result = compare_full(
        dut_output=dut,
        golden_pure=golden_fp32,
        golden_qnt=golden_qnt,
    )
    print(f"Status: {result.status.value}")

    # 统一管线: 四态 + bitwise + blocked 一步到位
    config = CompareConfig(
        fuzzy_min_qsnr=25.0,
        fuzzy_min_cosine=0.99,
        enable_bitwise=True,
        bitwise_fmt=FP32,
        enable_blocked=True,
        blocked_block_size=64,
    )
    result = compare_full(dut, golden, config=config)
    # result.status / result.bitwise / result.blocked 均可访问
"""

# --- 核心类型 ---
from .types import (
    CompareConfig,
    CompareResult,
    CompareStatus,
    ExactResult,
    FuzzyResult,
    SanityResult,
)

# --- 引擎 ---
from .engine import CompareEngine, compare_full, determine_status

# --- 比对函数 ---
from .exact import compare_exact, compare_bit
from .fuzzy import compare_fuzzy, compare_isclose
from .sanity import check_golden_sanity, check_data_sanity

# --- 指标计算 ---
from .metrics import (
    AllMetrics,
    calc_all_metrics,
    calc_all_metrics_early_exit,
    calc_qsnr,
    calc_cosine,
    calc_abs_error,
    calc_rel_error,
    calc_exceed_count,
    check_nan_inf,
    check_nonzero,
)

# --- 分块比对 ---
from .blocked import (
    BlockResult,
    compare_blocked,
    print_block_heatmap,
    find_worst_blocks,
)

# --- Bit 级分析 ---
from .bitwise import (
    FloatFormat,
    BitLayout,
    FP32,
    FP16,
    BFP16,
    BFP8,
    BFP4,
    INT8,
    UINT8,
    WarnLevel,
    BitDiffSummary,
    BitWarning,
    BitAnalysisResult,
    ModelBitAnalysis,
    compare_bitwise,
    compare_model_bitwise,
    print_bit_template,
    print_bit_analysis,
    print_bit_heatmap,
    print_model_bit_analysis,
    gen_bit_heatmap_svg,
    gen_perbit_bar_svg,
)

# --- 报告 ---
from .report import (
    print_compare_table,
    generate_text_report,
    generate_json_report,
)


# 公共 API — 仅包含 demos/CLI 实际使用的功能
# 其余符号仍可通过子模块直接导入 (如 from aidevtools.compare.metrics import ...)
__all__ = [
    # 核心类型
    "CompareConfig",
    "CompareResult",
    "CompareStatus",
    # 引擎
    "CompareEngine",
    "compare_full",
    # 比对函数
    "compare_exact",
    "compare_bit",
    "compare_fuzzy",
    "compare_isclose",
    "check_golden_sanity",
    # 指标
    "calc_all_metrics",
    "calc_qsnr",
    "calc_cosine",
    # 分块比对
    "compare_blocked",
    "print_block_heatmap",
    "find_worst_blocks",
    # 报告
    "print_compare_table",
    "generate_text_report",
    "generate_json_report",
    # Bit 级分析
    "FloatFormat",
    "BitLayout",
    "FP32",
    "FP16",
    "BFP16",
    "BFP8",
    "BFP4",
    "compare_bitwise",
    "compare_model_bitwise",
    "print_bit_analysis",
    "print_bit_heatmap",
    "print_model_bit_analysis",
    "gen_bit_heatmap_svg",
    "gen_perbit_bar_svg",
]
