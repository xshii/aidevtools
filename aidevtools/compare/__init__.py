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
    from aidevtools.compare import compare_full, CompareConfig

    # 执行完整比对
    result = compare_full(
        dut_output=dut,
        golden_pure=golden_fp32,
        golden_qnt=golden_qnt,
    )
    print(f"Status: {result.status.value}")

    # 自定义配置
    config = CompareConfig(
        fuzzy_min_qsnr=25.0,
        fuzzy_min_cosine=0.99,
    )
    result = compare_full(dut, golden, config=config)
"""

from .types import (
    CompareConfig,
    CompareResult,
    CompareStatus,
    ExactResult,
    FuzzyResult,
    SanityResult,
)
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
from .exact import compare_exact, compare_bit
from .fuzzy import compare_fuzzy, compare_isclose
from .sanity import check_golden_sanity, check_data_sanity
from .engine import CompareEngine, compare_full, determine_status
from .blocked import (
    BlockResult,
    compare_blocked,
    print_block_heatmap,
    find_worst_blocks,
)
from .bitwise import (
    FloatFormat,
    BitLayout,
    BFP8,
    BFP4,
    INT8,
    UINT8,
    WarnLevel,
    BitDiffSummary,
    BitWarning,
    BitAnalysisResult,
    compare_bitwise,
    print_bit_template,
    print_bit_analysis,
    print_bit_heatmap,
    gen_bit_heatmap_svg,
    gen_perbit_bar_svg,
)
from .report import (
    print_compare_table,
    generate_text_report,
    generate_json_report,
)

__all__ = [
    # 类型
    "CompareConfig",
    "CompareResult",
    "CompareStatus",
    "ExactResult",
    "FuzzyResult",
    "SanityResult",
    # 引擎
    "CompareEngine",
    "compare_full",
    "determine_status",
    # 精确比对
    "compare_exact",
    "compare_bit",
    # 模糊比对
    "compare_fuzzy",
    "compare_isclose",
    # Golden 自检
    "check_golden_sanity",
    "check_data_sanity",
    # 指标计算 (优化版)
    "AllMetrics",
    "calc_all_metrics",
    "calc_all_metrics_early_exit",
    # 指标计算 (独立函数)
    "calc_qsnr",
    "calc_cosine",
    "calc_abs_error",
    "calc_rel_error",
    "calc_exceed_count",
    "check_nan_inf",
    "check_nonzero",
    # 分块比对
    "BlockResult",
    "compare_blocked",
    "print_block_heatmap",
    "find_worst_blocks",
    # Bit 级分析
    "FloatFormat",
    "BitLayout",
    "BFP8",
    "BFP4",
    "INT8",
    "UINT8",
    "WarnLevel",
    "BitDiffSummary",
    "BitWarning",
    "BitAnalysisResult",
    "compare_bitwise",
    "print_bit_template",
    "print_bit_analysis",
    "print_bit_heatmap",
    "gen_bit_heatmap_svg",
    "gen_perbit_bar_svg",
    # 报告
    "print_compare_table",
    "generate_text_report",
    "generate_json_report",
]
