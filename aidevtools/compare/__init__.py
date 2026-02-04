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
    # 指标计算
    "calc_qsnr",
    "calc_cosine",
    "calc_abs_error",
    "calc_rel_error",
    "calc_exceed_count",
    "check_nan_inf",
    "check_nonzero",
    # 报告
    "print_compare_table",
    "generate_text_report",
    "generate_json_report",
]
