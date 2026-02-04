"""
比对引擎

统一协调精确比对、模糊比对和 Golden 自检，输出最终状态。

状态判定矩阵:
    DUT vs Golden | Golden 自检 | 判定状态
    --------------|-------------|---------------
    PASS          | PASS        | PASS
    PASS          | FAIL        | GOLDEN_SUSPECT
    FAIL          | PASS        | DUT_ISSUE
    FAIL          | FAIL        | BOTH_SUSPECT
"""

from typing import Optional

import numpy as np

from .exact import compare_exact
from .fuzzy import compare_fuzzy
from .sanity import check_golden_sanity
from .types import (
    CompareConfig,
    CompareResult,
    CompareStatus,
    ExactResult,
    FuzzyResult,
    SanityResult,
)


class CompareEngine:
    """
    比对引擎

    使用示例:
        engine = CompareEngine()
        result = engine.compare(
            dut_output=dut,
            golden_pure=golden_fp32,
            golden_qnt=golden_qnt,
        )
        print(f"Status: {result.status.value}")
    """

    def __init__(self, config: CompareConfig = None):
        """
        Args:
            config: 比对配置
        """
        self.config = config or CompareConfig()

    def compare(
        self,
        dut_output: np.ndarray,
        golden_pure: np.ndarray,
        golden_qnt: np.ndarray = None,
        name: str = "",
        op_id: int = 0,
    ) -> CompareResult:
        """
        执行完整比对

        并行执行:
        1. 精确比对 (DUT vs Golden_pure)
        2. 模糊比对 - 纯 fp32 (DUT vs Golden_pure)
        3. 模糊比对 - 量化感知 (DUT vs Golden_qnt)
        4. Golden 自检 (Golden_qnt vs Golden_pure)

        Args:
            dut_output: DUT 输出数据
            golden_pure: 纯 fp32 Golden
            golden_qnt: 量化感知 Golden (可选，默认使用 golden_pure)
            name: 算子/比对名称
            op_id: 算子 ID

        Returns:
            CompareResult
        """
        if golden_qnt is None:
            golden_qnt = golden_pure

        # 1. 精确比对
        exact = compare_exact(
            golden_pure,
            dut_output,
            max_abs=self.config.exact_max_abs,
            max_count=self.config.exact_max_count,
        )

        # 2. 模糊比对 - 纯 fp32
        fuzzy_pure = compare_fuzzy(golden_pure, dut_output, self.config)

        # 3. 模糊比对 - 量化感知
        fuzzy_qnt = compare_fuzzy(golden_qnt, dut_output, self.config)

        # 4. Golden 自检
        sanity = check_golden_sanity(golden_pure, golden_qnt, self.config)

        # 构建结果
        result = CompareResult(
            name=name,
            op_id=op_id,
            exact=exact,
            fuzzy_pure=fuzzy_pure,
            fuzzy_qnt=fuzzy_qnt,
            sanity=sanity,
        )

        # 判定最终状态
        result.status = result.determine_status()

        return result

    def compare_exact_only(
        self,
        dut_output: np.ndarray,
        golden: np.ndarray,
        name: str = "",
    ) -> CompareResult:
        """
        仅执行精确比对 (不做 Golden 自检)

        Args:
            dut_output: DUT 输出数据
            golden: Golden 数据
            name: 比对名称

        Returns:
            CompareResult
        """
        exact = compare_exact(
            golden,
            dut_output,
            max_abs=self.config.exact_max_abs,
            max_count=self.config.exact_max_count,
        )

        result = CompareResult(name=name, exact=exact)
        result.status = (
            CompareStatus.PASS if exact.passed else CompareStatus.DUT_ISSUE
        )
        return result

    def compare_fuzzy_only(
        self,
        dut_output: np.ndarray,
        golden: np.ndarray,
        name: str = "",
    ) -> CompareResult:
        """
        仅执行模糊比对 (不做 Golden 自检)

        Args:
            dut_output: DUT 输出数据
            golden: Golden 数据
            name: 比对名称

        Returns:
            CompareResult
        """
        fuzzy = compare_fuzzy(golden, dut_output, self.config)

        result = CompareResult(name=name, fuzzy_qnt=fuzzy)
        result.status = (
            CompareStatus.PASS if fuzzy.passed else CompareStatus.DUT_ISSUE
        )
        return result


def compare_full(
    dut_output: np.ndarray,
    golden_pure: np.ndarray,
    golden_qnt: np.ndarray = None,
    config: CompareConfig = None,
    name: str = "",
) -> CompareResult:
    """
    便捷函数: 执行完整比对

    Args:
        dut_output: DUT 输出数据
        golden_pure: 纯 fp32 Golden
        golden_qnt: 量化感知 Golden
        config: 比对配置
        name: 比对名称

    Returns:
        CompareResult
    """
    engine = CompareEngine(config)
    return engine.compare(dut_output, golden_pure, golden_qnt, name=name)


def determine_status(
    exact: Optional[ExactResult],
    _fuzzy_pure: Optional[FuzzyResult],  # 保留用于未来扩展
    fuzzy_qnt: Optional[FuzzyResult],
    sanity: Optional[SanityResult],
) -> CompareStatus:
    """
    根据各项比对结果判定最终状态

    Args:
        exact: 精确比对结果
        fuzzy_pure: 模糊比对结果 (纯 fp32), 保留用于未来扩展
        fuzzy_qnt: 模糊比对结果 (量化感知)
        sanity: Golden 自检结果

    Returns:
        CompareStatus
    """
    # 判定 DUT 是否通过
    dut_passed = False
    if exact and exact.passed:
        dut_passed = True
    elif fuzzy_qnt and fuzzy_qnt.passed:
        dut_passed = True

    # 判定 Golden 是否有效
    golden_valid = True
    if sanity and not sanity.valid:
        golden_valid = False

    # 状态矩阵
    if dut_passed and golden_valid:
        return CompareStatus.PASS
    if dut_passed and not golden_valid:
        return CompareStatus.GOLDEN_SUSPECT
    if not dut_passed and golden_valid:
        return CompareStatus.DUT_ISSUE
    return CompareStatus.BOTH_SUSPECT
