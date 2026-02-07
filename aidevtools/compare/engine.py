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

架构优化:
- 使用 _PreparedPair 预处理一次数据，exact/fuzzy 共享中间结果
- golden_qnt=None 时自动短路，避免重复计算和无意义的 QSNR 自检
- determine_status() 委托给 CompareResult.determine_status()，消除逻辑重复
"""

from typing import Optional

import numpy as np

from .blocked import compare_blocked
from .bitwise import compare_bitwise
from .exact import _compare_exact_prepared
from .fuzzy import _compare_fuzzy_prepared
from .sanity import check_golden_sanity
from .types import (
    CompareConfig,
    CompareResult,
    CompareStatus,
    ExactResult,
    FuzzyResult,
    SanityResult,
    _PreparedPair,
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

        步骤:
        1. 精确比对 (DUT vs Golden_pure)
        2. 模糊比对 - 纯 fp32 (DUT vs Golden_pure)
        3. 模糊比对 - 量化感知 (DUT vs Golden_qnt)
        4. Golden 自检 (Golden_qnt vs Golden_pure)
        5. Bit 级分析 (可选, enable_bitwise=True)
        6. 分块比对 (可选, enable_blocked=True)

        优化:
        - _PreparedPair 预处理一次，exact/fuzzy 共享 diff/abs_err/g_abs
        - golden_qnt=None 时复用 fuzzy_pure 结果，跳过无意义的 QSNR 自检

        Args:
            dut_output: DUT 输出数据
            golden_pure: 纯 fp32 Golden
            golden_qnt: 量化感知 Golden (可选，默认使用 golden_pure)
            name: 算子/比对名称
            op_id: 算子 ID

        Returns:
            CompareResult
        """
        has_separate_qnt = golden_qnt is not None

        # 预处理一次: golden_pure vs dut_output
        prep_pure = _PreparedPair.from_arrays(golden_pure, dut_output)

        # 1. 精确比对 — 复用 prep_pure.abs_err
        exact = _compare_exact_prepared(
            prep_pure,
            golden_pure,
            dut_output,
            max_abs=self.config.exact_max_abs,
            max_count=self.config.exact_max_count,
        )

        # 2. 模糊比对 - 纯 fp32 — 复用 prep_pure 全部中间结果
        fuzzy_pure = _compare_fuzzy_prepared(prep_pure, self.config)

        # 3. 模糊比对 - 量化感知
        if has_separate_qnt:
            prep_qnt = _PreparedPair.from_arrays(golden_qnt, dut_output)
            fuzzy_qnt = _compare_fuzzy_prepared(prep_qnt, self.config)
        else:
            # golden_qnt=None: 复用 fuzzy_pure 结果，避免重复计算
            fuzzy_qnt = fuzzy_pure

        # 4. Golden 自检
        if has_separate_qnt:
            sanity = check_golden_sanity(golden_pure, golden_qnt, self.config)
        else:
            # 跳过无意义的 golden_pure vs golden_pure QSNR 检查
            sanity = check_golden_sanity(golden_pure, None, self.config)

        # 5. Bit 级分析 (可选)
        bitwise_result = None
        if self.config.enable_bitwise:
            bitwise_result = compare_bitwise(
                golden_pure, dut_output, fmt=self.config.bitwise_fmt,
            )

        # 6. 分块比对 (可选)
        blocked_result = None
        if self.config.enable_blocked:
            blocked_result = compare_blocked(
                golden_pure, dut_output,
                block_size=self.config.blocked_block_size,
                min_qsnr=self.config.fuzzy_min_qsnr,
                min_cosine=self.config.fuzzy_min_cosine,
            )

        # 构建结果 + 判定最终状态
        result = CompareResult(
            name=name,
            op_id=op_id,
            exact=exact,
            fuzzy_pure=fuzzy_pure,
            fuzzy_qnt=fuzzy_qnt,
            sanity=sanity,
            bitwise=bitwise_result,
            blocked=blocked_result,
        )
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
        p = _PreparedPair.from_arrays(golden, dut_output)
        exact = _compare_exact_prepared(
            p,
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
        p = _PreparedPair.from_arrays(golden, dut_output)
        fuzzy = _compare_fuzzy_prepared(p, self.config)

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
    _fuzzy_pure: Optional[FuzzyResult],
    fuzzy_qnt: Optional[FuzzyResult],
    sanity: Optional[SanityResult],
) -> CompareStatus:
    """
    根据各项比对结果判定最终状态

    委托给 CompareResult.determine_status()，消除逻辑重复。

    Args:
        exact: 精确比对结果
        _fuzzy_pure: 模糊比对结果 (纯 fp32), 保留用于未来扩展
        fuzzy_qnt: 模糊比对结果 (量化感知)
        sanity: Golden 自检结果

    Returns:
        CompareStatus
    """
    tmp = CompareResult(
        exact=exact, fuzzy_pure=_fuzzy_pure,
        fuzzy_qnt=fuzzy_qnt, sanity=sanity,
    )
    return tmp.determine_status()
