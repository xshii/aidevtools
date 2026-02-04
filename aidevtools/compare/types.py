"""
比对结果类型定义

状态判定矩阵:
    DUT vs Golden | Golden 自检 | 判定状态
    --------------|-------------|---------------
    PASS          | PASS        | PASS
    PASS          | FAIL        | GOLDEN_SUSPECT
    FAIL          | PASS        | DUT_ISSUE
    FAIL          | FAIL        | BOTH_SUSPECT
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CompareStatus(Enum):
    """比对状态"""

    PASS = "PASS"  # DUT 匹配 Golden，且 Golden 有效
    GOLDEN_SUSPECT = "GOLDEN_SUSPECT"  # DUT 匹配，但 Golden 自检失败
    DUT_ISSUE = "DUT_ISSUE"  # Golden 有效，但 DUT 不匹配
    BOTH_SUSPECT = "BOTH_SUSPECT"  # 都可疑，需人工排查


@dataclass
class ExactResult:
    """精确比对结果"""

    passed: bool
    mismatch_count: int
    first_diff_offset: int  # -1 表示无差异
    max_abs: float
    total_elements: int = 0


@dataclass
class FuzzyResult:
    """模糊比对结果"""

    passed: bool
    max_abs: float
    mean_abs: float
    max_rel: float
    qsnr: float
    cosine: float
    total_elements: int
    exceed_count: int


@dataclass
class SanityResult:
    """Golden 自检结果"""

    valid: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

    # 详细检查项
    non_zero: bool = True
    no_nan_inf: bool = True
    range_valid: bool = True
    qsnr_valid: bool = True  # golden_qnt vs golden_pure QSNR >= 阈值


@dataclass
class CompareResult:
    """
    完整比对结果

    包含:
    - 精确比对结果 (DUT vs Golden)
    - 模糊比对结果 - 纯 fp32 (DUT vs Golden_pure)
    - 模糊比对结果 - 量化感知 (DUT vs Golden_qnt)
    - Golden 自检结果 (Golden_qnt vs Golden_pure)
    - 最终状态判定
    """

    # 标识
    name: str = ""
    op_id: int = 0

    # 三列比对结果
    exact: Optional[ExactResult] = None
    fuzzy_pure: Optional[FuzzyResult] = None
    fuzzy_qnt: Optional[FuzzyResult] = None

    # Golden 自检
    sanity: Optional[SanityResult] = None

    # 最终状态
    status: CompareStatus = CompareStatus.DUT_ISSUE

    @property
    def dut_passed(self) -> bool:
        """DUT 是否通过 (精确或模糊任一通过)"""
        if self.exact and self.exact.passed:
            return True
        if self.fuzzy_qnt and self.fuzzy_qnt.passed:
            return True
        return False

    @property
    def golden_valid(self) -> bool:
        """Golden 是否有效"""
        if self.sanity is None:
            return True
        return self.sanity.valid

    def determine_status(self) -> CompareStatus:
        """根据比对结果判定状态"""
        dut_ok = self.dut_passed
        golden_ok = self.golden_valid

        if dut_ok and golden_ok:
            return CompareStatus.PASS
        if dut_ok and not golden_ok:
            return CompareStatus.GOLDEN_SUSPECT
        if not dut_ok and golden_ok:
            return CompareStatus.DUT_ISSUE
        return CompareStatus.BOTH_SUSPECT


@dataclass
class CompareConfig:
    """比对配置"""

    # 精确比对阈值
    exact_max_abs: float = 0.0
    exact_max_count: int = 0

    # 模糊比对阈值
    fuzzy_atol: float = 1e-5
    fuzzy_rtol: float = 1e-3
    fuzzy_min_qsnr: float = 30.0
    fuzzy_min_cosine: float = 0.999
    fuzzy_max_exceed_ratio: float = 0.0

    # Golden 自检阈值
    sanity_min_qsnr: float = 20.0  # golden_qnt vs golden_pure
    sanity_max_nan_ratio: float = 0.0
    sanity_max_inf_ratio: float = 0.0
    sanity_min_nonzero_ratio: float = 0.01
