"""
比对结果类型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ExactResult:
    """精确比对结果（含 bit 级统计）"""

    passed: bool
    mismatch_count: int
    first_diff_offset: int  # -1 表示无差异
    max_abs: float
    total_elements: int = 0
    diff_bits: int = 0      # 差异 bit 数 (popcount)
    total_bits: int = 0     # 总 bit 数

    @property
    def diff_bit_ratio(self) -> float:
        """差异 bit 比例"""
        return self.diff_bits / self.total_bits if self.total_bits > 0 else 0.0


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

    non_zero: bool = True
    no_nan_inf: bool = True
    range_valid: bool = True
    qsnr_valid: bool = True


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
    sanity_min_qsnr: float = 20.0
    sanity_max_nan_ratio: float = 0.0
    sanity_max_inf_ratio: float = 0.0
    sanity_min_nonzero_ratio: float = 0.01


@dataclass
class _PreparedPair:
    """数据预处理缓存 — 消除跨模块重复转换"""

    g: np.ndarray        # float64, flattened golden
    r: np.ndarray        # float64, flattened result
    diff: np.ndarray     # g - r
    abs_err: np.ndarray  # |diff|
    g_abs: np.ndarray    # |g|
    total: int

    @classmethod
    def from_arrays(cls, golden: np.ndarray, result: np.ndarray) -> "_PreparedPair":
        g = golden.astype(np.float64).flatten()
        r = result.astype(np.float64).flatten()
        diff = g - r
        return cls(
            g=g, r=r, diff=diff,
            abs_err=np.abs(diff), g_abs=np.abs(g),
            total=len(g),
        )
