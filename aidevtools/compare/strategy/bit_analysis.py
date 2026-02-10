"""
Bit 级分析策略（高级调试工具 - 带格式理解）

对浮点数组进行语义化的 bit 级分析，理解 sign/exponent/mantissa 的差异。

使用场景：
  - 硬件错误诊断（符号位翻转、指数溢出）
  - 量化算法调试（理解精度损失来源）
  - 深度误差分析

与 bit_xor 的区别：
  - bit_xor: 纯 XOR，不理解格式
  - bit_analysis: 理解格式，分析 sign/exp/mant，生成告警
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np

from .base import CompareStrategy, CompareContext


# ============================================================================
# 格式定义
# ============================================================================


class FloatFormat(Enum):
    """标准浮点格式"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


_FORMAT_LAYOUT = {
    FloatFormat.FLOAT32: (1, 8, 23),
    FloatFormat.FLOAT16: (1, 5, 10),
    FloatFormat.BFLOAT16: (1, 8, 7),
}


@dataclass
class BitLayout:
    """
    Bit 分布配置

    描述 per-element 的 bit 分布（不处理共享指数等存储细节）

    示例:
        FP32 = BitLayout(sign_bits=1, exponent_bits=8, mantissa_bits=23, name="fp32")
        BFP8 = BitLayout(sign_bits=1, exponent_bits=0, mantissa_bits=7, name="bfp8")
    """
    sign_bits: int
    exponent_bits: int
    mantissa_bits: int
    name: str = ""

    @property
    def total_bits(self) -> int:
        return self.sign_bits + self.exponent_bits + self.mantissa_bits

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.sign_bits, self.exponent_bits, self.mantissa_bits)


# 预定义格式
FP32 = BitLayout(1, 8, 23, "fp32")
FP16 = BitLayout(1, 5, 10, "fp16")
BFLOAT16 = BitLayout(1, 8, 7, "bfloat16")
BFP16 = BitLayout(1, 0, 15, "bfp16")
BFP8 = BitLayout(1, 0, 7, "bfp8")
BFP4 = BitLayout(1, 0, 3, "bfp4")


# ============================================================================
# 结果类型
# ============================================================================


class WarnLevel(Enum):
    """告警级别"""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class BitAnalysisSummary:
    """Bit 级分析汇总"""
    total_elements: int
    diff_elements: int
    sign_flip_count: int
    exponent_diff_count: int
    mantissa_diff_count: int
    max_exponent_diff: int


@dataclass
class BitAnalysisWarning:
    """单条告警"""
    level: WarnLevel
    message: str
    indices: List[int]


@dataclass
class BitAnalysisResult:
    """Bit 级分析结果"""
    fmt: Union[FloatFormat, BitLayout]
    summary: BitAnalysisSummary
    warnings: List[BitAnalysisWarning]

    @property
    def has_critical(self) -> bool:
        return any(w.level == WarnLevel.CRITICAL for w in self.warnings)


# ============================================================================
# 内部辅助函数
# ============================================================================


def _to_uint(arr: np.ndarray) -> np.ndarray:
    """转换为 uint view"""
    flat = arr.flatten()
    if flat.dtype == np.float32:
        return flat.view(np.uint32)
    elif flat.dtype == np.float16:
        return flat.view(np.uint16)
    elif flat.dtype in (np.uint8, np.uint16, np.uint32):
        return flat
    else:
        return flat.astype(np.float32).view(np.uint32)


def _get_layout_from_format(fmt: FloatFormat) -> BitLayout:
    """从 FloatFormat 获取 BitLayout"""
    s, e, m = _FORMAT_LAYOUT[fmt]
    return BitLayout(s, e, m, fmt.value)


def _generate_warnings(
    summary: BitAnalysisSummary,
    sign_diff_mask: np.ndarray,
    exp_diff_mask: np.ndarray,
    exp_diff: np.ndarray,
    diff_mask: np.ndarray,
    total_elements: int,
    max_indices: int = 10,
) -> List[BitAnalysisWarning]:
    """生成告警"""
    warnings = []

    if summary.diff_elements == 0:
        warnings.append(BitAnalysisWarning(
            level=WarnLevel.INFO,
            message="Bit-exact match",
            indices=[],
        ))
        return warnings

    # 符号位翻转 → CRITICAL
    if summary.sign_flip_count > 0:
        idx = np.where(sign_diff_mask)[0][:max_indices].tolist()
        ratio = summary.sign_flip_count / total_elements
        warnings.append(BitAnalysisWarning(
            level=WarnLevel.CRITICAL,
            message=f"Sign flip: {summary.sign_flip_count} elements ({ratio:.2%})",
            indices=idx,
        ))

    # 大指数偏移 (>=2) → CRITICAL
    if summary.exponent_diff_count > 0 and summary.max_exponent_diff >= 2:
        large_shift = np.abs(exp_diff) >= 2
        count = int(np.sum(large_shift))
        if count > 0:
            idx = np.where(large_shift)[0][:max_indices].tolist()
            warnings.append(BitAnalysisWarning(
                level=WarnLevel.CRITICAL,
                message=f"Large exponent shift (>=2): {count} elements, max={summary.max_exponent_diff}",
                indices=idx,
            ))

    # 小指数偏移 (±1) → WARNING
    if summary.exponent_diff_count > 0:
        small_shift = (np.abs(exp_diff) == 1)
        count = int(np.sum(small_shift))
        if count > 0:
            idx = np.where(small_shift)[0][:max_indices].tolist()
            warnings.append(BitAnalysisWarning(
                level=WarnLevel.WARNING,
                message=f"Small exponent shift (±1): {count} elements",
                indices=idx,
            ))

    # 仅尾数差异 → INFO
    mant_only = diff_mask & ~sign_diff_mask & ~exp_diff_mask
    mant_only_count = int(np.sum(mant_only))
    if mant_only_count > 0:
        idx = np.where(mant_only)[0][:max_indices].tolist()
        warnings.append(BitAnalysisWarning(
            level=WarnLevel.INFO,
            message=f"Mantissa-only diff: {mant_only_count} elements (precision loss)",
            indices=idx,
        ))

    return warnings


def _compare_bit_analysis_impl(
    golden: np.ndarray,
    result: np.ndarray,
    fmt: Union[FloatFormat, BitLayout],
    max_warning_indices: int = 10,
) -> BitAnalysisResult:
    """
    Bit 级分析（带格式理解）

    Args:
        golden: 参考数据
        result: 待比对数据
        fmt: 格式描述
        max_warning_indices: 每条告警记录的最大索引数

    Returns:
        BitAnalysisResult
    """
    # 解析格式
    if isinstance(fmt, FloatFormat):
        sign_bits, exp_bits, mant_bits = _FORMAT_LAYOUT[fmt]
    else:
        sign_bits, exp_bits, mant_bits = fmt.as_tuple()

    total_bits = sign_bits + exp_bits + mant_bits

    g_uint = _to_uint(golden)
    r_uint = _to_uint(result)
    total_elements = len(g_uint)

    if len(g_uint) != len(r_uint):
        raise ValueError(f"Shape mismatch: {golden.shape} vs {result.shape}")

    # XOR
    xor = g_uint ^ r_uint
    diff_mask = xor != 0
    diff_elements = int(np.count_nonzero(diff_mask))

    # Sign bit
    if sign_bits > 0:
        sign_shift = total_bits - 1
        if g_uint.dtype == np.uint32:
            sign_diff_mask = (xor >> np.uint32(sign_shift)) & np.uint32(1)
        elif g_uint.dtype == np.uint16:
            sign_diff_mask = (xor >> np.uint16(sign_shift)) & np.uint16(1)
        else:
            sign_diff_mask = (xor >> sign_shift) & 1
        sign_flip_count = int(np.count_nonzero(sign_diff_mask))
    else:
        sign_diff_mask = np.zeros(total_elements, dtype=np.uint8)
        sign_flip_count = 0

    # Exponent bits
    if exp_bits > 0:
        exp_shift = mant_bits
        exp_mask_val = (1 << exp_bits) - 1
        g_exp = (g_uint >> exp_shift) & exp_mask_val
        r_exp = (r_uint >> exp_shift) & exp_mask_val
        exp_diff = g_exp.astype(np.int32) - r_exp.astype(np.int32)
        exp_diff_mask = exp_diff != 0
        exponent_diff_count = int(np.count_nonzero(exp_diff_mask))
        max_exponent_diff = int(np.max(np.abs(exp_diff))) if exponent_diff_count > 0 else 0
    else:
        exp_diff = np.zeros(total_elements, dtype=np.int32)
        exp_diff_mask = np.zeros(total_elements, dtype=bool)
        exponent_diff_count = 0
        max_exponent_diff = 0

    # Mantissa bits
    if mant_bits > 0:
        mant_mask_val = (1 << mant_bits) - 1
        g_mant = g_uint & mant_mask_val
        r_mant = r_uint & mant_mask_val
        mant_diff_mask = g_mant != r_mant
        mantissa_diff_count = int(np.count_nonzero(mant_diff_mask))
    else:
        mant_diff_mask = np.zeros(total_elements, dtype=bool)
        mantissa_diff_count = 0

    summary = BitAnalysisSummary(
        total_elements=total_elements,
        diff_elements=diff_elements,
        sign_flip_count=sign_flip_count,
        exponent_diff_count=exponent_diff_count,
        mantissa_diff_count=mantissa_diff_count,
        max_exponent_diff=max_exponent_diff,
    )

    warnings = _generate_warnings(
        summary, sign_diff_mask, exp_diff_mask, exp_diff,
        diff_mask, total_elements, max_warning_indices,
    )

    return BitAnalysisResult(fmt=fmt, summary=summary, warnings=warnings)


def _compare_as_double_impl(
    golden: np.ndarray,
    result: np.ndarray,
) -> dict:
    """
    转为 double 精度比对

    将数据转为 float64，计算绝对误差和相对误差。

    Returns:
        {
            "max_abs": float,
            "mean_abs": float,
            "max_rel": float,
            "mean_rel": float,
        }
    """
    g_f64 = golden.astype(np.float64).flatten()
    r_f64 = result.astype(np.float64).flatten()

    abs_err = np.abs(g_f64 - r_f64)
    max_abs = float(np.max(abs_err))
    mean_abs = float(np.mean(abs_err))

    # 相对误差（避免除零）
    g_abs = np.abs(g_f64)
    mask = g_abs > 1e-12
    if np.any(mask):
        rel_err = abs_err[mask] / g_abs[mask]
        max_rel = float(np.max(rel_err))
        mean_rel = float(np.mean(rel_err))
    else:
        max_rel = 0.0
        mean_rel = 0.0

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
    }


# ============================================================================
# 输出函数
# ============================================================================


def _print_bit_analysis_result_impl(result: BitAnalysisResult, name: str = ""):
    """打印 bit 级分析结果"""
    s = result.summary
    fmt_name = result.fmt.value if isinstance(result.fmt, FloatFormat) else result.fmt.name

    title = f"[{name}] Bit Analysis ({fmt_name})" if name else f"Bit Analysis ({fmt_name})"
    print(f"\n{title}")
    print("=" * 60)
    print(f"  Total elements:    {s.total_elements:,}")
    print(f"  Diff elements:     {s.diff_elements:,}")
    print(f"  Sign flips:        {s.sign_flip_count:,}")
    print(f"  Exponent diffs:    {s.exponent_diff_count:,} (max shift={s.max_exponent_diff})")
    print(f"  Mantissa diffs:    {s.mantissa_diff_count:,}")
    print("-" * 60)

    for w in result.warnings:
        prefix = {
            WarnLevel.INFO: "  [.]",
            WarnLevel.WARNING: "  [~]",
            WarnLevel.CRITICAL: "  [!]",
        }[w.level]
        print(f"{prefix} {w.message}")
        if w.indices:
            idx_str = ", ".join(str(i) for i in w.indices[:10])
            print(f"       indices: [{idx_str}]")

    print("=" * 60)
    print()


# ============================================================================
# Strategy 类
# ============================================================================


class BitAnalysisStrategy(CompareStrategy):
    """
    Bit 级分析策略（高级调试工具 - 可选）

    对数据进行 bit 级分析，理解格式，诊断硬件错误、量化问题。

    注意：这是可选的高级调试工具，不包含在标准策略中。
    适用于硬件错误诊断、量化算法调试等深度分析场景。

    使用方式：
        # 方式1: 直接调用静态方法
        result = BitAnalysisStrategy.compare(golden, dut, fmt=FP32)

        # 方式2: 通过引擎
        engine = CompareEngine(BitAnalysisStrategy(format=FP16))
        result = engine.run(dut, golden)
    """

    def __init__(
        self,
        format: Optional[Union[FloatFormat, BitLayout]] = None,
    ):
        """
        Args:
            format: 浮点格式（FP32/FP16/BFP8等）
        """
        self.format = format or FP32

    @staticmethod
    def compare(
        golden: np.ndarray,
        result: np.ndarray,
        fmt: Union[FloatFormat, BitLayout],
        max_warning_indices: int = 10,
    ) -> BitAnalysisResult:
        """
        Bit 级分析（静态方法）

        带格式理解的 bit 级分析。

        Args:
            golden: 参考数据
            result: 待比对数据
            fmt: 格式描述
            max_warning_indices: 每条告警记录的最大索引数

        Returns:
            BitAnalysisResult
        """
        return _compare_bit_analysis_impl(golden, result, fmt, max_warning_indices)

    @staticmethod
    def compare_as_double(
        golden: np.ndarray,
        result: np.ndarray,
    ) -> dict:
        """
        转为 double 精度比对（静态方法）

        将数据转为 float64，计算绝对误差和相对误差。

        Returns:
            {
                "max_abs": float,
                "mean_abs": float,
                "max_rel": float,
                "mean_rel": float,
            }
        """
        return _compare_as_double_impl(golden, result)

    @staticmethod
    def print_result(result: BitAnalysisResult, name: str = ""):
        """打印 bit 级分析结果（静态方法）"""
        return _print_bit_analysis_result_impl(result, name)

    def run(self, ctx: CompareContext) -> BitAnalysisResult:
        """执行 bit 级分析（Strategy 协议方法）"""
        return self.compare(ctx.golden, ctx.dut, fmt=self.format)

    @staticmethod
    def visualize(result: BitAnalysisResult) -> "Page":
        """
        BitAnalysis 策略级可视化

        体现比对原理：bit 语义分析
        - 展示 sign/exponent/mantissa 错误分类
        - 突出格式理解
        - 严重度分级
        """
        from aidevtools.compare.visualizer import Visualizer

        page = Visualizer.create_page(title="Bit Analysis Report")

        # 1. 错误类型分布饼图（体现语义分析）
        no_diff = result.summary.total_elements - result.summary.diff_elements
        error_data = {
            "✅ No Diff": no_diff,
            "🟡 Mantissa Only": result.summary.mantissa_diff_count,
            "🟠 Exponent Diff": result.summary.exponent_diff_count,
            "🔴 Sign Flip": result.summary.sign_flip_count,
        }

        fmt_name = result.fmt.value if isinstance(result.fmt, FloatFormat) else result.fmt.name
        pie = Visualizer.create_pie(
            error_data,
            title=f"Error Type Distribution ({fmt_name})",
        )
        page.add(pie)

        # 2. Bit 布局柱状图（体现格式理解）
        layout = result.fmt if isinstance(result.fmt, BitLayout) else _get_layout_from_format(result.fmt)
        x_data = [
            f"Sign (b{layout.total_bits-1})",
            f"Exponent (b{layout.total_bits-2}:b{layout.mantissa_bits})",
            f"Mantissa (b{layout.mantissa_bits-1}:b0)",
        ]
        y_data = {
            "Error Count": [
                result.summary.sign_flip_count,
                result.summary.exponent_diff_count,
                result.summary.mantissa_diff_count,
            ]
        }
        bar = Visualizer.create_bar(x_data, y_data, title=f"Bit Layout Analysis ({fmt_name})")
        page.add(bar)

        # 3. 告警摘要（体现严重度分级）
        if result.warnings:
            critical = sum(1 for w in result.warnings if w.level == WarnLevel.CRITICAL)
            warning = sum(1 for w in result.warnings if w.level == WarnLevel.WARNING)
            info = sum(1 for w in result.warnings if w.level == WarnLevel.INFO)

            warn_data = {"🔴 CRITICAL": critical, "🟠 WARNING": warning, "🟡 INFO": info}
            warn_bar = Visualizer.create_bar(
                list(warn_data.keys()),
                {"Count": list(warn_data.values())},
                title="Warning Summary",
            )
            page.add(warn_bar)

        return page

    @property
    def name(self) -> str:
        if isinstance(self.format, FloatFormat):
            return f"bit_analysis_{self.format.value}"
        else:
            return f"bit_analysis_{self.format.name}"


__all__ = [
    "FloatFormat",
    "BitLayout",
    "FP32",
    "FP16",
    "BFLOAT16",
    "BFP16",
    "BFP8",
    "BFP4",
    "WarnLevel",
    "BitAnalysisSummary",
    "BitAnalysisWarning",
    "BitAnalysisResult",
    "BitAnalysisStrategy",
]
