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

可视化功能：
  - print_bit_analysis: 打印分析结果
  - print_bit_template: 打印 bit 模板
  - print_bit_heatmap: 文本热力图
  - gen_bit_heatmap_svg: SVG 热力图
  - gen_perbit_bar_svg: per-bit 条形图
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

    描述 per-element 的 bit 分布（支持共享指数等 BFP 格式）

    示例:
        FP32 = BitLayout(sign_bits=1, exponent_bits=8, mantissa_bits=23, name="fp32")
        BFP8 = BitLayout(sign_bits=1, exponent_bits=0, mantissa_bits=7, name="bfp8",
                         shared_exponent_bits=8, block_size=16)

    统一模板语法:
        layout = BitLayout(template="EEEEEEEE(SMMMMMMM)*16", name="bfp8")
    """
    sign_bits: int = 0
    exponent_bits: int = 0
    mantissa_bits: int = 0
    name: str = ""
    shared_exponent_bits: int = 0
    block_size: int = 1
    template: str = ""

    def __post_init__(self):
        if self.template:
            self._parse_unified_template(self.template)
        self._build_raw_templates()

    def _parse_unified_template(self, tmpl: str):
        """解析统一模板: EEEEEEEE(SMMMMMMM)*16 或 SEEEEMMMM"""
        m = re.match(r'^([^(]*)\(([^)]+)\)\*(\d+)$', tmpl)
        if m:
            shared_part = m.group(1)
            element_part = m.group(2)
            bs = int(m.group(3))
            self.shared_exponent_bits = sum(1 for c in shared_part if c.upper() == 'E')
            self.block_size = bs
            self.sign_bits = sum(1 for c in element_part if c.upper() == 'S')
            self.exponent_bits = sum(1 for c in element_part if c.upper() == 'E')
            self.mantissa_bits = sum(
                1 for c in element_part if c.upper() not in ('S', 'E')
            )
        else:
            self.sign_bits = sum(1 for c in tmpl if c.upper() == 'S')
            self.exponent_bits = sum(1 for c in tmpl if c.upper() == 'E')
            self.mantissa_bits = sum(
                1 for c in tmpl if c.upper() not in ('S', 'E')
            )
            self.shared_exponent_bits = 0
            self.block_size = 1

    def _build_raw_templates(self):
        """从 bit 数或 template 字符串构建内部模板"""
        if self.template:
            m = re.match(r'^([^(]*)\(([^)]+)\)\*(\d+)$', self.template)
            if m:
                self._raw_template = m.group(2)
                self._shared_raw = m.group(1)
            else:
                self._raw_template = self.template
                self._shared_raw = ""
        else:
            parts = []
            if self.sign_bits > 0:
                parts.append('S' * self.sign_bits)
            if self.exponent_bits > 0:
                parts.append('E' * self.exponent_bits)
            if self.mantissa_bits > 0:
                if self.sign_bits == 0 and self.exponent_bits == 0 and self.name and 'int' in self.name.lower():
                    parts.append('I' * self.mantissa_bits)
                else:
                    parts.append('M' * self.mantissa_bits)
            self._raw_template = ''.join(parts)
            self._shared_raw = 'E' * self.shared_exponent_bits if self.shared_exponent_bits > 0 else ""

    @property
    def total_bits(self) -> int:
        return self.sign_bits + self.exponent_bits + self.mantissa_bits

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.sign_bits, self.exponent_bits, self.mantissa_bits)

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        parts = []
        if self.sign_bits > 0:
            parts.append(f"s{self.sign_bits}")
        if self.exponent_bits > 0:
            parts.append(f"e{self.exponent_bits}")
        if self.mantissa_bits > 0:
            parts.append(f"m{self.mantissa_bits}")
        base = "_".join(parts)
        if self.shared_exponent_bits > 0:
            base += "_shared_exp"
        return base

    @property
    def bit_template(self) -> str:
        return self._raw_template

    @property
    def bit_template_spaced(self) -> str:
        tmpl = self._raw_template
        if not tmpl:
            return ""
        groups = []
        current_char = tmpl[0].upper()
        current_group = tmpl[0]
        for c in tmpl[1:]:
            if c.upper() == current_char:
                current_group += c
            else:
                groups.append(current_group)
                current_char = c.upper()
                current_group = c
        groups.append(current_group)
        return ' '.join(groups)

    @property
    def shared_template(self) -> str:
        return self._shared_raw

    @property
    def bit_group_labels(self) -> Dict[str, str]:
        labels = {}
        for c in self._raw_template + self._shared_raw:
            cu = c.upper()
            if cu == 'S':
                labels['S'] = 'sign'
            elif cu == 'E':
                labels['E'] = 'exponent'
            elif cu == 'M':
                labels['M'] = 'mantissa'
            elif cu == 'I':
                labels['I'] = 'integer'
            elif cu == 'F':
                labels['F'] = 'flag'
            elif cu == 'D':
                labels['D'] = 'data'
            elif cu == 'P':
                labels['P'] = 'parity'
        return labels

    @classmethod
    def from_template(
        cls,
        template: str,
        name: str = "",
        shared_template: str = "",
        block_size: int = 1,
    ) -> "BitLayout":
        """从模板字符串创建 BitLayout"""
        sign = sum(1 for c in template if c.upper() == 'S')
        exp = sum(1 for c in template if c.upper() == 'E')
        mant = sum(1 for c in template if c.upper() not in ('S', 'E'))
        shared_exp = sum(1 for c in shared_template if c.upper() == 'E') if shared_template else 0

        layout = cls.__new__(cls)
        layout.sign_bits = sign
        layout.exponent_bits = exp
        layout.mantissa_bits = mant
        layout.name = name
        layout.shared_exponent_bits = shared_exp
        layout.block_size = block_size
        layout.template = ""
        layout._raw_template = template
        layout._shared_raw = shared_template
        return layout


# 预定义格式
FP32 = BitLayout(1, 8, 23, "fp32")
FP16 = BitLayout(1, 5, 10, "fp16")
BFLOAT16 = BitLayout(1, 8, 7, "bfloat16")
BFP16 = BitLayout(1, 0, 15, "bfp16", shared_exponent_bits=8, block_size=16)
BFP8 = BitLayout(1, 0, 7, "bfp8", shared_exponent_bits=8, block_size=16)
BFP4 = BitLayout(1, 0, 3, "bfp4", shared_exponent_bits=8, block_size=16)
INT8 = BitLayout(1, 0, 7, "int8")
UINT8 = BitLayout(0, 0, 8, "uint8")


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
    per_bit_error_count: List[int] = field(default_factory=list)

    @property
    def diff_ratio(self) -> float:
        if self.total_elements == 0:
            return 0.0
        return self.diff_elements / self.total_elements


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

    @property
    def format_name(self) -> str:
        if isinstance(self.fmt, FloatFormat):
            return self.fmt.value
        return self.fmt.display_name


@dataclass
class ModelBitAnalysis:
    """模型级 bit 分析结果"""
    per_op: Dict[str, BitAnalysisResult]
    global_result: Optional[BitAnalysisResult] = None

    @property
    def has_critical(self) -> bool:
        for r in self.per_op.values():
            if r.has_critical:
                return True
        if self.global_result and self.global_result.has_critical:
            return True
        return False


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
            message=f"符号位翻转: {summary.sign_flip_count} elements ({ratio:.2%})",
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
                message=f"指数域大偏移 (>=2): {count} elements, max={summary.max_exponent_diff}",
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
                message=f"指数域偏移 (±1): {count} elements",
                indices=idx,
            ))

    # 仅尾数差异 → INFO
    mant_only = diff_mask & ~sign_diff_mask.astype(bool) & ~exp_diff_mask
    mant_only_count = int(np.sum(mant_only))
    if mant_only_count > 0:
        idx = np.where(mant_only)[0][:max_indices].tolist()
        warnings.append(BitAnalysisWarning(
            level=WarnLevel.INFO,
            message=f"仅尾数差异: {mant_only_count} elements (precision loss)",
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

    # BFLOAT16 on float32 data: extract upper 16 bits
    if isinstance(fmt, FloatFormat) and fmt == FloatFormat.BFLOAT16 and g_uint.dtype == np.uint32:
        g_uint = (g_uint >> np.uint32(16)).astype(np.uint16)
        r_uint = (r_uint >> np.uint32(16)).astype(np.uint16)

    total_elements = len(g_uint)

    if len(g_uint) != len(r_uint):
        raise ValueError(f"Shape mismatch: {golden.shape} vs {result.shape}")

    # XOR
    xor = g_uint ^ r_uint
    diff_mask = xor != 0
    diff_elements = int(np.count_nonzero(diff_mask))

    # Per-bit error count
    per_bit = []
    for bit in range(total_bits):
        if g_uint.dtype == np.uint32:
            bit_mask = np.uint32(1 << bit)
        elif g_uint.dtype == np.uint16:
            bit_mask = np.uint16(1 << bit)
        elif g_uint.dtype == np.uint8:
            bit_mask = np.uint8(1 << bit)
        else:
            bit_mask = 1 << bit
        per_bit.append(int(np.count_nonzero(xor & bit_mask)))

    # Sign bit
    if sign_bits > 0:
        sign_shift = total_bits - 1
        if g_uint.dtype == np.uint32:
            sign_diff_mask = (xor >> np.uint32(sign_shift)) & np.uint32(1)
        elif g_uint.dtype == np.uint16:
            sign_diff_mask = (xor >> np.uint16(sign_shift)) & np.uint16(1)
        elif g_uint.dtype == np.uint8:
            sign_diff_mask = (xor >> np.uint8(sign_shift)) & np.uint8(1)
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
        per_bit_error_count=per_bit,
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
    """转为 double 精度比对"""
    g_f64 = golden.astype(np.float64).flatten()
    r_f64 = result.astype(np.float64).flatten()

    abs_err = np.abs(g_f64 - r_f64)
    max_abs = float(np.max(abs_err))
    mean_abs = float(np.mean(abs_err))

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
# 输出/可视化函数
# ============================================================================


def _get_format_info(fmt: Union[FloatFormat, BitLayout]) -> Tuple[str, int, "BitLayout"]:
    """获取格式的名称、总位数和 BitLayout"""
    if isinstance(fmt, FloatFormat):
        s, e, m = _FORMAT_LAYOUT[fmt]
        layout = BitLayout(s, e, m, fmt.value)
        return fmt.value, s + e + m, layout
    return fmt.display_name, fmt.total_bits, fmt


def print_bit_analysis(result: BitAnalysisResult, name: str = ""):
    """打印 bit 级分析结果"""
    s = result.summary
    fmt_name, total_bits, layout = _get_format_info(result.fmt)

    title = f"[{name}] Bit-Level Analysis ({fmt_name})" if name else f"Bit-Level Analysis ({fmt_name})"
    print(f"\n{title}")
    print("=" * 60)

    # Bit layout template
    if isinstance(result.fmt, BitLayout) or isinstance(result.fmt, FloatFormat):
        spaced = layout.bit_template_spaced
        if spaced:
            print(f"  Bit layout: {spaced}")
            legend = []
            labels = layout.bit_group_labels
            for letter, label in sorted(labels.items()):
                legend.append(f"{letter}={label}")
            if legend:
                print(f"  Legend:     {', '.join(legend)}")

    print(f"  Format:        {fmt_name} ({total_bits} bits)")
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


def print_bit_template(fmt: Union[FloatFormat, BitLayout]):
    """打印 bit 模板"""
    fmt_name, total_bits, layout = _get_format_info(fmt)

    print(f"\n{fmt_name} ({total_bits} bits)")
    print("=" * 40)
    print(f"  {layout.bit_template_spaced}")

    # Legend
    labels = layout.bit_group_labels
    for letter, label in sorted(labels.items()):
        print(f"  {letter}={label}")

    # Shared exponent info
    if layout.shared_exponent_bits > 0:
        print(f"  Shared exponent: {layout._shared_raw} ({layout.shared_exponent_bits} bits)")
        print(f"  Shared across block of {layout.block_size} elements (shared_exponent)")

    print()


def print_bit_heatmap(
    golden: np.ndarray,
    result: np.ndarray,
    fmt: Union[FloatFormat, BitLayout] = FP32,
    block_size: int = 256,
    cols: int = 10,
):
    """打印文本热力图"""
    g_uint = _to_uint(golden)
    r_uint = _to_uint(result)
    total = len(g_uint)
    n_blocks = (total + block_size - 1) // block_size

    print(f"\nBit Heatmap (block_size={block_size}, {n_blocks} blocks)")
    print("=" * 60)
    print("  '.' = exact  '-' = low diff  '#' = heavy diff")
    print()

    line = "  "
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, total)
        block_xor = g_uint[start:end] ^ r_uint[start:end]
        diff_count = int(np.count_nonzero(block_xor))

        if diff_count == 0:
            line += "."
        elif diff_count < (end - start) * 0.1:
            line += "-"
        else:
            line += "#"

        if (i + 1) % cols == 0:
            print(line)
            line = "  "

    if len(line.strip()) > 0:
        print(line)

    print()


def gen_bit_heatmap_svg(
    golden: np.ndarray,
    result: np.ndarray,
    output_path: str,
    fmt: Union[FloatFormat, BitLayout] = FP32,
    block_size: int = 256,
    cols: int = 32,
    cell_size: int = 12,
):
    """生成 SVG 热力图"""
    g_uint = _to_uint(golden)
    r_uint = _to_uint(result)
    total = len(g_uint)
    n_blocks = (total + block_size - 1) // block_size

    rows = (n_blocks + cols - 1) // cols
    width = cols * cell_size + 40
    height = rows * cell_size + 60

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="20" font-size="14">Bit Heatmap ({n_blocks} blocks)</text>',
    ]

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, total)
        block_xor = g_uint[start:end] ^ r_uint[start:end]
        diff_ratio = int(np.count_nonzero(block_xor)) / (end - start)

        col = i % cols
        row = i // cols
        x = col * cell_size + 20
        y = row * cell_size + 30

        if diff_ratio == 0:
            color = "#00cc00"
        elif diff_ratio < 0.1:
            color = "#ffcc00"
        else:
            r_val = min(255, int(diff_ratio * 255))
            color = f"#{r_val:02x}0000"

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{cell_size-1}" height="{cell_size-1}" '
            f'fill="{color}" title="block {i}: {diff_ratio:.2%}"/>'
        )

    svg_parts.append('</svg>')
    Path(output_path).write_text('\n'.join(svg_parts))


def gen_perbit_bar_svg(
    result: BitAnalysisResult,
    output_path: str,
    bar_width: int = 20,
    bar_max_height: int = 150,
):
    """生成 per-bit 条形图 SVG"""
    fmt_name, total_bits, layout = _get_format_info(result.fmt)
    per_bit = result.summary.per_bit_error_count

    if not per_bit:
        per_bit = [0] * total_bits

    max_count = max(per_bit) if per_bit else 1
    if max_count == 0:
        max_count = 1

    width = total_bits * (bar_width + 4) + 80
    height = bar_max_height + 120

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="10" y="20" font-size="14">Per-Bit Error Distribution ({fmt_name})</text>',
    ]

    # Determine bit regions from the template
    tmpl = layout.bit_template
    sign_bits = layout.sign_bits
    exp_bits = layout.exponent_bits

    for bit_idx in range(total_bits):
        count = per_bit[bit_idx] if bit_idx < len(per_bit) else 0
        bar_h = int((count / max_count) * bar_max_height) if max_count > 0 else 0

        x = bit_idx * (bar_width + 4) + 40
        y = bar_max_height - bar_h + 40

        # Color based on bit region (MSB first in template)
        bit_pos_from_msb = total_bits - 1 - bit_idx
        if bit_pos_from_msb >= (total_bits - sign_bits):
            color = "#cc0000"  # sign
            region = "sign"
        elif bit_pos_from_msb >= (total_bits - sign_bits - exp_bits):
            color = "#0066cc"  # exponent
            region = "exponent"
        else:
            color = "#00aa00"  # mantissa
            region = "mantissa"

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" '
            f'fill="{color}" title="bit {bit_idx}: {count} errors ({region})"/>'
        )
        # Bit index label
        svg_parts.append(
            f'<text x="{x + bar_width // 2}" y="{bar_max_height + 55}" '
            f'font-size="9" text-anchor="middle">{bit_idx}</text>'
        )

    # Legend
    legend_y = bar_max_height + 80
    svg_parts.append(f'<rect x="40" y="{legend_y}" width="12" height="12" fill="#cc0000"/>')
    svg_parts.append(f'<text x="56" y="{legend_y + 10}" font-size="11">sign</text>')
    svg_parts.append(f'<rect x="110" y="{legend_y}" width="12" height="12" fill="#0066cc"/>')
    svg_parts.append(f'<text x="126" y="{legend_y + 10}" font-size="11">exponent</text>')
    svg_parts.append(f'<rect x="200" y="{legend_y}" width="12" height="12" fill="#00aa00"/>')
    svg_parts.append(f'<text x="216" y="{legend_y + 10}" font-size="11">mantissa</text>')

    svg_parts.append('</svg>')
    Path(output_path).write_text('\n'.join(svg_parts))


# ============================================================================
# 模型级分析
# ============================================================================


def compare_model_bitwise(
    per_op_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    fmt: Union[FloatFormat, BitLayout] = FP32,
    final_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_warning_indices: int = 10,
) -> ModelBitAnalysis:
    """
    模型级 bit 分析

    Args:
        per_op_pairs: {op_name: (golden, dut)} 逐算子数据对
        fmt: 浮点格式
        final_pair: (golden, dut) 全局数据对（可选）
        max_warning_indices: 每条告警的最大索引数

    Returns:
        ModelBitAnalysis
    """
    per_op = {}
    for op_name, (g, d) in per_op_pairs.items():
        per_op[op_name] = _compare_bit_analysis_impl(g, d, fmt, max_warning_indices)

    global_result = None
    if final_pair is not None:
        g_final, d_final = final_pair
        global_result = _compare_bit_analysis_impl(g_final, d_final, fmt, max_warning_indices)

    return ModelBitAnalysis(per_op=per_op, global_result=global_result)


def print_model_bit_analysis(result: ModelBitAnalysis, name: str = ""):
    """打印模型级 bit 分析结果"""
    title = f"[{name}] Model Bit Analysis" if name else "Model Bit Analysis"
    print(f"\n{title}")
    print("=" * 70)

    # 逐算子
    print("\n  逐算子分析:")
    print("  " + "-" * 66)

    total_diff = 0
    total_elements = 0
    for op_name, r in result.per_op.items():
        s = r.summary
        total_diff += s.diff_elements
        total_elements += s.total_elements
        status = "[!]" if r.has_critical else "[.]"
        print(
            f"  {status} {op_name:<20} "
            f"diff={s.diff_elements}/{s.total_elements} "
            f"sign={s.sign_flip_count} exp={s.exponent_diff_count} mant={s.mantissa_diff_count}"
        )

    # 合计
    print("  " + "-" * 66)
    print(f"  合计: diff={total_diff}/{total_elements}")

    # Global
    if result.global_result is not None:
        print(f"\n  Global 分析:")
        s = result.global_result.summary
        status = "[!]" if result.global_result.has_critical else "[.]"
        print(
            f"  {status} diff={s.diff_elements}/{s.total_elements} "
            f"sign={s.sign_flip_count} exp={s.exponent_diff_count} mant={s.mantissa_diff_count}"
        )

    print("=" * 70)
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
        self.format = format or FP32

    @staticmethod
    def compare(
        golden: np.ndarray,
        result: np.ndarray,
        fmt: Union[FloatFormat, BitLayout] = FP32,
        max_warning_indices: int = 10,
    ) -> BitAnalysisResult:
        """Bit 级分析（静态方法）"""
        return _compare_bit_analysis_impl(golden, result, fmt, max_warning_indices)

    @staticmethod
    def compare_as_double(
        golden: np.ndarray,
        result: np.ndarray,
    ) -> dict:
        """转为 double 精度比对（静态方法）"""
        return _compare_as_double_impl(golden, result)

    @staticmethod
    def print_result(result: BitAnalysisResult, name: str = ""):
        """打印 bit 级分析结果（静态方法）"""
        return print_bit_analysis(result, name)

    def run(self, ctx: CompareContext) -> BitAnalysisResult:
        """执行 bit 级分析（Strategy 协议方法）"""
        return self.compare(ctx.golden, ctx.dut, fmt=self.format)

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
    "INT8",
    "UINT8",
    "WarnLevel",
    "BitAnalysisSummary",
    "BitAnalysisWarning",
    "BitAnalysisResult",
    "ModelBitAnalysis",
    "BitAnalysisStrategy",
    # 可视化
    "print_bit_analysis",
    "print_bit_template",
    "print_bit_heatmap",
    "gen_bit_heatmap_svg",
    "gen_perbit_bar_svg",
    # 模型级
    "compare_model_bitwise",
    "print_model_bit_analysis",
]
