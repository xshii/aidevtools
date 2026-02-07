"""
Bit 级比对分析 + 告警 + 热力图

对浮点数组进行逐 bit 对比，定位 sign/exponent/mantissa 差异，
生成结构化告警及热力图可视化。

支持:
  - 标准浮点: float32 (1+8+23), float16 (1+5+10), bfloat16 (1+8+7)
  - 自定义格式: 通过 BitLayout 配置任意 bit 分布
  - Block Floating Point: BFP16 (1+0+15), BFP8 (1+0+7), BFP4 (1+0+3) — 共享指数
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# === 浮点格式定义 ===

class FloatFormat(Enum):
    """标准浮点格式"""
    FLOAT32 = "float32"   # 1 + 8 + 23
    FLOAT16 = "float16"   # 1 + 5 + 10
    BFLOAT16 = "bfloat16" # 1 + 8 + 7


# 每种标准格式的 bit 分布: (sign_bits, exponent_bits, mantissa_bits)
_FORMAT_LAYOUT = {
    FloatFormat.FLOAT32: (1, 8, 23),
    FloatFormat.FLOAT16: (1, 5, 10),
    FloatFormat.BFLOAT16: (1, 8, 7),
}


# 字母 → 标签映射
_LETTER_TO_LABEL = {
    'S': 'sign',
    'E': 'exponent',
    'M': 'mantissa',
    'I': 'integer',
    'P': 'parity',
    'F': 'flag',
    'D': 'data',
}


@dataclass
class BitLayout:
    """
    通用 bit 分布配置 — 通过字母模板承载任意数据类型

    每个 bit 用一个大写字母标识所属分组:
        S = sign, E = exponent, M = mantissa, I = integer, etc.

    构造方式 1 — 经典 (sign/exponent/mantissa):
        layout = BitLayout(sign_bits=1, exponent_bits=8, mantissa_bits=23)
        # 自动生成 bit_template = "SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM"

    构造方式 2 — 统一字母模板:
        # 自定义 FP8 E4M3
        layout = BitLayout(template="SEEEEMMMM", name="fp8_e4m3")

        # BFP8: 共享指数 + per-element，()*N 语法
        layout = BitLayout(template="EEEEEEEE(SMMMMMMM)*16", name="bfp8")
        # 括号外 = 共享 bit (每 block 一份)
        # 括号内 = per-element bit
        # *N = block_size

        # 普通格式 (无共享): 直接写字母
        layout = BitLayout(template="SIIIIIII", name="int8")

    每个字母的语义:
        S = sign, E = exponent, M = mantissa
        I = integer, F = flag, D = data, P = parity
    """

    sign_bits: int = 0             # 符号位数 (经典模式)
    exponent_bits: int = 0         # per-element 指数位数 (经典模式)
    mantissa_bits: int = 0         # 尾数/数据位数 (经典模式)
    name: str = ""                 # 格式名称
    shared_exponent_bits: int = 0  # 共享指数位数 (BFP 特有)
    block_size: int = 1            # 共享指数的 block 大小

    # 统一字母模板: "EEEEEEEE(SMMMMMMM)*16" 或 "SIIIIIII"
    template: str = ""

    # 内部解析字段 (不在 __init__ 中)
    _shared_part: str = field(default="", init=False, repr=False)
    _element_part: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        import re

        if self.template:
            # 尝试解析 ()*N 语法: "EEEEEEEE(SMMMMMMM)*16"
            m = re.match(r'^([A-Z]*)(?:\(([A-Z]+)\)\*(\d+))?$', self.template)
            if m and m.group(2):
                # 有 ()*N 语法
                self._shared_part = m.group(1) or ""
                self._element_part = m.group(2)
                self.block_size = int(m.group(3))
            else:
                # 普通模板: 全部是 per-element
                self._shared_part = ""
                self._element_part = self.template

            # 从 shared part 推导 shared_exponent_bits
            if self._shared_part:
                self.shared_exponent_bits = len(self._shared_part)

            # 从 element part 推导 sign/exponent/mantissa bits
            s = e = mant = 0
            for ch in self._element_part:
                label = _LETTER_TO_LABEL.get(ch, ch.lower())
                if "sign" in label:
                    s += 1
                elif "exp" in label:
                    e += 1
                else:
                    mant += 1
            self.sign_bits = s
            self.exponent_bits = e
            self.mantissa_bits = mant

    @property
    def shared_template(self) -> str:
        """共享 bit 模板 (只读, 从 template 解析)"""
        return self._shared_part

    @classmethod
    def from_template(
        cls,
        template: str,
        name: str = "",
        shared_template: str = "",
        block_size: int = 1,
    ) -> "BitLayout":
        """
        从字母模板构建 BitLayout

        Args:
            template: per-element 字母模板或统一模板
                      e.g. "SMMMMMMM" (per-element only)
                      e.g. "EEEEEEEE(SMMMMMMM)*16" (统一模板)
            name: 格式名称
            shared_template: 共享 bit 字母模板 (向后兼容)
            block_size: block 大小 (向后兼容, 配合 shared_template 使用)
        """
        if shared_template:
            # 向后兼容: 组装成统一模板
            st = shared_template.split('*')[0] if '*' in shared_template else shared_template
            unified = f"{st}({template})*{block_size}"
        else:
            unified = template
        return cls(template=unified, name=name)

    @property
    def total_bits(self) -> int:
        """per-element 总 bit 数"""
        if self._element_part:
            return len(self._element_part)
        return self.sign_bits + self.exponent_bits + self.mantissa_bits

    @property
    def display_name(self) -> str:
        """显示名称"""
        if self.name:
            return self.name
        parts = []
        if self.sign_bits:
            parts.append(f"s{self.sign_bits}")
        if self.exponent_bits:
            parts.append(f"e{self.exponent_bits}")
        parts.append(f"m{self.mantissa_bits}")
        base = "+".join(parts)
        if self.shared_exponent_bits:
            return f"{base} (shared_exp={self.shared_exponent_bits}, blk={self.block_size})"
        return base

    def as_tuple(self) -> Tuple[int, int, int]:
        """返回 (sign_bits, exponent_bits, mantissa_bits) — 兼容分析逻辑"""
        return (self.sign_bits, self.exponent_bits, self.mantissa_bits)

    # --- template 属性 ---

    @property
    def bit_template(self) -> str:
        """
        Bit 模板字母版 (MSB → LSB)

        示例:
            float32: "SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM"
            bfp8:    "SMMMMMMM"
            custom:  "FEEEDDDP"  (flag/exp/data/parity)
        """
        if self._element_part:
            return self._element_part
        return "S" * self.sign_bits + "E" * self.exponent_bits + "M" * self.mantissa_bits

    @property
    def bit_template_spaced(self) -> str:
        """
        带空格分隔的 bit 模板 — 相同字母连续, 不同字母间加空格

        示例:
            float32: "S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM"
            bfp8:    "S MMMMMMM"
            custom:  "FF EEE DDDD P"
        """
        tmpl = self.bit_template
        if not tmpl:
            return ""
        parts = []
        current_char = tmpl[0]
        current_run = current_char
        for ch in tmpl[1:]:
            if ch == current_char:
                current_run += ch
            else:
                parts.append(current_run)
                current_char = ch
                current_run = ch
        parts.append(current_run)
        return " ".join(parts)

    @property
    def bit_group_labels(self) -> dict:
        """
        分组标签 — 每个字母的语义

        示例:
            float32: {'S': 'sign', 'E': 'exponent', 'M': 'mantissa'}
            INT8:    {'S': 'sign', 'I': 'integer'}
            custom:  {'F': 'flag', 'E': 'exponent', 'D': 'data', 'P': 'parity'}
        """
        tmpl = self.bit_template
        labels = {}
        for ch in tmpl:
            if ch not in labels:
                labels[ch] = _LETTER_TO_LABEL.get(ch, ch.lower())
        return labels

    @property
    def shared_group_labels(self) -> dict:
        """
        共享 bit 的分组标签, 自动加 shared_ 前缀

        示例:
            BFP8: {'E': 'shared_exponent'}
        """
        labels = {}
        for ch in self.shared_template:
            if ch not in labels:
                base = _LETTER_TO_LABEL.get(ch, ch.lower())
                labels[ch] = f"shared_{base}"
        return labels

    def group_colors(self) -> dict:
        """
        分组字母 → 颜色映射 (用于 SVG 可视化)

        约定:
        - sign-like → 红
        - exponent-like → 橙
        - mantissa-like → 蓝
        - 其他 → 绿/紫/灰 循环
        """
        labels = self.bit_group_labels
        _EXTRA_COLORS = ["#4caf50", "#9c27b0", "#607d8b", "#795548", "#009688"]
        colors = {}
        extra_idx = 0
        for letter, label in sorted(labels.items()):
            lower = label.lower()
            if "sign" in lower or "符号" in lower:
                colors[letter] = "#f44336"  # 红
            elif "exp" in lower or "指数" in lower:
                colors[letter] = "#ff9800"  # 橙
            elif "mant" in lower or "尾数" in lower or "data" in lower or "整数" in lower or "integer" in lower:
                colors[letter] = "#2196f3"  # 蓝
            else:
                colors[letter] = _EXTRA_COLORS[extra_idx % len(_EXTRA_COLORS)]
                extra_idx += 1
        return colors


# === 预定义 BitLayout ===

FP32 = BitLayout(template="SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM", name="fp32")
FP16 = BitLayout(template="SEEEEEMMMMMMMMMM", name="fp16")
BFP16 = BitLayout(template="EEEEEEEE(SMMMMMMMMMMMMMMM)*16", name="bfp16")
BFP8 = BitLayout(template="EEEEEEEE(SMMMMMMM)*16", name="bfp8")
BFP4 = BitLayout(template="EEEEEEEE(SMMM)*16", name="bfp4")
INT8 = BitLayout(template="SIIIIIII", name="int8")
UINT8 = BitLayout(template="IIIIIIII", name="uint8")


# === 结果类型 ===

class WarnLevel(Enum):
    """告警级别"""
    CRITICAL = "CRITICAL"  # sign flip / exponent overflow
    WARNING = "WARNING"    # exponent mismatch (small)
    INFO = "INFO"          # mantissa-only diff (expected for quantization)


@dataclass
class BitDiffSummary:
    """bit 级差异汇总"""

    total_elements: int
    diff_elements: int       # 有任何 bit 差异的元素数
    sign_flip_count: int     # 符号位翻转数
    exponent_diff_count: int # 指数域有差异的元素数
    mantissa_diff_count: int # 尾数域有差异的元素数

    # 每个 bit position 的错误计数 (length = total_bits)
    per_bit_error_count: np.ndarray = field(default=None, repr=False)

    # 指数差异统计
    max_exponent_diff: int = 0  # 最大指数差 (解码后)

    @property
    def diff_ratio(self) -> float:
        return self.diff_elements / self.total_elements if self.total_elements > 0 else 0.0

    @property
    def sign_flip_ratio(self) -> float:
        return self.sign_flip_count / self.total_elements if self.total_elements > 0 else 0.0


@dataclass
class BitWarning:
    """单条 bit 级告警"""

    level: WarnLevel
    message: str
    count: int           # 涉及的元素数
    indices: np.ndarray = field(default=None, repr=False)  # 前 N 个索引


@dataclass
class BitAnalysisResult:
    """完整 bit 级分析结果"""

    fmt: Union[FloatFormat, BitLayout]
    summary: BitDiffSummary
    warnings: List[BitWarning]

    @property
    def has_critical(self) -> bool:
        return any(w.level == WarnLevel.CRITICAL for w in self.warnings)

    @property
    def layout(self) -> Tuple[int, int, int]:
        """返回 (sign_bits, exponent_bits, mantissa_bits)"""
        return _resolve_layout(self.fmt)

    @property
    def format_name(self) -> str:
        """格式显示名称"""
        if isinstance(self.fmt, BitLayout):
            return self.fmt.display_name
        return self.fmt.value


# === 核心分析 ===

def _resolve_layout(fmt: Union[FloatFormat, BitLayout, None]) -> Tuple[int, int, int]:
    """从 FloatFormat 或 BitLayout 解析 (sign_bits, exponent_bits, mantissa_bits)"""
    if isinstance(fmt, BitLayout):
        return fmt.as_tuple()
    if isinstance(fmt, FloatFormat):
        return _FORMAT_LAYOUT[fmt]
    return (1, 8, 23)  # default: float32


def _make_bit_layout(fmt: Union[FloatFormat, BitLayout, None]) -> BitLayout:
    """将 FloatFormat 转为等价的 BitLayout"""
    if isinstance(fmt, BitLayout):
        return fmt
    s, e, m = _resolve_layout(fmt)
    name = fmt.value if isinstance(fmt, FloatFormat) else "custom"
    return BitLayout(sign_bits=s, exponent_bits=e, mantissa_bits=m, name=name)


def print_bit_template(fmt: Union[FloatFormat, BitLayout]):
    """
    打印 bit 模板 — 可视化每个 bit 的归属 (字母标识)

    输出示例 (float32):
        Bit Template: float32 (32 bits)
          MSB                            LSB
          S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM
          ─── S=sign  E=exponent  M=mantissa

    输出示例 (BFP8):
        Bit Template: bfp8 (8 bits/element)
          Per-element:
            MSB      LSB
            S MMMMMMM
          Shared (per block of 16):
            EEEEEEEE  (8 bits)
          ─── M=mantissa  S=sign  | shared: E=shared_exponent

    输出示例 (INT8):
        Bit Template: int8 (8 bits)
          MSB      LSB
          S IIIIIII
          ─── I=integer  S=sign
    """
    layout = _make_bit_layout(fmt)
    total = layout.total_bits

    tmpl_letter = layout.bit_template_spaced
    labels = layout.bit_group_labels

    # 构造 groups 说明
    group_desc = "  ".join(f"{letter}={label}" for letter, label in sorted(labels.items()))

    if layout.shared_template or layout.shared_exponent_bits:
        # BFP / 含共享 bit 的格式
        print(f"\n  Bit Template: {layout.display_name} ({total} bits/element)")
        print(f"    Per-element:")
        print(f"      {'MSB':<{len(tmpl_letter) - 3}}LSB")
        print(f"      {tmpl_letter}")

        # Shared bits
        shared_tmpl = layout.shared_template
        if shared_tmpl:
            shared_letter = shared_tmpl
            shared_labels = layout.shared_group_labels
            shared_desc = "  ".join(
                f"{letter}={label}" for letter, label in sorted(shared_labels.items())
            )
        else:
            shared_letter = "E" * layout.shared_exponent_bits
            shared_desc = "shared_exponent"

        print(f"    Shared (per block of {layout.block_size}):")
        print(f"      {shared_letter}  ({len(shared_letter)} bits)")

        if shared_tmpl:
            group_desc += f"  | shared: {shared_desc}"
    else:
        print(f"\n  Bit Template: {layout.display_name} ({total} bits)")
        print(f"    {'MSB':<{len(tmpl_letter) - 3}}LSB")
        print(f"    {tmpl_letter}")

    print(f"    ─── {group_desc}")
    print()


def _detect_format(arr: np.ndarray) -> FloatFormat:
    """从 numpy dtype 推断浮点格式"""
    if arr.dtype == np.float32:
        return FloatFormat.FLOAT32
    if arr.dtype == np.float16:
        return FloatFormat.FLOAT16
    # bfloat16 在 numpy 中通常是 uint16 或自定义 dtype
    return FloatFormat.FLOAT32


def _to_uint(arr: np.ndarray, fmt: Union[FloatFormat, BitLayout]) -> np.ndarray:
    """将数组转为无符号整型 (保留 bit pattern)

    当 fmt 是 BFP 格式 (shared_exponent_bits > 0) 且输入是 fp32 时，
    自动通过 simulate_quantize 量化后再 view 为 uint，避免静默错误。
    """
    # BitLayout: 根据 total_bits 选择 uint 类型
    if isinstance(fmt, BitLayout):
        total = fmt.total_bits
        # BFP 格式 + fp32 输入: 自动量化后比较 fp32 bit pattern
        if fmt.shared_exponent_bits > 0 and arr.dtype in (np.float32, np.float64):
            from aidevtools.formats.quantize import simulate_quantize
            qtype_map = {"bfp16": "bfp16", "bfp8": "bfp8", "bfp4": "bfp4"}
            qtype = qtype_map.get(fmt.name)
            if qtype:
                arr = simulate_quantize(arr.astype(np.float32), qtype)
            # 量化后 arr 仍为 fp32，用 fp32 bit pattern 比较
            return np.ascontiguousarray(arr.astype(np.float32)).view(np.uint32).flatten()
        if total <= 8:
            # 已经是 uint8 → 直接用
            if arr.dtype == np.uint8:
                return arr.flatten().copy()
            if arr.dtype in (np.float32, np.float64):
                return np.ascontiguousarray(arr.astype(np.float32)).view(np.uint32).flatten()
            return arr.flatten().astype(np.uint8)
        if total <= 16:
            if arr.dtype == np.uint16:
                return arr.flatten().copy()
            if arr.dtype in (np.float32, np.float64):
                return np.ascontiguousarray(arr.astype(np.float32)).view(np.uint32).flatten()
            return arr.flatten().astype(np.uint16)
        # total > 16: use uint32
        if arr.dtype in (np.float32, np.float64):
            return np.ascontiguousarray(arr.astype(np.float32)).view(np.uint32).flatten()
        return arr.flatten().astype(np.uint32)

    # FloatFormat: 标准处理
    if fmt == FloatFormat.FLOAT32:
        return np.ascontiguousarray(arr.astype(np.float32)).view(np.uint32).flatten()
    if fmt == FloatFormat.FLOAT16:
        return np.ascontiguousarray(arr.astype(np.float16)).view(np.uint16).flatten()
    if fmt == FloatFormat.BFLOAT16:
        if arr.dtype == np.uint16:
            return arr.flatten().copy()
        f32 = np.ascontiguousarray(arr.astype(np.float32)).view(np.uint32).flatten()
        return (f32 >> 16).astype(np.uint16)
    return np.ascontiguousarray(arr.astype(np.float32)).view(np.uint32).flatten()


def compare_bitwise(
    golden: np.ndarray,
    result: np.ndarray,
    fmt: Optional[Union[FloatFormat, BitLayout]] = None,
    max_warning_indices: int = 10,
) -> BitAnalysisResult:
    """
    Bit 级比对分析

    逐元素 XOR 比较 golden 和 result 的原始 bit pattern，
    分析 sign/exponent/mantissa 维度的差异并生成告警。

    Args:
        golden: 参考数据
        result: 待比对数据
        fmt: 浮点格式 — FloatFormat (标准浮点) 或 BitLayout (自定义格式)
             None=自动检测 dtype
        max_warning_indices: 每条告警最多记录的索引数

    Returns:
        BitAnalysisResult

    示例:
        # 标准 float32
        result = compare_bitwise(golden, dut, fmt=FloatFormat.FLOAT32)

        # BFP8 packed uint8 数据
        from aidevtools.compare.bitwise import BFP8
        result = compare_bitwise(golden_u8, dut_u8, fmt=BFP8)

        # 完全自定义
        layout = BitLayout(sign_bits=1, exponent_bits=5, mantissa_bits=2, name="fp8_e5m2")
        result = compare_bitwise(golden, dut, fmt=layout)
    """
    if fmt is None:
        fmt = _detect_format(golden)

    # BFP 格式 + fp32 输入: _to_uint 自动量化后返回 uint32 (fp32 bit pattern)
    # 此时用 FP32 的 layout (1+8+23) 做分析，才能正确解读 sign/exp/mant
    analysis_fmt = fmt
    if (isinstance(fmt, BitLayout) and fmt.shared_exponent_bits > 0
            and golden.dtype in (np.float32, np.float64)):
        analysis_fmt = FP32

    sign_bits, exp_bits, mant_bits = _resolve_layout(analysis_fmt)
    total_bits = sign_bits + exp_bits + mant_bits

    g_uint = _to_uint(golden, fmt)
    r_uint = _to_uint(result, fmt)
    total_elements = len(g_uint)

    if len(g_uint) != len(r_uint):
        raise ValueError(f"Shape mismatch: golden={golden.shape}, result={result.shape}")

    # 对于 <=8 bit 的非 BFP 格式 (如自定义 int8)，mask 掉高位
    if (isinstance(fmt, BitLayout) and total_bits <= 8
            and g_uint.dtype != np.uint8 and fmt.shared_exponent_bits == 0):
        elem_mask = np.uint32((1 << total_bits) - 1) if g_uint.dtype == np.uint32 else np.uint16((1 << total_bits) - 1)
        g_uint = g_uint & elem_mask
        r_uint = r_uint & elem_mask

    # XOR → 差异 bit mask
    xor = g_uint ^ r_uint

    # 全局: 有差异的元素
    diff_mask = xor != 0
    diff_elements = int(np.count_nonzero(diff_mask))

    # --- sign bit ---
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

    # --- exponent bits ---
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
        # 无 per-element 指数 (e.g. BFP)
        exp_diff = np.zeros(total_elements, dtype=np.int32)
        exp_diff_mask = np.zeros(total_elements, dtype=bool)
        exponent_diff_count = 0
        max_exponent_diff = 0

    # --- mantissa bits ---
    mant_mask_val = (1 << mant_bits) - 1
    g_mant = g_uint & mant_mask_val
    r_mant = r_uint & mant_mask_val
    mant_diff_mask = g_mant != r_mant
    mantissa_diff_count = int(np.count_nonzero(mant_diff_mask))

    # --- per-bit error count (vectorized) ---
    per_bit = _per_bit_count(xor, total_bits)

    summary = BitDiffSummary(
        total_elements=total_elements,
        diff_elements=diff_elements,
        sign_flip_count=sign_flip_count,
        exponent_diff_count=exponent_diff_count,
        mantissa_diff_count=mantissa_diff_count,
        per_bit_error_count=per_bit,
        max_exponent_diff=max_exponent_diff,
    )

    # === 生成告警 ===
    warnings = _generate_warnings(
        summary, sign_diff_mask, exp_diff_mask, exp_diff,
        diff_mask, total_elements, max_warning_indices,
    )

    return BitAnalysisResult(fmt=fmt, summary=summary, warnings=warnings)


def _generate_warnings(
    summary: BitDiffSummary,
    sign_diff_mask: np.ndarray,
    exp_diff_mask: np.ndarray,
    exp_diff: np.ndarray,
    diff_mask: np.ndarray,
    total: int,
    max_indices: int,
) -> List[BitWarning]:
    """根据分析结果生成结构化告警"""
    warnings = []

    # CRITICAL: sign bit flip
    if summary.sign_flip_count > 0:
        indices = np.where(sign_diff_mask)[0][:max_indices]
        ratio = summary.sign_flip_count / total * 100
        warnings.append(BitWarning(
            level=WarnLevel.CRITICAL,
            message=f"符号位翻转: {summary.sign_flip_count} 个元素 ({ratio:.2f}%) — "
                    f"数值正负反转，可能导致严重功能错误",
            count=summary.sign_flip_count,
            indices=indices,
        ))

    # CRITICAL: exponent diff >= 2 (magnitude change)
    large_exp_mask = np.abs(exp_diff) >= 2
    large_exp_count = int(np.count_nonzero(large_exp_mask))
    if large_exp_count > 0:
        indices = np.where(large_exp_mask)[0][:max_indices]
        warnings.append(BitWarning(
            level=WarnLevel.CRITICAL,
            message=f"指数域大偏移 (>=2): {large_exp_count} 个元素, "
                    f"最大偏移={summary.max_exponent_diff} — 数值量级严重偏离",
            count=large_exp_count,
            indices=indices,
        ))

    # WARNING: exponent diff == 1
    small_exp_mask = np.abs(exp_diff) == 1
    small_exp_count = int(np.count_nonzero(small_exp_mask))
    if small_exp_count > 0:
        indices = np.where(small_exp_mask)[0][:max_indices]
        ratio = small_exp_count / total * 100
        warnings.append(BitWarning(
            level=WarnLevel.WARNING,
            message=f"指数域偏移 (±1): {small_exp_count} 个元素 ({ratio:.2f}%) — "
                    f"数值约 2x 偏差，需关注",
            count=small_exp_count,
            indices=indices,
        ))

    # INFO: mantissa-only diff
    mant_only_mask = diff_mask & ~sign_diff_mask.astype(bool) & ~exp_diff_mask
    mant_only_count = int(np.count_nonzero(mant_only_mask))
    if mant_only_count > 0:
        indices = np.where(mant_only_mask)[0][:max_indices]
        ratio = mant_only_count / total * 100
        warnings.append(BitWarning(
            level=WarnLevel.INFO,
            message=f"仅尾数差异: {mant_only_count} 个元素 ({ratio:.2f}%) — "
                    f"量化舍入导致的正常精度损失",
            count=mant_only_count,
            indices=indices,
        ))

    # 无差异
    if not warnings and summary.diff_elements == 0:
        warnings.append(BitWarning(
            level=WarnLevel.INFO,
            message="Bit-exact: 无任何 bit 差异",
            count=0,
        ))

    return warnings


# === 打印输出 ===

_LEVEL_MARKS = {
    WarnLevel.CRITICAL: "[!]",
    WarnLevel.WARNING: "[~]",
    WarnLevel.INFO: "[.]",
}


def print_bit_analysis(result: BitAnalysisResult, name: str = ""):
    """打印 bit 级分析报告 (含 bit 模板)"""
    s = result.summary
    name_str = f"[{name}] " if name else ""
    layout = _make_bit_layout(result.fmt)

    print(f"\n{name_str}Bit-Level Analysis ({result.format_name})")
    print("=" * 60)

    # Bit template
    tmpl_letter = layout.bit_template_spaced
    labels = layout.bit_group_labels
    group_desc = "  ".join(f"{letter}={label}" for letter, label in sorted(labels.items()))
    print(f"  Bit layout:  {tmpl_letter}")
    print(f"               {group_desc}")
    if layout.shared_template or layout.shared_exponent_bits:
        shared_tmpl = layout.shared_template
        if shared_tmpl:
            shared_letter = shared_tmpl
            shared_labels = layout.shared_group_labels
            shared_desc = "  ".join(
                f"{letter}={label}" for letter, label in sorted(shared_labels.items())
            )
        else:
            shared_letter = "E" * layout.shared_exponent_bits
            shared_desc = "shared_exponent"
        print(f"  Shared:      {shared_letter} ({len(shared_letter)}b x blk{layout.block_size})")
    print()
    print(f"  Total elements:    {s.total_elements:,}")
    print(f"  Diff elements:     {s.diff_elements:,} ({s.diff_ratio:.2%})")
    print(f"  Sign flips:        {s.sign_flip_count:,} ({s.sign_flip_ratio:.2%})")
    print(f"  Exponent diffs:    {s.exponent_diff_count:,} (max shift={s.max_exponent_diff})")
    print(f"  Mantissa diffs:    {s.mantissa_diff_count:,}")
    print("-" * 60)

    for w in result.warnings:
        mark = _LEVEL_MARKS[w.level]
        print(f"  {mark} {w.message}")
        if w.indices is not None and len(w.indices) > 0:
            idx_str = ", ".join(str(i) for i in w.indices[:10])
            print(f"       indices: [{idx_str}]")

    print("=" * 60)
    print()


def _per_bit_count(xor: np.ndarray, total_bits: int) -> np.ndarray:
    """向量化 per-bit 错误计数: 返回 shape=(total_bits,) 的 int64 数组

    将 xor 数组 unpack 为 bit 矩阵后按列求和，
    比逐 bit Python loop + count_nonzero 快约 3-5x。
    """
    flat = xor.flatten()
    raw = flat.view(np.uint8)  # (N * bytes_per_elem,)
    bits = np.unpackbits(raw, bitorder='little')  # (N * bytes_per_elem * 8,)
    bytes_per_elem = xor.dtype.itemsize
    total_raw_bits = bytes_per_elem * 8
    bits = bits.reshape(-1, total_raw_bits)  # (N, total_raw_bits)
    # 只取有效 bit 位 [0, total_bits)
    per_bit = bits[:, :total_bits].sum(axis=0).astype(np.int64)
    return per_bit


def _popcount_block(xor_block: np.ndarray, total_bits: int) -> int:
    """统计 XOR block 中的差异 bit 总数"""
    return int(_per_bit_count(xor_block, total_bits).sum())


def print_bit_heatmap(
    golden: np.ndarray,
    result: np.ndarray,
    fmt: Optional[Union[FloatFormat, BitLayout]] = None,
    block_size: int = 256,
    cols: int = 40,
):
    """
    打印 bit 级误差热力图

    将数组分块，每块统计 bit 差异数占总 bit 数的比例，
    以字符表示每块的 bit 健康度。

    符号:
      '.' = 0% diff (bit-exact)
      'o' = < 1% diff bits
      'X' = < 10% diff bits
      '#' = >= 10% diff bits

    Args:
        golden: 参考数据
        result: 待比对数据
        fmt: 浮点格式 — FloatFormat 或 BitLayout (None=自动检测)
        block_size: 每块元素数
        cols: 每行显示的块数
    """
    if fmt is None:
        fmt = _detect_format(golden)

    total_bits_per_elem = sum(_resolve_layout(fmt))

    g_uint = _to_uint(golden, fmt)
    r_uint = _to_uint(result, fmt)
    xor = g_uint ^ r_uint

    total = len(g_uint)
    blocks_info = []

    for offset in range(0, total, block_size):
        end = min(offset + block_size, total)
        blk_xor = xor[offset:end]
        # popcount: 统计每个元素的差异 bit 数
        diff_bits = _popcount_block(blk_xor, total_bits_per_elem)
        total_bits_in_block = (end - offset) * total_bits_per_elem
        ratio = diff_bits / total_bits_in_block if total_bits_in_block > 0 else 0.0
        blocks_info.append((offset, end - offset, diff_bits, ratio))

    # 打印
    def _char(ratio):
        if ratio == 0:
            return "."
        if ratio < 0.01:
            return "o"
        if ratio < 0.10:
            return "X"
        return "#"

    n_blocks = len(blocks_info)
    total_diff = sum(b[2] for b in blocks_info)
    total_possible = total * total_bits_per_elem
    global_ratio = total_diff / total_possible if total_possible > 0 else 0.0

    print(f"\n  Bit Heatmap ({n_blocks} blocks x {block_size} elem, "
          f"overall {global_ratio:.2%} bits differ)")
    print(f"  {'=' * cols}")

    for i in range(0, n_blocks, cols):
        row = blocks_info[i:i + cols]
        chars = "".join(_char(b[3]) for b in row)
        start = row[0][0]
        print(f"  {start:>8} |{chars}|")

    print(f"  {'=' * cols}")
    print(f"  Legend: . = exact, o < 1%, X < 10%, # >= 10% bits differ")
    print()


def gen_bit_heatmap_svg(
    golden: np.ndarray,
    result: np.ndarray,
    output_path: str,
    fmt: Optional[Union[FloatFormat, BitLayout]] = None,
    block_size: int = 256,
    cols: int = 64,
    cell_size: int = 8,
):
    """
    生成 bit 级差异 SVG 热力图

    颜色编码:
      绿色 (#4caf50) = bit-exact
      浅绿 (#8bc34a) = < 0.1% diff
      黄色 (#ffeb3b) = < 1% diff
      橙色 (#ff9800) = < 10% diff
      红色 (#f44336) = >= 10% diff

    Args:
        golden: 参考数据
        result: 待比对数据
        output_path: SVG 输出路径
        fmt: 浮点格式 — FloatFormat 或 BitLayout (None=自动检测)
        block_size: 每块元素数
        cols: 每行显示的块数
        cell_size: 每个单元格像素大小
    """
    from pathlib import Path

    if fmt is None:
        fmt = _detect_format(golden)

    total_bits_per_elem = sum(_resolve_layout(fmt))

    g_uint = _to_uint(golden, fmt)
    r_uint = _to_uint(result, fmt)
    xor = g_uint ^ r_uint

    total = len(g_uint)
    ratios = []

    for offset in range(0, total, block_size):
        end = min(offset + block_size, total)
        blk_xor = xor[offset:end]
        diff_bits = _popcount_block(blk_xor, total_bits_per_elem)
        total_bits_in_block = (end - offset) * total_bits_per_elem
        ratio = diff_bits / total_bits_in_block if total_bits_in_block > 0 else 0.0
        ratios.append(ratio)

    n_blocks = len(ratios)
    rows = (n_blocks + cols - 1) // cols
    width = cols * cell_size
    height = rows * cell_size

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#f0f0f0"/>',
    ]

    for i, ratio in enumerate(ratios):
        x = (i % cols) * cell_size
        y = (i // cols) * cell_size

        if ratio == 0:
            color = "#4caf50"  # 绿
        elif ratio < 0.001:
            color = "#8bc34a"  # 浅绿
        elif ratio < 0.01:
            color = "#ffeb3b"  # 黄
        elif ratio < 0.10:
            color = "#ff9800"  # 橙
        else:
            color = "#f44336"  # 红

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{cell_size - 1}" height="{cell_size - 1}" '
            f'fill="{color}" title="block:{i} diff:{ratio:.2%}"/>'
        )

    svg_parts.append('</svg>')
    Path(output_path).write_text("\n".join(svg_parts), encoding="utf-8")


def gen_perbit_bar_svg(
    result: BitAnalysisResult,
    output_path: str,
    bar_width: int = 16,
    bar_max_height: int = 200,
):
    """
    生成 per-bit 错误分布条形图 SVG

    X 轴: bit position (MSB 在左 → LSB 在右)
    Y 轴: 该 bit 出错的元素数
    颜色: sign=红, exponent=橙, mantissa=蓝

    Args:
        result: compare_bitwise 的输出
        output_path: SVG 输出路径
        bar_width: 每根柱子的像素宽度
        bar_max_height: 最高柱子的像素高度
    """
    from pathlib import Path

    layout = _make_bit_layout(result.fmt)
    s = result.summary
    per_bit = s.per_bit_error_count
    total_bits = layout.total_bits

    max_count = int(per_bit.max()) if len(per_bit) > 0 and per_bit.max() > 0 else 1

    # 从字母模板构建 bit_pos → 字母 → 颜色映射 (MSB=left)
    tmpl = layout.bit_template
    labels = layout.bit_group_labels
    colors = layout.group_colors()

    # tmpl[0] = MSB, tmpl[-1] = LSB → bit_pos = total_bits-1-i
    def _color_for_bit(bit_pos):
        idx = total_bits - 1 - bit_pos  # index into template
        if 0 <= idx < len(tmpl):
            letter = tmpl[idx]
            return colors.get(letter, "#607d8b")
        return "#607d8b"

    margin_top = 30
    margin_bottom = 40
    margin_left = 60
    width = margin_left + total_bits * bar_width + 20
    height = margin_top + bar_max_height + margin_bottom

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="monospace" font-size="10">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width // 2}" y="18" text-anchor="middle" font-size="12">'
        f'Per-Bit Error Distribution ({result.format_name})</text>',
    ]

    for i in range(total_bits):
        bit_pos = total_bits - 1 - i
        count = int(per_bit[bit_pos])
        bar_h = int(count / max_count * bar_max_height) if max_count > 0 else 0
        bar_h = max(bar_h, 1) if count > 0 else 0

        x = margin_left + i * bar_width
        y = margin_top + bar_max_height - bar_h
        color = _color_for_bit(bit_pos)

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width - 2}" height="{bar_h}" '
            f'fill="{color}" title="bit{bit_pos}: {count}"/>'
        )

        # bit label
        label_y = margin_top + bar_max_height + 12
        svg_parts.append(
            f'<text x="{x + bar_width // 2}" y="{label_y}" '
            f'text-anchor="middle" font-size="8">{bit_pos}</text>'
        )

    # Y-axis label
    svg_parts.append(
        f'<text x="10" y="{margin_top + bar_max_height // 2}" '
        f'text-anchor="middle" transform="rotate(-90, 10, {margin_top + bar_max_height // 2})" '
        f'font-size="10">error count</text>'
    )

    # Legend — 从 bit_group_labels + group_colors 动态生成
    ly = margin_top + bar_max_height + 30
    lx = margin_left
    for letter, label in sorted(labels.items()):
        color = colors.get(letter, "#607d8b")
        svg_parts.append(f'<rect x="{lx}" y="{ly - 8}" width="10" height="10" fill="{color}"/>')
        svg_parts.append(f'<text x="{lx + 14}" y="{ly}" font-size="10">{label}</text>')
        lx += max(len(label) * 8 + 24, 80)

    svg_parts.append('</svg>')
    Path(output_path).write_text("\n".join(svg_parts), encoding="utf-8")


# === 一键式模型比对 ===

@dataclass
class ModelBitAnalysis:
    """整体模型 bit 级分析结果

    Attributes:
        per_op: 逐算子分析结果 {op_name: BitAnalysisResult}
        global_result: 全局 (最终输出) 分析结果, 可选
    """
    per_op: Dict[str, BitAnalysisResult]
    global_result: Optional[BitAnalysisResult] = None

    @property
    def has_critical(self) -> bool:
        """任何算子或全局有 CRITICAL"""
        if self.global_result and self.global_result.has_critical:
            return True
        return any(r.has_critical for r in self.per_op.values())


def compare_model_bitwise(
    per_op_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    fmt: Optional[Union[FloatFormat, BitLayout]] = None,
    final_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> ModelBitAnalysis:
    """一键式 bit 级分析: 逐算子 + 全局

    Args:
        per_op_pairs: {op_name: (golden, dut)} 逐算子比对对
        fmt: 浮点格式 (FloatFormat 或 BitLayout)
        final_pair: (golden_final, dut_final) 全局最终输出, 可选

    Returns:
        ModelBitAnalysis 含 per_op + global_result

    示例:
        # fp32 输出比较
        result = compare_model_bitwise(
            per_op_pairs={"Q": (golden_q, dut_q), "K": (golden_k, dut_k)},
            fmt=FP32,
            final_pair=(golden_out, dut_out),
        )
        # BFP8 格式: 传 fp32 数组时自动 simulate_quantize 后比较
        result = compare_model_bitwise(per_op_pairs=pairs, fmt=BFP8)
        print_model_bit_analysis(result, name="Encoder")
    """
    per_op = {}
    for op_name, (golden, dut) in per_op_pairs.items():
        per_op[op_name] = compare_bitwise(golden, dut, fmt=fmt)

    global_result = None
    if final_pair is not None:
        global_result = compare_bitwise(final_pair[0], final_pair[1], fmt=fmt)

    return ModelBitAnalysis(per_op=per_op, global_result=global_result)


def print_model_bit_analysis(result: ModelBitAnalysis, name: str = ""):
    """打印模型 bit 级分析报告 (全局 + 逐算子表)

    Args:
        result: compare_model_bitwise 返回值
        name: 模型/报告名称
    """
    name_str = f"[{name}] " if name else ""

    # 全局分析
    if result.global_result is not None:
        print_bit_analysis(result.global_result, name=f"{name} Global" if name else "Global")

    # 逐算子表
    print(f"\n{name_str}逐算子 bit 级分析:")
    print(f"  {'算子':<20} {'总元素':>8} {'Diff':>8} {'SignFlip':>9} "
          f"{'ExpDiff':>8} {'MantDiff':>9} {'Level':>10}")
    print(f"  {'-'*74}")

    for op_name, r in result.per_op.items():
        s = r.summary
        if r.has_critical:
            level = "CRITICAL"
        elif any(w.level.value == "WARNING" for w in r.warnings):
            level = "WARNING"
        else:
            level = "INFO"
        print(f"  {op_name:<20} {s.total_elements:>8} {s.diff_elements:>8} "
              f"{s.sign_flip_count:>9} {s.exponent_diff_count:>8} "
              f"{s.mantissa_diff_count:>9} {level:>10}")

    # 汇总
    total_diff = sum(r.summary.diff_elements for r in result.per_op.values())
    total_elems = sum(r.summary.total_elements for r in result.per_op.values())
    total_sign = sum(r.summary.sign_flip_count for r in result.per_op.values())
    print(f"  {'-'*74}")
    print(f"  {'合计':<20} {total_elems:>8} {total_diff:>8} {total_sign:>9}")
    print()
