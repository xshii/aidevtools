"""
Bit XOR 比对策略（无语义理解）

纯粹的 bit-by-bit XOR 对比，不关心数据格式。
用于快速判断数据是否完全一致。

使用场景:
  - 快速检查数据是否有任何差异
  - 调试硬件错误（bit翻转）
  - 不需要理解浮点格式的场景
"""

from dataclasses import dataclass
import numpy as np

from .base import CompareStrategy, CompareContext


# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class BitXorResult:
    """Bit XOR 比对结果"""
    total_elements: int
    diff_elements: int  # 有差异的元素数
    total_bits: int     # 总 bit 数
    diff_bits: int      # 差异的 bit 数

    @property
    def diff_element_ratio(self) -> float:
        """差异元素比例"""
        return self.diff_elements / self.total_elements if self.total_elements > 0 else 0.0

    @property
    def diff_bit_ratio(self) -> float:
        """差异 bit 比例"""
        return self.diff_bits / self.total_bits if self.total_bits > 0 else 0.0


# ============================================================================
# 核心比对函数
# ============================================================================


# ============================================================================
# Strategy 类（包含静态方法）
# ============================================================================


class BitXorStrategy(CompareStrategy):
    """
    Bit XOR 比对策略（无语义理解）

    纯粹的 bit-by-bit XOR 对比，快速判断数据是否有差异。

    特点：
    - 最快的比对方式
    - 不理解浮点格式
    - 适合快速筛选

    使用方式：
        # 方式1: 直接调用静态方法
        result = BitXorStrategy.compare(golden, dut)

        # 方式2: 通过引擎
        engine = CompareEngine(BitXorStrategy())
        result = engine.run(dut, golden)
    """

    @staticmethod
    def compare(
        golden: np.ndarray,
        result: np.ndarray,
    ) -> BitXorResult:
        """
        Bit XOR 比对（静态方法）

        纯粹的 bit-by-bit XOR 对比，不关心数据格式。

        Args:
            golden: 参考数据
            result: 待比对数据

        Returns:
            BitXorResult

        示例:
            result = BitXorStrategy.compare(golden, dut)
            print(f"差异元素: {result.diff_elements}/{result.total_elements}")
            print(f"差异 bit: {result.diff_bits}/{result.total_bits}")
        """
        flat_g = golden.flatten()
        flat_r = result.flatten()

        if len(flat_g) != len(flat_r):
            raise ValueError(f"Shape mismatch: {golden.shape} vs {result.shape}")

        total_elements = len(flat_g)

        # 转为统一格式做 XOR
        if flat_g.dtype == np.float32:
            g_uint = flat_g.view(np.uint32)
            r_uint = flat_r.view(np.uint32)
        elif flat_g.dtype == np.float16:
            g_uint = flat_g.view(np.uint16)
            r_uint = flat_r.view(np.uint16)
        elif flat_g.dtype in (np.uint8, np.uint16, np.uint32):
            g_uint = flat_g
            r_uint = flat_r
        else:
            # 默认转 float32
            g_uint = flat_g.astype(np.float32).view(np.uint32)
            r_uint = flat_r.astype(np.float32).view(np.uint32)

        # XOR
        xor = g_uint ^ r_uint
        diff_mask = xor != 0
        diff_elements = int(np.count_nonzero(diff_mask))

        # 统计差异 bit 数（popcount）
        diff_bits = int(np.unpackbits(xor.view(np.uint8)).sum())
        total_bits = g_uint.size * g_uint.itemsize * 8

        return BitXorResult(
            total_elements=total_elements,
            diff_elements=diff_elements,
            total_bits=total_bits,
            diff_bits=diff_bits,
        )

    @staticmethod
    def compare_bytes(
        golden: bytes,
        result: bytes,
    ) -> BitXorResult:
        """
        Bit XOR 比对（字节级，静态方法）

        Args:
            golden: 参考字节数据
            result: 待比对字节数据

        Returns:
            BitXorResult
        """
        if len(golden) != len(result):
            raise ValueError(f"Length mismatch: {len(golden)} vs {len(result)}")

        g_arr = np.frombuffer(golden, dtype=np.uint8)
        r_arr = np.frombuffer(result, dtype=np.uint8)

        return BitXorStrategy.compare(g_arr, r_arr)

    def run(self, ctx: CompareContext) -> BitXorResult:
        """执行 Bit XOR 比对（Strategy 协议方法）"""
        return self.compare(ctx.golden, ctx.dut)

    @staticmethod
    def print_result(result: BitXorResult, name: str = ""):
        """打印 Bit XOR 比对结果（静态方法）"""
        title = f"[{name}] Bit XOR Comparison" if name else "Bit XOR Comparison"
        print(f"\n{title}")
        print("=" * 60)
        print(f"  Total elements:    {result.total_elements:,}")
        print(f"  Diff elements:     {result.diff_elements:,} ({result.diff_element_ratio:.2%})")
        print(f"  Total bits:        {result.total_bits:,}")
        print(f"  Diff bits:         {result.diff_bits:,} ({result.diff_bit_ratio:.4%})")
        print("=" * 60)
        print()

    @staticmethod
    def visualize(result: BitXorResult) -> "Page":
        """
        BitXor 策略级可视化

        体现比对原理：bit-by-bit XOR（无语义理解）
        - 元素差异率
        - bit 差异率
        """
        from aidevtools.compare.visualizer import Visualizer

        try:
            from pyecharts.charts import Gauge
            from pyecharts import options as opts
        except ImportError:
            raise ImportError("pyecharts is required for visualization. Install: pip install pyecharts")

        page = Visualizer.create_page(title="Bit XOR Analysis Report")

        # 1. 元素差异率仪表盘
        elem_diff_pct = result.diff_element_ratio * 100
        gauge_elem = (
            Gauge()
            .add("", [("Diff Element %", elem_diff_pct)])
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"Element Diff Rate ({result.diff_elements}/{result.total_elements})"),
            )
        )
        page.add(gauge_elem)

        # 2. Bit 差异率仪表盘
        bit_diff_pct = result.diff_bit_ratio * 100
        gauge_bit = (
            Gauge()
            .add("", [("Diff Bit %", bit_diff_pct)])
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"Bit Diff Rate ({result.diff_bits}/{result.total_bits})"),
            )
        )
        page.add(gauge_bit)

        # 3. 统计对比
        bar = Visualizer.create_bar(
            ["Elements", "Bits"],
            {
                "Total": [result.total_elements, result.total_bits],
                "Diff": [result.diff_elements, result.diff_bits],
            },
            title="XOR Statistics",
        )
        page.add(bar)

        return page

    @property
    def name(self) -> str:
        return "bit_xor"


__all__ = [
    "BitXorResult",
    "BitXorStrategy",
]
