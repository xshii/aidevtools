"""
精确比对策略

提供bit级精确比对功能。
"""
import numpy as np

from .base import CompareStrategy, CompareContext
from ..types import ExactResult, _PreparedPair


# ============================================================================
# 底层实现函数
# ============================================================================


def _compare_exact_prepared(
    p: _PreparedPair,
    golden_orig: np.ndarray,
    result_orig: np.ndarray,
    max_abs: float = 0.0,
    max_count: int = 0,
) -> ExactResult:
    """使用预处理数据的精确比对 — 复用 p.abs_err 避免重复计算

    当 max_abs > 0 时，直接使用 p.abs_err 而不再重新计算 |g - r|。
    当 max_abs == 0 时，仍需原始数组做字节级比对。
    """
    max_abs_actual = float(p.abs_err.max()) if p.total > 0 else 0.0

    if max_abs == 0:
        # bit 级精确比对 (需要 contiguous 数组)
        g_cont = np.ascontiguousarray(golden_orig)
        r_cont = np.ascontiguousarray(result_orig)
        mismatch_mask = g_cont.view(np.uint8) != r_cont.view(np.uint8)
        mismatch_count = int(np.sum(mismatch_mask))
        first_diff = int(np.argmax(mismatch_mask)) if mismatch_count > 0 else -1
    else:
        # 允许一定误差 — 复用 p.abs_err
        exceed_mask = p.abs_err > max_abs
        mismatch_count = int(np.sum(exceed_mask))
        first_diff = int(np.argmax(exceed_mask)) if mismatch_count > 0 else -1

    return ExactResult(
        passed=mismatch_count <= max_count,
        mismatch_count=mismatch_count,
        first_diff_offset=first_diff,
        max_abs=max_abs_actual,
        total_elements=p.total,
    )


# ============================================================================
# 策略类
# ============================================================================


class ExactStrategy(CompareStrategy):
    """
    精确比对策略

    检查DUT和Golden是否完全一致（bit级）。

    使用场景：
    - 验证确定性算子（如整数运算）
    - 检测是否有任何量化误差

    使用方式：
        # 方式1: 直接调用静态方法
        result = ExactStrategy.compare(golden, dut)
        is_same = ExactStrategy.compare_bytes(golden_bytes, dut_bytes)

        # 方式2: 通过引擎
        engine = CompareEngine(ExactStrategy())
        result = engine.run(dut, golden)
    """

    def __init__(self, use_bit_compare: bool = False):
        """
        Args:
            use_bit_compare: 是否使用bit级对比（更严格）
        """
        self.use_bit_compare = use_bit_compare

    @staticmethod
    def compare(
        golden: np.ndarray,
        result: np.ndarray,
        max_abs: float = 0.0,
        max_count: int = 0,
    ) -> ExactResult:
        """
        精确比对（静态方法）

        Args:
            golden: golden 数据
            result: 待比对数据
            max_abs: 允许的最大绝对误差 (0=bit级精确)
            max_count: 允许超阈值的元素个数

        Returns:
            ExactResult
        """
        p = _PreparedPair.from_arrays(golden, result)
        return _compare_exact_prepared(p, golden, result, max_abs, max_count)

    @staticmethod
    def compare_bytes(golden: bytes, result: bytes) -> bool:
        """
        bit 级对比，完全一致（静态方法）

        Args:
            golden: golden 字节数据
            result: 待比对字节数据

        Returns:
            是否完全一致
        """
        return golden == result

    def run(self, ctx: CompareContext) -> ExactResult:
        """执行精确比对（Strategy 协议方法）"""
        if self.use_bit_compare:
            # 对于字节比对，需要先转换
            g_bytes = ctx.golden.tobytes()
            r_bytes = ctx.dut.tobytes()
            passed = self.compare_bytes(g_bytes, r_bytes)
            return ExactResult(
                passed=passed,
                mismatch_count=0 if passed else 1,
                first_diff_offset=-1 if passed else 0,
                max_abs=0.0,
                total_elements=ctx.golden.size,
            )
        else:
            return self.compare(
                ctx.golden,
                ctx.dut,
                max_abs=ctx.config.exact_max_abs,
                max_count=ctx.config.exact_max_count,
            )

    @staticmethod
    def visualize(result: ExactResult) -> "Page":
        """
        Exact 策略级可视化

        体现比对原理：精确匹配
        - 通过率仪表盘
        - 误差分布（如有不匹配）
        """
        from aidevtools.compare.visualizer import Visualizer

        try:
            from pyecharts.charts import Gauge
            from pyecharts import options as opts
        except ImportError:
            raise ImportError("pyecharts is required for visualization. Install: pip install pyecharts")

        page = Visualizer.create_page(title="Exact Analysis Report")

        # 1. 通过率仪表盘
        pass_rate = (1 - result.mismatch_count / result.total_elements) * 100 if result.total_elements > 0 else 100.0

        gauge = (
            Gauge()
            .add("", [("Pass Rate", pass_rate)])
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"Exact Match ({result.mismatch_count}/{result.total_elements} mismatch)"),
            )
        )
        page.add(gauge)

        # 2. 状态饼图
        status_data = {
            "✅ Match": result.total_elements - result.mismatch_count,
            "❌ Mismatch": result.mismatch_count,
        }
        pie = Visualizer.create_pie(status_data, title="Match Status")
        page.add(pie)

        # 3. 关键指标
        if result.mismatch_count > 0:
            metric_data = {
                "First Diff Offset": result.first_diff_offset,
                "Max Abs Diff": result.max_abs,
            }
            bar = Visualizer.create_bar(
                list(metric_data.keys()),
                {"Value": [float(v) for v in metric_data.values()]},
                title="Error Metrics",
            )
            page.add(bar)

        return page

    @property
    def name(self) -> str:
        return "exact" if not self.use_bit_compare else "exact_bit"
