"""
模糊比对策略

提供基于QSNR、余弦相似度的模糊比对功能。
"""
from .base import CompareStrategy, CompareContext
from ..metrics import calc_all_metrics_early_exit, _calc_all_metrics_early_exit_prepared
from ..types import FuzzyResult, CompareConfig, _PreparedPair


# ============================================================================
# 底层实现函数
# ============================================================================


def _compare_fuzzy_prepared(
    p: _PreparedPair,
    config: CompareConfig = None,
) -> FuzzyResult:
    """使用预处理数据的模糊比对 — 避免重复 astype + flatten"""
    if config is None:
        config = CompareConfig()

    m = _calc_all_metrics_early_exit_prepared(
        p,
        atol=config.fuzzy_atol,
        rtol=config.fuzzy_rtol,
        min_qsnr=config.fuzzy_min_qsnr,
        min_cosine=config.fuzzy_min_cosine,
        max_exceed_ratio=config.fuzzy_max_exceed_ratio,
    )

    exceed_ratio = m.exceed_count / m.total_elements if m.total_elements > 0 else 0.0

    passed = (
        exceed_ratio <= config.fuzzy_max_exceed_ratio
        and m.qsnr >= config.fuzzy_min_qsnr
        and m.cosine >= config.fuzzy_min_cosine
    )

    return FuzzyResult(
        passed=passed,
        max_abs=m.max_abs,
        mean_abs=m.mean_abs,
        max_rel=m.max_rel,
        qsnr=m.qsnr,
        cosine=m.cosine,
        total_elements=m.total_elements,
        exceed_count=m.exceed_count,
    )


# ============================================================================
# 策略类
# ============================================================================


class FuzzyStrategy(CompareStrategy):
    """
    模糊比对策略

    基于统计指标（QSNR、余弦相似度）判断相似度。

    使用场景：
    - 量化算子比对（允许一定误差）
    - 浮点运算验证

    使用方式：
        # 方式1: 直接调用静态方法
        result = FuzzyStrategy.compare(golden, dut, config)
        result = FuzzyStrategy.compare_isclose(golden, dut, atol=1e-5)

        # 方式2: 通过引擎
        engine = CompareEngine(FuzzyStrategy())
        result = engine.run(dut, golden)
    """

    def __init__(self, use_golden_qnt: bool = True, enable_early_exit: bool = True):
        """
        Args:
            use_golden_qnt: 是否使用golden_qnt进行量化感知比对
            enable_early_exit: 是否启用early exit优化
        """
        self.use_golden_qnt = use_golden_qnt
        self.enable_early_exit = enable_early_exit

    @staticmethod
    def compare(
        golden,
        result,
        config=None,
    ) -> FuzzyResult:
        """
        模糊比对（静态方法，单次遍历 + early exit 优化版）

        判定条件:
        1. 超阈值元素比例 <= max_exceed_ratio
        2. QSNR >= min_qsnr
        3. Cosine >= min_cosine

        优化:
        - 单次 flatten + float64 转换 (原先 4 次)
        - cosine 不达标 → early exit
        - exceed 不达标 → early exit
        - 大数组 2-3x 提速, 失败 case 额外 2-3x
        """
        if config is None:
            config = CompareConfig()

        m = calc_all_metrics_early_exit(
            golden, result,
            atol=config.fuzzy_atol,
            rtol=config.fuzzy_rtol,
            min_qsnr=config.fuzzy_min_qsnr,
            min_cosine=config.fuzzy_min_cosine,
            max_exceed_ratio=config.fuzzy_max_exceed_ratio,
        )

        exceed_ratio = m.exceed_count / m.total_elements if m.total_elements > 0 else 0.0

        passed = (
            exceed_ratio <= config.fuzzy_max_exceed_ratio
            and m.qsnr >= config.fuzzy_min_qsnr
            and m.cosine >= config.fuzzy_min_cosine
        )

        return FuzzyResult(
            passed=passed,
            max_abs=m.max_abs,
            mean_abs=m.mean_abs,
            max_rel=m.max_rel,
            qsnr=m.qsnr,
            cosine=m.cosine,
            total_elements=m.total_elements,
            exceed_count=m.exceed_count,
        )

    @staticmethod
    def compare_isclose(golden, result, atol=1e-5, rtol=1e-3, max_exceed_ratio=0.0) -> FuzzyResult:
        """
        IsClose 比对（静态方法） - 类似 numpy.isclose

        判断条件: |result - golden| <= atol + rtol * |golden|
        通过条件: exceed_ratio <= max_exceed_ratio
        """
        config = CompareConfig(
            fuzzy_atol=atol,
            fuzzy_rtol=rtol,
            fuzzy_max_exceed_ratio=max_exceed_ratio,
            fuzzy_min_qsnr=0.0,
            fuzzy_min_cosine=0.0,
        )
        return FuzzyStrategy.compare(golden, result, config)

    def run(self, ctx: CompareContext) -> FuzzyResult:
        """执行模糊比对（Strategy 协议方法）"""
        golden_ref = ctx.golden_qnt if (self.use_golden_qnt and ctx.golden_qnt is not None) else ctx.golden
        return self.compare(golden_ref, ctx.dut, config=ctx.config)

    @property
    def name(self) -> str:
        suffix = "_qnt" if self.use_golden_qnt else "_pure"
        return f"fuzzy{suffix}"
