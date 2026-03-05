"""
比对引擎

使用策略模式，灵活组合不同的比对策略。

使用:
    # 渐进式分级 (默认)
    engine = CompareEngine.progressive()
    results = engine.run(dut, golden)

    # 深度模式 (三级全执行)
    engine = CompareEngine.progressive(deep=True)
    results = engine.run(dut, golden)

    # 自定义策略
    engine = CompareEngine(CompositeStrategy([ExactStrategy(), FuzzyStrategy()]))
    results = engine.run(dut, golden)
"""

from typing import Optional, Dict, Any
import numpy as np

from .types import CompareConfig, _PreparedPair
from .strategy import CompareStrategy, CompareContext


class CompareEngine:
    """比对引擎"""

    def __init__(
        self,
        strategy: Optional[CompareStrategy] = None,
        config: Optional[CompareConfig] = None,
    ):
        if strategy is None:
            from .strategy import ProgressiveStrategy
            strategy = ProgressiveStrategy()
        self.strategy = strategy
        self.config = config or CompareConfig()

    def run(
        self,
        dut: np.ndarray,
        golden: np.ndarray,
        golden_qnt: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        执行比对

        Args:
            dut: DUT输出数据
            golden: Golden数据（纯FP32/FP64）
            golden_qnt: Golden量化数据（可选）
            metadata: 额外元数据（_raw_golden/_raw_dut 用于双路径比对）

        Returns:
            {strategy_name: result} 字典
        """
        # 从 metadata 提取 raw 数据 (双路径: 源格式字节 + fp32)
        raw_golden = metadata.pop("_raw_golden", None) if metadata else None
        raw_dut = metadata.pop("_raw_dut", None) if metadata else None

        ctx = CompareContext(
            golden=golden,
            dut=dut,
            config=self.config,
            golden_qnt=golden_qnt,
            raw_golden=raw_golden,
            raw_dut=raw_dut,
            metadata=metadata,
        )

        ctx.prepared = _PreparedPair.from_arrays(golden, dut)
        self.strategy.prepare(ctx)
        return self.strategy.run(ctx)

    # ========================================================================
    # 工厂方法
    # ========================================================================

    @classmethod
    def progressive(
        cls,
        config: Optional[CompareConfig] = None,
        block_size: int = 64,
        deep: bool = False,
    ) -> "CompareEngine":
        """
        渐进式分级引擎

        三级策略:
        - L1: Exact（初步检查）
        - L2: Fuzzy + Sanity（中度诊断）
        - L3: BitAnalysis + Blocked（深度定位）

        Args:
            deep: True 时三级全部执行，不做早停
        """
        from .strategy import ProgressiveStrategy
        return cls(
            strategy=ProgressiveStrategy(block_size=block_size, deep=deep),
            config=config,
        )

    @staticmethod
    def model_progressive(config: Optional[CompareConfig] = None) -> "ModelTieredAnalyzer":
        """
        模型级渐进式分析器

        全局协调的三级分析，根据整体通过率决定是否深入下一级。
        适用于大模型调试（几十到几百个算子）。
        """
        from .model import ModelTieredAnalyzer
        return ModelTieredAnalyzer(config=config)
