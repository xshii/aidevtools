"""
比对引擎 (重构版)

使用策略模式，灵活组合不同的比对策略。

架构设计:
- CompareEngine: 执行引擎，负责调度策略
- CompareStrategy: 策略接口，具体比对逻辑
- CompareContext: 上下文，携带共享数据
- _PreparedPair: 预处理缓存，避免重复计算

优势:
- 易于扩展新策略
- 策略可灵活组合
- 解耦比对逻辑
- 易于单元测试
"""

from typing import Optional, List, Dict, Any
import numpy as np

from .types import CompareConfig, CompareStatus, _PreparedPair
from .strategy import (
    CompareStrategy,
    CompareContext,
    StandardStrategy,
)


class CompareEngine:
    """
    比对引擎

    使用示例:
        # 方式1: 使用预定义策略
        engine = CompareEngine.standard()
        results = engine.run(dut, golden)

        # 方式2: 自定义策略
        from aidevtools.compare.strategy import ExactStrategy, FuzzyStrategy
        engine = CompareEngine([
            ExactStrategy(),
            FuzzyStrategy(),
        ])
        results = engine.run(dut, golden)

        # 方式3: 使用组合策略
        from aidevtools.compare.strategy import DeepAnalysisStrategy
        engine = CompareEngine(DeepAnalysisStrategy())
        results = engine.run(dut, golden)
    """

    def __init__(
        self,
        strategy: Optional[CompareStrategy] = None,
        config: Optional[CompareConfig] = None,
    ):
        """
        Args:
            strategy: 比对策略（可以是单个策略或组合策略）
            config: 比对配置
        """
        self.strategy = strategy or StandardStrategy()
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
            metadata: 额外元数据

        Returns:
            {strategy_name: result} 字典
        """
        # 创建上下文
        ctx = CompareContext(
            golden=golden,
            dut=dut,
            config=self.config,
            golden_qnt=golden_qnt,
            metadata=metadata,
        )

        # 预处理数据（创建PreparedPair缓存）
        ctx.prepared = _PreparedPair.from_arrays(golden, dut)

        # 执行策略预处理
        self.strategy.prepare(ctx)

        # 执行比对
        return self.strategy.run(ctx)

    def determine_status(
        self,
        dut_pass: bool,
        golden_sanity_pass: bool,
    ) -> CompareStatus:
        """
        判定最终状态

        状态矩阵:
            DUT vs Golden | Golden 自检 | 判定状态
            --------------|-------------|---------------
            PASS          | PASS        | PASS
            PASS          | FAIL        | GOLDEN_SUSPECT
            FAIL          | PASS        | DUT_ISSUE
            FAIL          | FAIL        | BOTH_SUSPECT

        Args:
            dut_pass: DUT比对是否通过
            golden_sanity_pass: Golden自检是否通过

        Returns:
            CompareStatus
        """
        if dut_pass and golden_sanity_pass:
            return CompareStatus.PASS
        elif dut_pass and not golden_sanity_pass:
            return CompareStatus.GOLDEN_SUSPECT
        elif not dut_pass and golden_sanity_pass:
            return CompareStatus.DUT_ISSUE
        else:
            return CompareStatus.BOTH_SUSPECT

    # ========================================================================
    # 便捷工厂方法
    # ========================================================================

    @classmethod
    def standard(cls, config: Optional[CompareConfig] = None) -> "CompareEngine":
        """
        创建标准比对引擎

        包含策略:
        - 精确比对
        - 模糊比对（纯FP32）
        - 模糊比对（量化感知）
        - Golden自检
        """
        from .strategy import StandardStrategy
        return cls(strategy=StandardStrategy(), config=config)

    @classmethod
    def quick(cls, config: Optional[CompareConfig] = None) -> "CompareEngine":
        """
        创建快速检查引擎

        包含策略:
        - 精确比对
        - 模糊比对（纯FP32）
        """
        from .strategy import QuickCheckStrategy
        return cls(strategy=QuickCheckStrategy(), config=config)

    @classmethod
    def deep(
        cls,
        config: Optional[CompareConfig] = None,
        block_size: int = 1024,
    ) -> "CompareEngine":
        """
        创建深度分析引擎

        包含策略:
        - 精确比对
        - 模糊比对（纯FP32）
        - 模糊比对（量化感知）
        - Golden自检
        - Bit级分析
        - 分块分析
        """
        from .strategy import DeepAnalysisStrategy
        return cls(strategy=DeepAnalysisStrategy(block_size=block_size), config=config)

    @classmethod
    def minimal(cls, config: Optional[CompareConfig] = None) -> "CompareEngine":
        """
        创建最小比对引擎（性能优先）

        包含策略:
        - 模糊比对（纯FP32）
        """
        from .strategy import MinimalStrategy
        return cls(strategy=MinimalStrategy(), config=config)

    @classmethod
    def progressive(
        cls,
        config: Optional[CompareConfig] = None,
        block_size: int = 64,
    ) -> "CompareEngine":
        """
        创建渐进式分级引擎

        三级策略:
        - L1: Exact + Bitwise（初步检查）
        - L2: Fuzzy + Sanity（中度诊断）
        - L3: BitwisePro + Blocked（深度定位）

        适用场景:
        - 单算子渐进式分析
        - 不确定错误类型

        注意:
        - 每个算子独立分级
        - 如需模型级协调，使用 CompareEngine.model_progressive()
        """
        from .strategy import ProgressiveStrategy
        return cls(strategy=ProgressiveStrategy(block_size=block_size), config=config)

    @classmethod
    def quick_then_deep(
        cls,
        config: Optional[CompareConfig] = None,
        block_size: int = 64,
    ) -> "CompareEngine":
        """
        创建快速检查后深度分析引擎

        两级策略:
        - L1: Exact + Bitwise（快速判断）
        - L2: Fuzzy + Sanity + Blocked（深度分析，如果L1失败）

        适用场景:
        - 预期大部分数据一致
        - 需要快速筛选问题
        """
        from .strategy import QuickThenDeepStrategy
        return cls(strategy=QuickThenDeepStrategy(block_size=block_size), config=config)

    @staticmethod
    def model_progressive(config: Optional[CompareConfig] = None) -> "ModelTieredAnalyzer":
        """
        创建模型级渐进式分析器

        支持全局协调的三级分析:
        - L1: 所有算子快速检查
        - L2: 失败算子中度分析（可根据L1通过率跳过）
        - L3: 仍失败的深度分析（可根据L2通过率跳过）

        适用场景:
        - 大模型调试（几十到几百个算子）
        - 需要节省计算资源
        - 需要全局协调决策

        Returns:
            ModelTieredAnalyzer 实例

        示例:
            analyzer = CompareEngine.model_progressive()
            results = analyzer.progressive_analyze(
                per_op_pairs,
                l1_threshold=0.85,  # 85%通过则停止
            )
            analyzer.print_summary(results)
        """
        from .model import ModelTieredAnalyzer
        return ModelTieredAnalyzer(config=config)


# ============================================================================
# 便捷函数
# ============================================================================


def compare_full(
    dut: np.ndarray,
    golden: np.ndarray,
    golden_qnt: Optional[np.ndarray] = None,
    config: Optional[CompareConfig] = None,
) -> Dict[str, Any]:
    """
    一键完整比对（使用标准策略）

    Args:
        dut: DUT输出
        golden: Golden数据
        golden_qnt: Golden量化数据（可选）
        config: 比对配置

    Returns:
        比对结果字典
    """
    engine = CompareEngine.standard(config=config)
    return engine.run(dut, golden, golden_qnt)


def compare_quick(
    dut: np.ndarray,
    golden: np.ndarray,
    config: Optional[CompareConfig] = None,
) -> Dict[str, Any]:
    """
    快速比对

    Args:
        dut: DUT输出
        golden: Golden数据
        config: 比对配置

    Returns:
        比对结果字典
    """
    engine = CompareEngine.quick(config=config)
    return engine.run(dut, golden)
