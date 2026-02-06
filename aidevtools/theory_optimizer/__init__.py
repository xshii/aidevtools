"""
Theory Optimizer 理论优化器

用于融合策略评估和优化的框架。

设计模式:
- Facade 模式: FusionEvaluator 提供统一入口
- Strategy 模式: 可插拔的 Tiling 策略
- Builder 模式: Benchmark 链式构建
- Factory 模式: BenchmarkSuite 预定义用例
- Singleton 模式: FusionRules 全局融合规则
- Observer 模式: 参数更新通知
- Composite 模式: CostParameters 参数组合
- Template Method 模式: View 渲染流程
- Registry 模式: 策略/视图注册

模块结构:
- benchmark: Benchmark 定义和构建
- fusion_rules: 全局融合规则配置
- cost_model: 参数化成本模型
- calibration: ML 校准框架
- strategy: Tiling 策略
- views: 可视化视图
- memory_plan: 内存规划和 DMA 生成
- evaluator: 评估门面

使用示例:
```python
from aidevtools.theory_optimizer import (
    FusionEvaluator,
    Benchmark,
    BenchmarkSuite,
    FusionRules,
    get_fusion_rules,
)

# 查看/修改全局融合规则
rules = get_fusion_rules()
print(rules.summary())

# 添加自定义规则
from aidevtools.theory_optimizer import FusionRule
rules.add_rule(FusionRule(
    op_type_a="custom_op",
    op_type_b="matmul",
    ratio=0.95,
    fuse_speedup=1.2,
))

# 使用预定义 suite (融合规则自动从全局获取)
evaluator = FusionEvaluator()
result = evaluator.evaluate_suite("bert_ffn", seq_len=512, hidden=768, intermediate=3072)
print(result.summary())

# 自定义 benchmark (融合规则自动匹配)
benchmark = (
    Benchmark("custom")
    .add_op("matmul1", "matmul", M=512, N=768, K=768)
    .add_op("gelu", "gelu", M=512, N=768)
    .add_op("matmul2", "matmul", M=512, N=768, K=768)
    # 可选：覆盖特定算子对的规则
    .override_pair("matmul1", "gelu", ratio=0.92, fuse_speedup=1.35)
)
print(benchmark.summary())

# 策略比较
compare = evaluator.compare(benchmark, strategies=["baseline", "efficiency_aware", "fuse_speedup"])
print(compare.summary())

# 生成视图
roofline = evaluator.generate_view(result.tiling_result, "roofline")
print(roofline.content)

# 校准
evaluator.import_measurements("measurements.csv")
evaluator.calibrate(method="layered")
evaluator.save_calibration("calibrated_params.json")
```
"""

# Benchmark
from .benchmark import (
    OpType,
    OpSpec,
    FusePair,
    Benchmark,
    BenchmarkSuite,
    ComputeUnitCfg,
)

# Fusion Rules (全局配置)
from .fusion_rules import (
    FusionRule,
    FusionPattern,
    FusionRules,
    FusionHyperParams,
    FuseConstraint,
    get_fusion_rules,
    can_fuse,
    get_fuse_ratio,
    get_fuse_speedup,
)

# Cost Model
from .cost_model import (
    CostParameters,
    ComputeEfficiencyParams,
    MemoryEfficiencyParams,
    OverheadParams,
    FuseParams,
    CostResult,
    ParameterizedCostModel,
)

# Calibration
from .calibration import (
    MeasurementRecord,
    FitResult,
    EvalMetrics,
    MLCalibrator,
    LinearFitStrategy,
    LayeredFitStrategy,
    GradientFitStrategy,
)

# Measurement Archive (ML 数据归档)
from .measurement_archive import (
    MeasurementSource,
    FeatureVector,
    LabelVector,
    MeasurementSample,
    MeasurementArchive,
    create_sample_from_benchmark,
)

# Hyper Calibrator (超参数校准)
from .hyper_calibrator import (
    OptimizeMethod,
    CalibrationResult,
    HyperCalibrator,
    calibrate_from_archive,
    quick_calibrate,
)

# Strategy
from .strategy import (
    TileConfig,
    TilingStrategy,
    BaselineStrategy,
    EfficiencyAwareStrategy,
    FuseSpeedupStrategy,
)
from .strategy.base import (
    TilingResult,
    FusionConfig,
    StrategyRegistry,
)

# Views
from .views import (
    View,
    ViewResult,
    MemoryFlowView,
    ComputeView,
    RooflineView,
    BandwidthPipelineView,
)
from .views.base import ViewFormat, ViewRegistry

# Memory Plan
from .memory_plan import (
    MemoryLevel,
    DMADirection,
    MemoryRegion,
    TensorAllocation,
    DMAOp,
    MemoryPlan,
    MemoryPlanBuilder,
)

# Evaluator
from .evaluator import (
    EvalResult,
    CompareResult,
    FusionEvaluator,
    quick_evaluate,
    quick_compare,
)

__version__ = "0.1.0"

__all__ = [
    # Benchmark
    "OpType",
    "OpSpec",
    "FusePair",
    "Benchmark",
    "BenchmarkSuite",
    "ComputeUnitCfg",

    # Fusion Rules
    "FusionRule",
    "FusionPattern",
    "FusionRules",
    "FusionHyperParams",
    "FuseConstraint",
    "get_fusion_rules",
    "can_fuse",
    "get_fuse_ratio",
    "get_fuse_speedup",

    # Cost Model
    "CostParameters",
    "ComputeEfficiencyParams",
    "MemoryEfficiencyParams",
    "OverheadParams",
    "FuseParams",
    "CostResult",
    "ParameterizedCostModel",

    # Calibration
    "MeasurementRecord",
    "FitResult",
    "EvalMetrics",
    "MLCalibrator",
    "LinearFitStrategy",
    "LayeredFitStrategy",
    "GradientFitStrategy",

    # Measurement Archive
    "MeasurementSource",
    "FeatureVector",
    "LabelVector",
    "MeasurementSample",
    "MeasurementArchive",
    "create_sample_from_benchmark",

    # Hyper Calibrator
    "OptimizeMethod",
    "CalibrationResult",
    "HyperCalibrator",
    "calibrate_from_archive",
    "quick_calibrate",

    # Strategy
    "TileConfig",
    "TilingStrategy",
    "TilingResult",
    "FusionConfig",
    "StrategyRegistry",
    "BaselineStrategy",
    "EfficiencyAwareStrategy",
    "FuseSpeedupStrategy",

    # Views
    "View",
    "ViewResult",
    "ViewFormat",
    "ViewRegistry",
    "MemoryFlowView",
    "ComputeView",
    "RooflineView",
    "BandwidthPipelineView",

    # Memory Plan
    "MemoryLevel",
    "DMADirection",
    "MemoryRegion",
    "TensorAllocation",
    "DMAOp",
    "MemoryPlan",
    "MemoryPlanBuilder",

    # Evaluator
    "EvalResult",
    "CompareResult",
    "FusionEvaluator",
    "quick_evaluate",
    "quick_compare",
]
