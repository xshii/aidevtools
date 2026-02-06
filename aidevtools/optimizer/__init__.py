"""
Optimizer 优化器

用于融合策略评估和优化的框架。

两种前端方式:
1. PyTorch 劫持 (推荐): 自动从计算图提取 Benchmark
2. 手动构建: Benchmark 链式 API

使用示例 (PyTorch 劫持方式):
```python
import aidevtools.golden as golden
import torch.nn.functional as F

# 执行 PyTorch 代码 (自动被劫持)
x = torch.randn(512, 768)
y = F.linear(x, w1)
y = F.gelu(y)
y = F.linear(y, w2)

# 自动提取为 Benchmark 并评估
from aidevtools.optimizer import extract_and_evaluate
result = extract_and_evaluate("my_ffn")
print(result.summary())
```

使用示例 (手动构建方式):
```python
from aidevtools.optimizer import Benchmark, FusionEvaluator

# 链式构建
bm = (
    Benchmark("my_ffn")
    .add_op("mm1", "matmul", M=512, N=3072, K=768)
    .add_op("gelu", "gelu", M=512, N=3072)
    .add_op("mm2", "matmul", M=512, N=768, K=3072)
)

# 评估
evaluator = FusionEvaluator()
result = evaluator.evaluate(bm)
print(result.summary())
```

ML 校准示例:
```python
from aidevtools.optimizer import MeasurementArchive, calibrate_and_compare

# 导入实测数据
archive = MeasurementArchive()
archive.import_results([("bm1", 125.5), ("bm2", 98.2)], suite)

# 校准并对比
result = calibrate_and_compare(archive)
print(result.summary())
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

# Comparison (理论 vs 工程化)
from .comparison import (
    PredictMethod,
    PredictionResult,
    ComparisonMetrics,
    MethodComparison,
    MethodComparator,
    CalibrateAndCompareResult,
    compare_methods,
    calibrate_and_compare,
)

# Bridge (PyTorch 劫持 → Benchmark)
from .bridge import (
    extract_benchmark,
    extract_and_evaluate,
    trace_and_extract,
    TracedBenchmark,
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

    # Comparison
    "PredictMethod",
    "PredictionResult",
    "ComparisonMetrics",
    "MethodComparison",
    "MethodComparator",
    "CalibrateAndCompareResult",
    "compare_methods",
    "calibrate_and_compare",

    # Bridge (PyTorch → Benchmark)
    "extract_benchmark",
    "extract_and_evaluate",
    "trace_and_extract",
    "TracedBenchmark",
]
