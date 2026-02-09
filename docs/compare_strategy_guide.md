# Compare 策略模式使用指南

> 版本: 1.0
> 最后更新: 2026-02-09
> 适用于: aidevtools >= 1.0.0

## 概述

aidevtools 的 Compare 模块采用策略模式设计，提供灵活、可扩展的比对能力。本文档介绍如何使用预定义策略、组合自定义策略，以及如何开发新策略。

## 策略架构

```
CompareEngine
    ↓
CompareStrategy (接口)
    ├── ExactStrategy          # 精确比对
    ├── FuzzyStrategy          # 模糊比对
    ├── SanityStrategy         # Golden 自检
    ├── BlockedStrategy        # 分块定位
    ├── BitAnalysisStrategy    # Bit 级分析
    ├── BitXorStrategy         # Bit XOR 分析
    ├── CompositeStrategy      # 组合策略
    │   ├── StandardStrategy   # 标准组合
    │   ├── QuickCheckStrategy # 快速检查
    │   ├── DeepAnalysisStrategy # 深度分析
    │   └── MinimalStrategy    # 最小策略
    └── TieredStrategy         # 分级策略
        ├── ProgressiveStrategy     # 渐进式
        └── QuickThenDeepStrategy   # 快速+深度
```

## 预定义策略

### 1. StandardStrategy (推荐)

适用于日常开发和回归测试的完整比对。

```python
from aidevtools.compare import CompareEngine, CompareConfig

config = CompareConfig(
    fuzzy_min_qsnr=30.0,
    fuzzy_min_cosine=0.999,
)

engine = CompareEngine.standard(config=config)
result = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)
```

**包含策略**:
- ✅ ExactStrategy: Bit 级精确比对
- ✅ FuzzyStrategy(Pure): 纯 FP32 模糊比对
- ✅ FuzzyStrategy(Qnt): 量化感知模糊比对
- ✅ SanityStrategy: Golden 自检

**输出**:
```python
{
    'exact': ExactResult,
    'fuzzy_pure': FuzzyResult,
    'fuzzy_qnt': FuzzyResult,  # 如果提供 golden_qnt
    'sanity': SanityResult,
    'status': CompareStatus,   # 自动计算
}
```

### 2. QuickCheckStrategy

适用于 CI/CD 流程，快速验证基本正确性。

```python
engine = CompareEngine.quick(config=config)
result = engine.run(dut=dut, golden=golden)
```

**包含策略**:
- ✅ ExactStrategy
- ✅ FuzzyStrategy(Pure)

**特点**:
- 执行速度快
- 不包含 Golden 自检（节省时间）
- 适合快速迭代

### 3. DeepAnalysisStrategy

适用于深度调试和误差定位。

```python
engine = CompareEngine.deep(
    config=config,
    block_size=1024,  # 分块大小
)
result = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)
```

**包含策略**:
- ✅ ExactStrategy
- ✅ FuzzyStrategy(Pure)
- ✅ FuzzyStrategy(Qnt)
- ✅ SanityStrategy
- ✅ BlockedStrategy: 分块误差定位

**输出**:
```python
{
    'exact': ExactResult,
    'fuzzy_pure': FuzzyResult,
    'fuzzy_qnt': FuzzyResult,
    'sanity': SanityResult,
    'blocked': List[BlockResult],  # 分块结果
    'status': CompareStatus,
}
```

**使用示例**:
```python
result = engine.run(dut=dut, golden=golden)

# 查看分块结果
blocked = result.get('blocked')
if blocked:
    # 按误差排序，显示前10个最差的块
    sorted_blocks = sorted(blocked, key=lambda b: b.max_abs, reverse=True)
    for block in sorted_blocks[:10]:
        print(f"Block {block.block_id}: "
              f"offset={block.offset}, "
              f"size={block.size}, "
              f"max_abs={block.max_abs:.6e}")
```

### 4. MinimalStrategy

适用于性能敏感场景，最小开销。

```python
engine = CompareEngine.minimal(config=config)
result = engine.run(dut=dut, golden=golden)
```

**包含策略**:
- ✅ FuzzyStrategy(Pure) 仅此一项

**特点**:
- 最快的执行速度
- 最小的内存占用
- 适合大规模批量比对

### 5. ProgressiveStrategy (渐进式)

适用于单算子的分级诊断，根据前一级结果决定是否继续。

```python
engine = CompareEngine.progressive(
    config=config,
    block_size=64,
)
result = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)
```

**分级流程**:
```
L1: Exact + BitXor        # 初步检查
    ↓ (如果失败)
L2: Fuzzy + Sanity        # 中度诊断
    ↓ (如果失败)
L3: Blocked               # 深度定位
```

**特点**:
- 自动分级执行
- 通过则跳过后续级别
- 节省计算资源

**示例**:
```python
result = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)

# 查看执行了哪些级别
if result.get('exact'):
    print("L1 executed")
if result.get('fuzzy_pure'):
    print("L2 executed")
if result.get('blocked'):
    print("L3 executed")
```

### 6. QuickThenDeepStrategy

快速检查后深度分析，两级策略。

```python
engine = CompareEngine.quick_then_deep(
    config=config,
    block_size=64,
)
result = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)
```

**分级流程**:
```
L1: Exact + BitXor        # 快速判断
    ↓ (如果失败)
L2: Fuzzy + Sanity + Blocked  # 深度分析
```

## 自定义策略组合

### 方式1: 直接组合策略

```python
from aidevtools.compare import CompareEngine
from aidevtools.compare.strategy import (
    ExactStrategy,
    FuzzyStrategy,
    BitAnalysisStrategy,
)

# 创建策略列表
strategies = [
    ExactStrategy(),
    FuzzyStrategy(use_golden_qnt=False),
    BitAnalysisStrategy(format=FP32),
]

engine = CompareEngine(strategy=strategies, config=config)
result = engine.run(dut=dut, golden=golden)
```

### 方式2: 使用 CompositeStrategy

```python
from aidevtools.compare.strategy import CompositeStrategy

# 创建自定义组合
custom = CompositeStrategy(
    strategies=[
        ExactStrategy(),
        FuzzyStrategy(use_golden_qnt=True),
        SanityStrategy(),
    ],
    name="my_custom_strategy",
)

engine = CompareEngine(strategy=custom, config=config)
```

### 方式3: 条件组合

```python
def create_adaptive_strategy(dtype: str):
    """根据数据类型选择策略"""
    if dtype in ['bfp4', 'bfp8']:
        # 低精度量化 - 不做精确比对
        return CompositeStrategy(
            strategies=[
                FuzzyStrategy(use_golden_qnt=False),
                SanityStrategy(),
            ],
            name="low_precision",
        )
    elif dtype in ['fp16', 'bfloat16']:
        # 中等精度
        return StandardStrategy()
    else:
        # 高精度 - 包含精确比对
        return CompositeStrategy(
            strategies=[
                ExactStrategy(),
                FuzzyStrategy(use_golden_qnt=False),
            ],
            name="high_precision",
        )

# 使用
strategy = create_adaptive_strategy(dtype='bfp8')
engine = CompareEngine(strategy=strategy, config=config)
```

## 单独使用策略

每个策略都可以独立使用。

### ExactStrategy

```python
from aidevtools.compare.strategy import ExactStrategy
from aidevtools.compare import CompareConfig, CompareContext

strategy = ExactStrategy()
ctx = CompareContext(
    golden=golden,
    dut=dut,
    config=CompareConfig(exact_max_abs=1e-6),
)

result = strategy.run(ctx)
print(f"Passed: {result.passed}")
print(f"Mismatch count: {result.mismatch_count}")
print(f"Max abs: {result.max_abs:.6e}")
```

### FuzzyStrategy

```python
from aidevtools.compare.strategy import FuzzyStrategy

# 纯 FP32 模糊比对
strategy_pure = FuzzyStrategy(use_golden_qnt=False)

# 量化感知模糊比对
strategy_qnt = FuzzyStrategy(use_golden_qnt=True)

ctx = CompareContext(
    golden=golden,
    dut=dut,
    golden_qnt=golden_qnt,
    config=CompareConfig(fuzzy_min_qsnr=30.0),
)

result = strategy_pure.run(ctx)
print(f"QSNR: {result.qsnr:.2f} dB")
print(f"Cosine: {result.cosine:.6f}")
```

### BitAnalysisStrategy

```python
from aidevtools.compare.strategy import BitAnalysisStrategy, FP32, BFP8

# 静态方法调用（推荐）
result = BitAnalysisStrategy.compare(
    golden=golden,
    result=dut,
    fmt=FP32,  # 或 FP16, BFP8, BFP4 等
)

# 查看结果
print(f"Sign flip count: {result.summary.sign_flip_count}")
print(f"Exponent diff count: {result.summary.exponent_diff_count}")
print(f"Max exponent diff: {result.summary.max_exponent_diff}")

# 查看告警
for warning in result.warnings:
    print(f"[{warning.level.value}] {warning.message}")
    if warning.indices:
        print(f"  First few indices: {warning.indices[:5]}")
```

### BlockedStrategy

```python
from aidevtools.compare.strategy import BlockedStrategy

strategy = BlockedStrategy(block_size=256)
ctx = CompareContext(
    golden=golden,
    dut=dut,
    config=CompareConfig(),
)

blocks = strategy.run(ctx)

# 找出误差最大的块
worst_block = max(blocks, key=lambda b: b.max_abs)
print(f"Worst block: offset={worst_block.offset}, "
      f"max_abs={worst_block.max_abs:.6e}")
```

## 开发自定义策略

### 基础接口

```python
from aidevtools.compare.strategy import CompareStrategy, CompareContext
from typing import Any

class MyCustomStrategy(CompareStrategy):
    """自定义策略示例"""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "my_custom"

    def prepare(self, ctx: CompareContext) -> None:
        """
        预处理阶段（可选）
        可以在此进行数据准备、缓存等
        """
        # 访问上下文
        golden = ctx.golden
        dut = ctx.dut
        config = ctx.config

        # 预处理逻辑
        pass

    def run(self, ctx: CompareContext) -> Any:
        """
        执行策略

        Args:
            ctx: 比对上下文，包含 golden, dut, config 等

        Returns:
            任意结果对象
        """
        import numpy as np

        # 自定义比对逻辑
        diff = np.abs(ctx.golden - ctx.dut)
        max_diff = np.max(diff)
        passed = max_diff < self.threshold

        # 返回自定义结果
        return {
            'passed': passed,
            'max_diff': float(max_diff),
            'threshold': self.threshold,
        }
```

### 使用自定义策略

```python
# 创建策略实例
my_strategy = MyCustomStrategy(threshold=0.001)

# 使用策略
engine = CompareEngine(strategy=my_strategy, config=config)
result = engine.run(dut=dut, golden=golden)

# 查看结果
print(f"Passed: {result.get('passed')}")
print(f"Max diff: {result.get('max_diff'):.6e}")
```

### 组合到现有策略

```python
from aidevtools.compare.strategy import CompositeStrategy, ExactStrategy

# 将自定义策略与现有策略组合
combined = CompositeStrategy(
    strategies=[
        ExactStrategy(),
        MyCustomStrategy(threshold=0.001),
    ],
    name="combined_with_custom",
)

engine = CompareEngine(strategy=combined, config=config)
```

## 高级示例

### 示例1: 分精度策略选择

```python
from aidevtools.compare.strategy import CompositeStrategy

def get_strategy_for_dtype(dtype: str, config: CompareConfig):
    """根据数据类型返回合适的策略"""
    from aidevtools.compare.strategy import (
        ExactStrategy,
        FuzzyStrategy,
        SanityStrategy,
    )

    if dtype in ['bfp4', 'bfp8']:
        # 低精度 - 宽松阈值
        config.fuzzy_min_qsnr = 15.0
        config.fuzzy_min_cosine = 0.98
        return CompositeStrategy([
            FuzzyStrategy(use_golden_qnt=False),
            SanityStrategy(),
        ], name="low_precision")

    elif dtype in ['fp16', 'bfloat16']:
        # 中精度
        config.fuzzy_min_qsnr = 30.0
        config.fuzzy_min_cosine = 0.999
        return CompositeStrategy([
            FuzzyStrategy(use_golden_qnt=False),
            FuzzyStrategy(use_golden_qnt=True),
            SanityStrategy(),
        ], name="medium_precision")

    else:
        # 高精度 - 包含精确比对
        config.fuzzy_min_qsnr = 60.0
        config.fuzzy_min_cosine = 0.9999
        return CompositeStrategy([
            ExactStrategy(),
            FuzzyStrategy(use_golden_qnt=False),
        ], name="high_precision")

# 使用
config = CompareConfig()
strategy = get_strategy_for_dtype('bfp8', config)
engine = CompareEngine(strategy=strategy, config=config)
```

### 示例2: 条件执行策略

```python
from aidevtools.compare.strategy import CompareStrategy, CompareContext

class ConditionalStrategy(CompareStrategy):
    """条件执行策略 - 只在需要时执行深度分析"""

    def __init__(self, quick_strategy, deep_strategy):
        self.quick_strategy = quick_strategy
        self.deep_strategy = deep_strategy

    @property
    def name(self) -> str:
        return "conditional"

    def run(self, ctx: CompareContext) -> dict:
        # 先执行快速检查
        quick_result = self.quick_strategy.run(ctx)

        # 如果快速检查通过，直接返回
        fuzzy = quick_result.get('fuzzy_pure')
        if fuzzy and fuzzy.passed:
            return {
                **quick_result,
                'stage': 'quick',
            }

        # 否则执行深度分析
        deep_result = self.deep_strategy.run(ctx)
        return {
            **deep_result,
            'stage': 'deep',
        }

# 使用
from aidevtools.compare.strategy import QuickCheckStrategy, DeepAnalysisStrategy

conditional = ConditionalStrategy(
    quick_strategy=QuickCheckStrategy(),
    deep_strategy=DeepAnalysisStrategy(block_size=256),
)
engine = CompareEngine(strategy=conditional, config=config)
```

## 最佳实践

### 1. 选择合适的策略

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 日常开发 | StandardStrategy | 全面覆盖，平衡速度与深度 |
| CI/CD | QuickCheckStrategy | 快速验证，节省时间 |
| 调试问题 | DeepAnalysisStrategy | 分块定位，深度分析 |
| 批量比对 | MinimalStrategy | 最小开销，最快速度 |
| 渐进诊断 | ProgressiveStrategy | 自动分级，节省资源 |

### 2. 配置合理的阈值

```python
# 根据精度调整阈值
thresholds = {
    'bfp4': {'qsnr': 10.0, 'cosine': 0.95},
    'bfp8': {'qsnr': 15.0, 'cosine': 0.98},
    'fp16': {'qsnr': 30.0, 'cosine': 0.999},
    'fp32': {'qsnr': 60.0, 'cosine': 0.9999},
}

dtype = 'bfp8'
config = CompareConfig(
    fuzzy_min_qsnr=thresholds[dtype]['qsnr'],
    fuzzy_min_cosine=thresholds[dtype]['cosine'],
)
```

### 3. 复用策略实例

```python
# 创建一次，多次使用
standard_engine = CompareEngine.standard(config=config)

for op_name, (dut, golden) in data.items():
    result = standard_engine.run(dut=dut, golden=golden)
    # 处理结果...
```

### 4. 日志和调试

```python
from aidevtools.core.log import logger

# 启用详细日志
logger.setLevel('DEBUG')

engine = CompareEngine.standard(config=config)
result = engine.run(dut=dut, golden=golden)

# 输出策略信息
print(f"Strategy: {engine.strategy.name}")
print(f"Strategies included: {[s.name for s in engine.strategy.strategies]}")
```

## 参考资料

- [Compare 使用指南](./compare_guide.md)
- [架构设计文档](./design/compare_module_design.md)
- [Demo 示例](../demos/compare/)
- [源代码](../aidevtools/compare/strategy/)

## 常见问题

### Q: 如何查看策略包含哪些子策略？

```python
engine = CompareEngine.standard(config=config)
if hasattr(engine.strategy, 'strategies'):
    for strategy in engine.strategy.strategies:
        print(f"- {strategy.name}")
```

### Q: 如何跳过某些策略？

```python
# 方式1: 自定义组合
from aidevtools.compare.strategy import CompositeStrategy, ExactStrategy, FuzzyStrategy

# 只要 Exact 和 Fuzzy，跳过 Sanity
custom = CompositeStrategy([
    ExactStrategy(),
    FuzzyStrategy(use_golden_qnt=False),
], name="no_sanity")

# 方式2: 使用 QuickCheckStrategy (默认不包含 Sanity)
engine = CompareEngine.quick(config=config)
```

### Q: 如何实现算子级的策略选择？

```python
def get_strategy_for_op(op_name: str, config: CompareConfig):
    """根据算子类型选择策略"""
    if op_name.startswith('matmul'):
        # MatMul - 深度分析
        return CompareEngine.deep(config=config, block_size=256)
    elif op_name.startswith('layernorm'):
        # LayerNorm - 标准策略
        return CompareEngine.standard(config=config)
    else:
        # 其他 - 快速检查
        return CompareEngine.quick(config=config)

# 使用
for op_name, (dut, golden) in data.items():
    engine = get_strategy_for_op(op_name, config)
    result = engine.run(dut=dut, golden=golden)
```
