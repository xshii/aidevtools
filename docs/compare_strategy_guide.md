# 比对策略模式指南

## 架构概览

```
CompareEngine ← 执行引擎
    ↓
CompareStrategy ← 策略接口 (ABC)
    ├─ ExactStrategy         ← 精确比对
    ├─ FuzzyStrategy         ← 模糊比对
    ├─ SanityStrategy        ← Golden 自检
    ├─ BitXorStrategy        ← Bit XOR 比对
    ├─ BitAnalysisStrategy   ← Bit 级分析
    ├─ BlockedStrategy       ← 分块分析
    └─ CompositeStrategy     ← 组合策略
        ├─ StandardStrategy      ← 标准比对 (推荐)
        ├─ QuickCheckStrategy    ← 快速检查
        ├─ DeepAnalysisStrategy  ← 深度分析
        └─ MinimalStrategy       ← 最小比对

TieredStrategy ← 分级策略 (条件执行)
    ├─ ProgressiveStrategy       ← 三级渐进式
    └─ QuickThenDeepStrategy     ← 两级快速+深度

ModelTieredAnalyzer ← 模型级分级分析器
```

## 策略接口

所有策略都继承 `CompareStrategy`，实现两个核心方法：

```python
from abc import ABC, abstractmethod

class CompareStrategy(ABC):
    @abstractmethod
    def run(self, ctx: CompareContext) -> Any:
        """执行比对，返回结果"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    def prepare(self, ctx: CompareContext) -> None:
        """预处理（可选）"""
        pass
```

### CompareContext

上下文对象，包含策略共享的数据：

```python
@dataclass
class CompareContext:
    golden: np.ndarray           # Golden 数据
    dut: np.ndarray              # DUT 待测数据
    config: CompareConfig        # 比对配置
    golden_qnt: np.ndarray = None  # 量化 Golden（可选）
    prepared: _PreparedPair = None  # 预处理缓存
    metadata: dict = None        # 额外元数据
```

## 基础策略

### ExactStrategy — 精确比对

逐元素比对，判断是否完全一致（或在允许误差范围内）。

```python
from aidevtools.compare.strategy import ExactStrategy

# 方式 1：静态方法（独立使用）
result = ExactStrategy.compare(golden, dut, max_abs=0.0, max_count=0)
print(result.passed)           # bool
print(result.mismatch_count)   # 不一致元素数
print(result.first_diff_offset)  # 第一个差异位置 (-1 = 无)
print(result.max_abs)          # 最大绝对误差

# 方式 2：字节级比对
is_same = ExactStrategy.compare_bytes(golden_bytes, dut_bytes)

# 方式 3：通过引擎
from aidevtools.compare import CompareEngine
engine = CompareEngine(ExactStrategy())
results = engine.run(dut=dut, golden=golden)
# results["exact"] → ExactResult
```

**返回类型**：`ExactResult`

### FuzzyStrategy — 模糊比对

计算 QSNR、余弦相似度等指标，支持容差判定。

```python
from aidevtools.compare.strategy import FuzzyStrategy

# 静态方法
result = FuzzyStrategy.compare(golden, dut, config)
print(result.passed)     # bool
print(result.qsnr)       # QSNR (dB)
print(result.cosine)     # 余弦相似度
print(result.max_abs)    # 最大绝对误差
print(result.exceed_count)  # 超阈值元素数

# 通过引擎（use_golden_qnt 控制使用哪个 Golden）
engine = CompareEngine(FuzzyStrategy(use_golden_qnt=False))
results = engine.run(dut=dut, golden=golden)
# results["fuzzy_pure"] → FuzzyResult

engine = CompareEngine(FuzzyStrategy(use_golden_qnt=True))
results = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)
# results["fuzzy_qnt"] → FuzzyResult
```

**返回类型**：`FuzzyResult`

### SanityStrategy — Golden 自检

验证 Golden 数据有效性（非零、无 NaN/Inf、范围合理、量化 QSNR）。

```python
from aidevtools.compare.strategy import SanityStrategy

# 静态方法
result = SanityStrategy.compare(golden_pure, golden_qnt, config)
print(result.valid)      # bool
print(result.messages)   # 失败信息列表
print(result.non_zero)   # 是否非零
print(result.no_nan_inf) # 是否无 NaN/Inf

# 通过引擎
engine = CompareEngine(SanityStrategy())
results = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)
# results["sanity"] → SanityResult
```

**返回类型**：`SanityResult`

### BlockedStrategy — 分块比对

将大张量分块比对，快速定位误差集中区域。

```python
from aidevtools.compare.strategy import BlockedStrategy

# 静态方法
blocks = BlockedStrategy.compare(golden, dut, block_size=1024)
for b in blocks:
    print(f"offset={b.offset}, qsnr={b.qsnr:.1f}, passed={b.passed}")

# 打印文本热力图
BlockedStrategy.print_heatmap(blocks, cols=40)
# 输出：
#   Block Heatmap (100 blocks, 3 failed)
#   ========================================
#        0 |....o..X.........o.................|
#    34816 |...............................#...|
#   ========================================
#   Legend: . >= 40dB, o >= 20dB, X >= 10dB, # < 10dB

# 找最差的 5 个 block
worst = BlockedStrategy.find_worst(blocks, top_n=5)

# 通过引擎
engine = CompareEngine(BlockedStrategy(block_size=1024))
results = engine.run(dut=dut, golden=golden)
# results["blocked_1024"] → List[BlockResult]
```

**返回类型**：`List[BlockResult]`

### BitXorStrategy — Bit XOR 比对

按 bit 进行 XOR 比对，快速发现位级差异。

```python
from aidevtools.compare.strategy import BitXorStrategy

# 通过引擎
engine = CompareEngine(BitXorStrategy())
results = engine.run(dut=dut, golden=golden)
# results["bit_xor"] → BitXorResult
```

**返回类型**：`BitXorResult`

### BitAnalysisStrategy — Bit 级分析

分析每个 bit 位的翻转情况，支持多种浮点格式。

```python
from aidevtools.compare.strategy import BitAnalysisStrategy
from aidevtools.compare import FP32, FP16, BFLOAT16, BFP16, BFP8, INT8

# 通过引擎
engine = CompareEngine(BitAnalysisStrategy())
results = engine.run(dut=dut, golden=golden)
# results["bit_analysis"] → BitAnalysisResult

# 指定格式
engine = CompareEngine(BitAnalysisStrategy(fmt=BFLOAT16))

# 可视化
from aidevtools.compare import (
    print_bit_analysis,
    print_bit_template,
    print_bit_heatmap,
    gen_bit_heatmap_svg,
    gen_perbit_bar_svg,
)

# 打印 bit 模板
print_bit_template(FP32)
# 输出: SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM

# 打印分析结果
print_bit_analysis(result)

# 生成 SVG 热力图
svg = gen_bit_heatmap_svg(result)
with open("heatmap.svg", "w") as f:
    f.write(svg)

# 生成 per-bit 条形图
svg = gen_perbit_bar_svg(result)
```

**支持的格式**：

| 格式 | 位宽 | 布局 | 说明 |
|------|------|------|------|
| `FP32` | 32 | 1S+8E+23M | IEEE 754 单精度 |
| `FP16` | 16 | 1S+5E+10M | IEEE 754 半精度 |
| `BFLOAT16` | 16 | 1S+8E+7M | Brain Float 16 |
| `BFP16` | 16+8 | 1S+0E+15M + 共享 8E | Block FP16 |
| `BFP8` | 8+8 | 1S+0E+7M + 共享 8E | Block FP8 |
| `BFP4` | 4+8 | 1S+0E+3M + 共享 8E | Block FP4 |
| `INT8` | 8 | 1S+0E+7M | 有符号整数 |
| `UINT8` | 8 | 0S+0E+8M | 无符号整数 |

**返回类型**：`BitAnalysisResult`

## 组合策略

`CompositeStrategy` 将多个策略组合，按顺序执行。

### 自定义组合

```python
from aidevtools.compare.strategy import (
    CompositeStrategy, ExactStrategy, FuzzyStrategy, SanityStrategy,
)

# 自定义组合
strategy = CompositeStrategy([
    ExactStrategy(),
    FuzzyStrategy(use_golden_qnt=False),
    SanityStrategy(),
], name="my_combo")

engine = CompareEngine(strategy=strategy)
results = engine.run(dut=dut, golden=golden)
# results = {"exact": ExactResult, "fuzzy_pure": FuzzyResult, "sanity": SanityResult}
```

### 预定义组合

#### StandardStrategy（推荐）

```python
from aidevtools.compare.strategy import StandardStrategy
engine = CompareEngine(StandardStrategy())
# 等价于: CompareEngine.standard()
# 包含: Exact + Fuzzy(pure) + Fuzzy(qnt) + Sanity
```

#### QuickCheckStrategy

```python
from aidevtools.compare.strategy import QuickCheckStrategy
engine = CompareEngine(QuickCheckStrategy())
# 等价于: CompareEngine.quick()
# 包含: Exact + Fuzzy(pure)
```

#### DeepAnalysisStrategy

```python
from aidevtools.compare.strategy import DeepAnalysisStrategy
engine = CompareEngine(DeepAnalysisStrategy(block_size=1024))
# 等价于: CompareEngine.deep(block_size=1024)
# 包含: Exact + Fuzzy(pure) + Fuzzy(qnt) + Sanity + Blocked
```

#### MinimalStrategy

```python
from aidevtools.compare.strategy import MinimalStrategy
engine = CompareEngine(MinimalStrategy())
# 等价于: CompareEngine.minimal()
# 包含: Fuzzy(pure)
```

## 分级策略

分级策略（`TieredStrategy`）根据前一级结果决定是否执行下一级，节省计算资源。

### 工作原理

```
Level 1 (策略组)
    ↓ condition(results) == True?
Level 2 (策略组)
    ↓ condition(results) == True?
Level 3 (策略组)
    ↓ 结束
```

每级包含一组策略和一个条件函数。条件函数返回 `True` 表示继续下一级。

### 自定义分级策略

```python
from aidevtools.compare.strategy import (
    TieredStrategy, StrategyLevel,
    ExactStrategy, FuzzyStrategy, BlockedStrategy,
)

def my_condition(results):
    """exact 不通过则继续"""
    exact = results.get("exact")
    return exact is None or not exact.passed

strategy = TieredStrategy([
    StrategyLevel("L1", [ExactStrategy()], my_condition),
    StrategyLevel("L2", [FuzzyStrategy(), BlockedStrategy()], lambda r: False),
])

engine = CompareEngine(strategy)
results = engine.run(dut=dut, golden=golden)

# 查看执行信息
print(results["_executed_levels"])  # ['L1'] 或 ['L1', 'L2']
print(results["_stopped_at"])       # 'L1' 或 'L2'
```

### 预定义分级策略

#### ProgressiveStrategy — 三级渐进式

```python
engine = CompareEngine.progressive(block_size=64)
results = engine.run(dut=dut, golden=golden)
```

三级策略：
- **L1_quick**: Exact + BitXor（快速检查）
  - 条件: exact 通过则停止
- **L2_medium**: Fuzzy + Sanity（中度诊断）
  - 条件: fuzzy 通过则停止
- **L3_deep**: BitAnalysis + Blocked（深度定位）

#### QuickThenDeepStrategy — 两级

```python
engine = CompareEngine.quick_then_deep(block_size=64)
results = engine.run(dut=dut, golden=golden)
```

两级策略：
- **L1_quick**: Exact + BitXor
  - 条件: exact 通过则停止
- **L2_deep**: Fuzzy + Sanity + Blocked

## 模型级分析

`ModelTieredAnalyzer` 支持全局协调的三级分析，适用于大模型（几十到几百个算子）。

### 工作流程

```
L1: 所有算子快速检查 (Exact + BitXor)
    ↓ 通过率 < l1_threshold?
L2: 失败算子中度分析 (Fuzzy + Sanity)
    ↓ 通过率 < l2_threshold?
L3: 仍失败的深度分析 (BitAnalysis + Blocked)
```

### 使用示例

```python
from aidevtools.compare import CompareEngine, CompareConfig
import numpy as np

# 准备算子数据
pairs = {
    "matmul_0": (golden_matmul, dut_matmul),
    "conv_0": (golden_conv, dut_conv),
    "relu_0": (golden_relu, dut_relu),
    # ... 几十到几百个算子
}

# 创建分析器
config = CompareConfig(fuzzy_min_qsnr=30.0)
analyzer = CompareEngine.model_progressive(config=config)

# 执行渐进分析
results = analyzer.progressive_analyze(
    pairs,
    l1_threshold=0.9,   # 90% L1 通过则停止
    l2_threshold=0.8,   # 80% L2 通过则停止
    block_size=64,
    verbose=True,
)

# 打印汇总
analyzer.print_summary(results)

# 查看某算子的分析级别
print(results["softmax_0"]["_levels"])  # ['L1', 'L2', 'L3']
```

输出示例：
```
============================================================
  模型级渐进式分析: 100 个算子
============================================================

[L1] 快速检查: Exact + Bitwise
  通过: 85/100 (85.0%)

[L2] 中度分析: Fuzzy + Sanity (15 个算子)
  新增通过: 8/15 (53.3%)

[L3] 深度分析: BitAnalysis + Blocked (7 个算子)
  完成深度分析

============================================================
  分级执行统计:
    L1: 85 ops (85.0%)
    L2: 8 ops (8.0%)
    L3: 7 ops (7.0%)

  最终状态:
    Exact 通过: 85/100
    Fuzzy 通过: 8/15 (L2算子)
    剩余失败: 7
============================================================
```

## 模型级 Bit 分析

对整个模型的所有算子进行 bit 级分析：

```python
from aidevtools.compare import compare_model_bitwise, print_model_bit_analysis
from aidevtools.compare import FP32

# 准备算子数据
op_pairs = {
    "matmul_0": (golden_matmul, dut_matmul),
    "conv_0": (golden_conv, dut_conv),
}

# 执行模型级 bit 分析
model_result = compare_model_bitwise(op_pairs, fmt=FP32)

# 打印模型级分析
print_model_bit_analysis(model_result)

# 查看全局结果
print(model_result.has_critical)    # 是否有严重问题
print(model_result.global_result)   # 全局 BitAnalysisResult
```

## 自定义策略开发

### 实现步骤

1. 继承 `CompareStrategy`
2. 实现 `run()` 方法
3. 实现 `name` 属性

```python
from aidevtools.compare.strategy import CompareStrategy, CompareContext
from dataclasses import dataclass

@dataclass
class MyResult:
    passed: bool
    score: float

class MyStrategy(CompareStrategy):
    """自定义比对策略"""

    def run(self, ctx: CompareContext) -> MyResult:
        # 使用 ctx.golden, ctx.dut, ctx.config
        diff = ctx.golden.astype(float) - ctx.dut.astype(float)
        score = float(1.0 - abs(diff).mean())
        return MyResult(passed=score > 0.99, score=score)

    @property
    def name(self) -> str:
        return "my_strategy"
```

### 集成到组合策略

```python
from aidevtools.compare.strategy import CompositeStrategy, ExactStrategy

strategy = CompositeStrategy([
    ExactStrategy(),
    MyStrategy(),
])

engine = CompareEngine(strategy=strategy)
results = engine.run(dut=dut, golden=golden)
# results["exact"] → ExactResult
# results["my_strategy"] → MyResult
```

### 集成到分级策略

```python
from aidevtools.compare.strategy import TieredStrategy, StrategyLevel

strategy = TieredStrategy([
    StrategyLevel("L1", [ExactStrategy()], lambda r: not r["exact"].passed),
    StrategyLevel("L2", [MyStrategy()], lambda r: False),
])
```

## 预定义条件函数

用于分级策略的条件判断：

| 函数 | 说明 |
|------|------|
| `always_continue` | 总是继续下一级 |
| `never_continue` | 从不继续（用于最后一级） |
| `stop_if_exact_passed` | exact 通过则停止 |
| `stop_if_fuzzy_passed` | fuzzy 通过则停止 |
