# 比对套件使用指南

## 概述

比对套件用于验证自研芯片算子实现的正确性，支持从单算子到完整图的渐进式比对。

核心特性：
- **策略模式**：灵活组合不同的比对策略（精确/模糊/自检/分块/Bit级）
- **四状态判定**：PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT
- **Golden 自检**：自动检测 Golden 数据有效性
- **渐进式分析**：L1 快速 → L2 中度 → L3 深度，节省计算资源

## 四状态判定模型

| DUT vs Golden | Golden 自检 | 判定状态 | 含义 |
|---------------|-------------|----------|------|
| PASS | PASS | **PASS** | DUT 正确，Golden 有效 |
| PASS | FAIL | **GOLDEN_SUSPECT** | DUT 匹配，但 Golden 可疑 |
| FAIL | PASS | **DUT_ISSUE** | Golden 有效，DUT 有问题 |
| FAIL | FAIL | **BOTH_SUSPECT** | 都可疑，需人工排查 |

## 快速开始

### 方式 1：使用引擎 (推荐)

```python
from aidevtools.compare import CompareEngine, CompareConfig

# 1. 创建配置
config = CompareConfig(
    fuzzy_min_qsnr=30.0,      # 最小 QSNR 阈值
    fuzzy_min_cosine=0.999,   # 最小余弦相似度
    sanity_min_qsnr=20.0,     # Golden 自检 QSNR 阈值
)

# 2. 创建引擎（使用标准策略）
engine = CompareEngine.standard(config=config)

# 3. 执行比对
results = engine.run(
    dut=dut_output,           # DUT 输出
    golden=golden_fp32,       # 纯 fp32 Golden
    golden_qnt=golden_qnt,    # 量化感知 Golden (可选)
)

# 4. 查看结果
print(f"Status: {results['status'].value}")
print(f"Exact passed: {results['exact'].passed}")
print(f"Fuzzy QSNR: {results['fuzzy_qnt'].qsnr:.1f} dB")
print(f"Golden valid: {results['sanity'].valid}")
```

### 方式 2：便捷函数

```python
from aidevtools.compare import compare_full, compare_quick

# 完整比对（标准策略）
results = compare_full(
    dut=dut_output,
    golden=golden_fp32,
    golden_qnt=golden_qnt,
)

# 快速比对（仅 Exact + Fuzzy）
results = compare_quick(dut=dut_output, golden=golden_fp32)
```

### 方式 3：自定义策略组合

```python
from aidevtools.compare import CompareEngine
from aidevtools.compare.strategy import (
    ExactStrategy, FuzzyStrategy, SanityStrategy,
    CompositeStrategy,
)

# 自定义策略组合
strategy = CompositeStrategy([
    ExactStrategy(),
    FuzzyStrategy(use_golden_qnt=False),
    SanityStrategy(),
])

engine = CompareEngine(strategy=strategy, config=config)
results = engine.run(dut=dut, golden=golden)
```

## 预定义引擎

| 工厂方法 | 策略 | 适用场景 |
|----------|------|----------|
| `CompareEngine.standard()` | Exact + Fuzzy(pure) + Fuzzy(qnt) + Sanity | 日常开发验证 |
| `CompareEngine.quick()` | Exact + Fuzzy(pure) | CI/CD 快速检查 |
| `CompareEngine.deep()` | Standard + Blocked | 深度调试、误差定位 |
| `CompareEngine.minimal()` | Fuzzy(pure) | 性能优先 |
| `CompareEngine.progressive()` | L1→L2→L3 渐进式 | 单算子渐进分析 |
| `CompareEngine.quick_then_deep()` | L1→L2 两级 | 快速筛选 + 深度分析 |
| `CompareEngine.model_progressive()` | 模型级三级分析 | 大模型调试 |

## 配置参数

### CompareConfig

```python
@dataclass
class CompareConfig:
    # 精确比对阈值
    exact_max_abs: float = 0.0     # 允许的最大绝对误差
    exact_max_count: int = 0       # 允许的最大不匹配数

    # 模糊比对阈值
    fuzzy_atol: float = 1e-5       # 绝对容差
    fuzzy_rtol: float = 1e-3       # 相对容差
    fuzzy_min_qsnr: float = 30.0   # 最小 QSNR (dB)
    fuzzy_min_cosine: float = 0.999 # 最小余弦相似度
    fuzzy_max_exceed_ratio: float = 0.0  # 最大超限比例

    # Golden 自检阈值
    sanity_min_qsnr: float = 20.0  # golden_qnt vs golden_pure
    sanity_max_nan_ratio: float = 0.0
    sanity_max_inf_ratio: float = 0.0
    sanity_min_nonzero_ratio: float = 0.01
```

## 精度指标

| 指标 | 公式 | 参考值 | 说明 |
|------|------|--------|------|
| max_abs | max(\|g-r\|) | < 1e-5 | 最大绝对误差 |
| mean_abs | mean(\|g-r\|) | < 1e-6 | 平均绝对误差 |
| qsnr | 10*log10(signal/noise) | > 30dB | 量化信噪比 |
| cosine | dot(g,r)/(norm*norm) | > 0.999 | 余弦相似度 |

## 结果格式

### engine.run() 返回值

`engine.run()` 返回 `Dict[str, Any]`，key 是策略名称：

```python
results = engine.run(dut=dut, golden=golden)

# 精确比对结果 (ExactResult)
results["exact"].passed          # bool
results["exact"].mismatch_count  # int
results["exact"].first_diff_offset  # int (-1 = 无差异)
results["exact"].max_abs         # float

# 模糊比对结果 (FuzzyResult)
results["fuzzy_pure"].passed     # bool
results["fuzzy_pure"].qsnr       # float (dB)
results["fuzzy_pure"].cosine     # float
results["fuzzy_pure"].max_abs    # float

# 量化感知模糊比对 (FuzzyResult)
results["fuzzy_qnt"].passed      # bool

# Golden 自检 (SanityResult)
results["sanity"].valid          # bool
results["sanity"].messages       # List[str]

# 总体状态 (CompareStatus)
results["status"]                # CompareStatus enum
```

## Golden 自检

| 检查项 | 说明 | 失败原因 |
|--------|------|----------|
| non_zero | 数据非全零 | Golden 可能未正确生成 |
| no_nan_inf | 无 NaN/Inf | 数值溢出或异常 |
| range_valid | 数值范围合理 | 数据可能是常数 |
| qsnr_valid | 量化 QSNR 达标 | 量化误差过大 |

## 报告生成

### 策略结果表格

```python
from aidevtools.compare import print_strategy_table

# results_list: List[Dict]，每个元素是 engine.run() 的返回值
print_strategy_table(results_list, names=["matmul_0", "layernorm_0"])
```

输出示例：
```
==============================================================================================================
name            exact  f_pure   f_qnt   sanity     max_abs     qsnr   cosine        status
--------------------------------------------------------------------------------------------------------------
matmul_0           Y       Y       Y       Y     0.00e+00      inf 1.000000          PASS
layernorm_0        N       Y       Y       Y     2.52e-01    17.54 0.991358          PASS
softmax_0          N       Y       N       N     2.63e-02    14.54 0.982997   BOTH_SUSPECT
==============================================================================================================
Summary: 2 PASS, 0 GOLDEN_SUSPECT, 0 DUT_ISSUE, 1 BOTH_SUSPECT (total: 3)
```

### JSON 报告

```python
from aidevtools.compare import generate_strategy_json

json_data = generate_strategy_json(results, name="matmul_0")
```

## 渐进式分析

### 单算子渐进式

```python
# L1→L2→L3，精确通过则停止
engine = CompareEngine.progressive()
results = engine.run(dut=dut, golden=golden)

# 查看执行了哪些级别
print(results["_executed_levels"])  # ['L1_quick']  (如果 exact 通过)
print(results["_stopped_at"])       # 'L1_quick'
```

### 模型级渐进式

```python
# 全局协调：根据整体通过率决定是否深入
analyzer = CompareEngine.model_progressive(config=config)
results = analyzer.progressive_analyze(
    pairs={"matmul_0": (golden, dut), "conv_0": (golden2, dut2)},
    l1_threshold=0.9,   # 90% L1 通过则停止
    l2_threshold=0.8,   # 80% L2 通过则停止
)
analyzer.print_summary(results)
```

## 工作流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. 生成    │ ──> │  2. 执行    │ ──> │  3. 比对    │ ──> │  4. 报告    │
│  Golden     │     │  DUT        │     │  验证       │     │  分析       │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                   │
      v                   v                   v                   v
 golden_pure         dut_output         Dict[str,Any]       表格/JSON
 golden_qnt
```

### 完整示例

```python
import numpy as np
from aidevtools.compare import CompareEngine, CompareConfig
from aidevtools.compare import print_strategy_table

# 1. 准备数据
golden = np.random.randn(2, 8, 64).astype(np.float32)
dut = golden + np.random.randn(2, 8, 64).astype(np.float32) * 0.001

# 2. 创建引擎
config = CompareConfig(fuzzy_min_qsnr=30.0, fuzzy_min_cosine=0.99)
engine = CompareEngine.standard(config=config)

# 3. 执行比对
results = engine.run(dut=dut, golden=golden)

# 4. 输出结果
print(f"Status: {results['status'].value}")
print_strategy_table([results], names=["matmul_0"])
```

## 失败处理

### GOLDEN_SUSPECT

Golden 自检失败，但 DUT 匹配 Golden。

**处理方法**：
1. 检查 Golden 生成逻辑
2. 检查量化参数配置
3. 查看 `results["sanity"].messages` 获取详细信息

### DUT_ISSUE

Golden 有效，但 DUT 不匹配。

**处理方法**：
1. 查看 `results["fuzzy_qnt"].max_abs` 定位误差范围
2. 查看 `results["fuzzy_qnt"].qsnr` 评估整体质量
3. 使用 `CompareEngine.deep()` 获取分块分析，定位误差区域
4. 检查 DUT 算子实现

### BOTH_SUSPECT

Golden 和 DUT 都可疑。

**处理方法**：
1. 优先修复 Golden 问题
2. 重新生成 Golden 后再测试 DUT

## 旧 API 兼容

以下旧 API 仍可用（已废弃，建议迁移到新 API）：

```python
# 旧 API（已废弃）
engine = CompareEngine(config=config)
result = engine.compare(dut, golden, golden_qnt, name="op_0")  # 返回 CompareResult

# 新 API（推荐）
engine = CompareEngine.standard(config=config)
results = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)  # 返回 Dict
```

| 旧 API | 新 API |
|--------|--------|
| `engine.compare()` | `engine.run()` |
| `engine.compare_exact_only()` | `CompareEngine.quick()` + `engine.run()` |
| `engine.compare_fuzzy_only()` | `CompareEngine.minimal()` + `engine.run()` |
| `CompareEngine(config)` | `CompareEngine.standard(config=config)` |
| `compare_exact(g, r)` | `ExactStrategy.compare(g, r)` |
| `compare_fuzzy(g, r, cfg)` | `FuzzyStrategy.compare(g, r, cfg)` |
| `check_golden_sanity(g, q, cfg)` | `SanityStrategy.compare(g, q, cfg)` |

## 命令行工具

```bash
# 生成 CSV
aidev trace run model.py -o workspace/

# 比数
aidev compare run model_compare.csv

# 归档
aidev compare archive model_compare.csv
```
