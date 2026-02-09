# 比对套件使用指南

> 版本: 2.0
> 最后更新: 2026-02-09
> 适用于: aidevtools >= 1.0.0

## 概述

比对套件用于验证自研芯片算子实现的正确性，支持从单算子到完整图的渐进式比对。

核心特性：
- **策略模式设计**：灵活组合不同比对策略
- **四状态判定**：PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT
- **Golden 自检**：自动检测 Golden 数据有效性
- **多种比对模式**：精确比对、模糊比对、Bit级分析、分块定位

## 四状态判定模型

| DUT vs Golden | Golden 自检 | 判定状态 | 含义 |
|---------------|-------------|----------|------|
| PASS | PASS | **PASS** | DUT 正确，Golden 有效 |
| PASS | FAIL | **GOLDEN_SUSPECT** | DUT 匹配，但 Golden 可疑 |
| FAIL | PASS | **DUT_ISSUE** | Golden 有效，DUT 有问题 |
| FAIL | FAIL | **BOTH_SUSPECT** | 都可疑，需人工排查 |

## 快速开始

### 基本用法

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
result = engine.run(
    dut=dut_output,           # DUT 输出
    golden=golden_fp32,       # 纯 fp32 Golden
    golden_qnt=golden_qnt,    # 量化感知 Golden (可选)
)

# 4. 查看结果
print(f"Status: {result.get('status')}")
print(f"Exact: {result.get('exact').passed if result.get('exact') else 'N/A'}")
print(f"Fuzzy QSNR: {result.get('fuzzy_pure').qsnr if result.get('fuzzy_pure') else 'N/A':.2f} dB")
```

### 便捷函数

```python
from aidevtools.compare import compare_full

# 一行代码完成比对
result = compare_full(
    dut=dut_output,
    golden=golden_fp32,
    golden_qnt=golden_qnt,
    config=config,
)
```

## 策略选择

### 预定义策略

aidevtools 提供多种预定义策略，适用于不同场景：

```python
# 1. 标准策略（推荐）- 完整的四层比对
engine = CompareEngine.standard(config=config)
# 包含: Exact + Fuzzy(Pure) + Fuzzy(Qnt) + Sanity

# 2. 快速检查 - 用于CI/CD
engine = CompareEngine.quick(config=config)
# 包含: Exact + Fuzzy(Pure)

# 3. 深度分析 - 包含分块定位
engine = CompareEngine.deep(config=config, block_size=1024)
# 包含: Exact + Fuzzy(Pure/Qnt) + Sanity + Blocked

# 4. 最小策略 - 性能优先
engine = CompareEngine.minimal(config=config)
# 仅包含: Fuzzy(Pure)

# 5. 渐进式分析 - 三级递进
engine = CompareEngine.progressive(config=config)
# L1: Exact → L2: Fuzzy+Sanity → L3: Blocked
```

### 结果格式

所有策略返回统一的字典格式：

```python
result = {
    'exact': ExactResult,           # 精确比对结果
    'fuzzy_pure': FuzzyResult,      # 纯FP32模糊比对
    'fuzzy_qnt': FuzzyResult,       # 量化感知模糊比对 (可选)
    'sanity': SanityResult,         # Golden自检结果
    'status': CompareStatus,        # 总体状态
}
```

## 配置参数

### CompareConfig

```python
from dataclasses import dataclass

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

## Golden 自检项

| 检查项 | 说明 | 失败原因 |
|--------|------|----------|
| non_zero | 数据非全零 | Golden 可能未正确生成 |
| no_nan_inf | 无 NaN/Inf | 数值溢出或异常 |
| range_valid | 数值范围合理 | 数据可能是常数 |
| qsnr_valid | 量化 QSNR 达标 | 量化误差过大 |

## 完整示例

### 单算子比对

```python
import numpy as np
from aidevtools import ops
from aidevtools.ops import _functional as F
from aidevtools.compare import CompareEngine, CompareConfig

# 1. 生成 Golden
ops.clear()
x = np.random.randn(2, 8, 64).astype(np.float32)
w = np.random.randn(64, 128).astype(np.float32)

y_golden = F.matmul(x, w)

# 2. 模拟 DUT 输出 (带噪声)
y_dut = y_golden + np.random.randn(*y_golden.shape).astype(np.float32) * 0.001

# 3. 比对
config = CompareConfig(fuzzy_min_qsnr=30.0, fuzzy_min_cosine=0.99)
engine = CompareEngine.standard(config=config)

result = engine.run(dut=y_dut, golden=y_golden)

# 4. 查看结果
status = result.get('status')
exact = result.get('exact')
fuzzy = result.get('fuzzy_pure')

print(f"Status: {status}")
if exact:
    print(f"Exact: {'PASS' if exact.passed else 'FAIL'}")
if fuzzy:
    print(f"QSNR: {fuzzy.qsnr:.2f} dB")
    print(f"Cosine: {fuzzy.cosine:.6f}")
```

### 多算子批量比对

```python
from aidevtools.compare.report import print_strategy_table

# 比对多个算子
results = []
for record in ops.get_records():
    result = engine.run(
        dut=dut_outputs[record.op_name],
        golden=record.golden,
    )
    results.append({
        'name': record.op_name,
        'result': result,
    })

# 打印结果表格
print_strategy_table(results)
```

输出示例：
```
================================================================================
Strategy Table - StandardStrategy
================================================================================
name          exact  fuzzy_pure  fuzzy_qnt  sanity      status
--------------------------------------------------------------------------------
matmul_0      PASS   PASS        PASS       PASS        PASS
layernorm_0   FAIL   PASS        PASS       PASS        PASS
softmax_0     FAIL   FAIL        FAIL       FAIL        BOTH_SUSPECT
================================================================================
Summary: 2 PASS, 0 GOLDEN_SUSPECT, 0 DUT_ISSUE, 1 BOTH_SUSPECT
================================================================================
```

## 高级用法

### 自定义策略组合

```python
from aidevtools.compare import CompareEngine
from aidevtools.compare.strategy import (
    ExactStrategy,
    FuzzyStrategy,
    SanityStrategy,
)

# 方式1: 使用策略列表
engine = CompareEngine(
    strategy=[
        ExactStrategy(),
        FuzzyStrategy(use_golden_qnt=False),
        SanityStrategy(),
    ],
    config=config,
)

# 方式2: 使用组合策略
from aidevtools.compare.strategy import CompositeStrategy

custom_strategy = CompositeStrategy(
    strategies=[
        ExactStrategy(),
        FuzzyStrategy(use_golden_qnt=True),
    ],
    name="my_custom_strategy",
)
engine = CompareEngine(strategy=custom_strategy, config=config)
```

### Bit级分析

```python
from aidevtools.compare.strategy import BitAnalysisStrategy, FP32

# 静态方法调用
result = BitAnalysisStrategy.compare(
    golden=golden_data,
    result=dut_data,
    fmt=FP32,
)

# 查看分析结果
print(f"Total elements: {result.summary.total_elements}")
print(f"Sign flip count: {result.summary.sign_flip_count}")
print(f"Exponent diff count: {result.summary.exponent_diff_count}")
print(f"Max exponent diff: {result.summary.max_exponent_diff}")

# 查看告警
for warning in result.warnings:
    print(f"[{warning.level.value}] {warning.message}")
```

## 报告生成

### 文本报告

```python
from aidevtools.compare.report import generate_text_report

results = [result1, result2, result3]
report = generate_text_report(results, output_path="report.txt")
```

### JSON 报告

```python
from aidevtools.compare.report import generate_json_report

report = generate_json_report(results, output_path="report.json")
```

## 失败处理

### GOLDEN_SUSPECT

Golden 自检失败，但 DUT 匹配 Golden。

**处理方法**：
1. 检查 Golden 生成逻辑
2. 检查量化参数配置
3. 查看 sanity 结果的 messages 获取详细信息
   ```python
   sanity = result.get('sanity')
   if sanity and not sanity.valid:
       for msg in sanity.messages:
           print(msg)
   ```

### DUT_ISSUE

Golden 有效，但 DUT 不匹配。

**处理方法**：
1. 查看 max_abs 定位误差范围
   ```python
   fuzzy = result.get('fuzzy_pure')
   print(f"Max abs error: {fuzzy.max_abs:.6e}")
   ```
2. 查看 qsnr 评估整体质量
   ```python
   print(f"QSNR: {fuzzy.qsnr:.2f} dB")
   ```
3. 检查 DUT 算子实现
4. 使用深度分析定位问题
   ```python
   engine = CompareEngine.deep(config=config, block_size=256)
   result = engine.run(dut=dut_output, golden=golden)

   # 查看分块结果
   blocked = result.get('blocked')
   if blocked:
       for block in blocked[:5]:  # 显示前5个误差最大的块
           print(f"Block {block.block_id}: max_abs={block.max_abs:.6e}")
   ```

### BOTH_SUSPECT

Golden 和 DUT 都可疑。

**处理方法**：
1. 优先修复 Golden 问题
2. 重新生成 Golden 后再测试 DUT
3. 检查数据范围是否合理
   ```python
   sanity = result.get('sanity')
   print(f"Non-zero check: {sanity.non_zero}")
   print(f"No NaN/Inf: {sanity.no_nan_inf}")
   print(f"Range valid: {sanity.range_valid}")
   ```

## 最佳实践

### 1. 选择合适的策略

- **日常开发**: 使用 `CompareEngine.standard()`
- **CI/CD**: 使用 `CompareEngine.quick()` 加快速度
- **调试问题**: 使用 `CompareEngine.deep()` 深度分析
- **性能敏感**: 使用 `CompareEngine.minimal()` 最小开销

### 2. 配置合理的阈值

```python
# BFP8 量化 - 降低精度要求
config_bfp8 = CompareConfig(
    fuzzy_min_qsnr=15.0,      # BFP8 QSNR 较低
    fuzzy_min_cosine=0.98,
    fuzzy_max_exceed_ratio=0.05,
)

# FP16 量化 - 中等精度
config_fp16 = CompareConfig(
    fuzzy_min_qsnr=30.0,
    fuzzy_min_cosine=0.999,
)

# FP32 计算 - 高精度
config_fp32 = CompareConfig(
    fuzzy_min_qsnr=60.0,
    fuzzy_min_cosine=0.9999,
)
```

### 3. 渐进式调试

```python
# 第一步：快速检查
quick_engine = CompareEngine.quick(config=config)
result = quick_engine.run(dut=dut, golden=golden)

if result.get('status') != CompareStatus.PASS:
    # 第二步：深度分析
    deep_engine = CompareEngine.deep(config=config, block_size=256)
    result = deep_engine.run(dut=dut, golden=golden)

    # 第三步：Bit级分析
    from aidevtools.compare.strategy import BitAnalysisStrategy, FP32
    bit_result = BitAnalysisStrategy.compare(golden, dut, fmt=FP32)
```

## 参考资料

- [策略模式使用指南](./compare_strategy_guide.md)
- [架构设计文档](./design/compare_module_design.md)
- [Demo 示例](../demos/compare/)

## API 迁移指南

从旧版 API 升级到 v2.0：

```python
# 旧版 API (已废弃)
result = engine.compare(
    dut_output=dut,
    golden_pure=golden,
    name="matmul_0",
)
print(f"Status: {result.status.value}")
print(f"QSNR: {result.fuzzy_pure.qsnr}")

# 新版 API (v2.0+)
result = engine.run(
    dut=dut,
    golden=golden,
)
print(f"Status: {result.get('status')}")
fuzzy = result.get('fuzzy_pure')
print(f"QSNR: {fuzzy.qsnr if fuzzy else 'N/A'}")
```

主要变化：
1. `engine.compare()` → `engine.run()`
2. `dut_output` → `dut`
3. `golden_pure` → `golden`
4. 返回值从对象改为字典，需使用 `.get()` 访问
5. 移除 `name` 参数（由调用方管理）
