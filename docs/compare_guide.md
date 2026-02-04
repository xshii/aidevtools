# 比对套件使用指南

## 概述

比对套件用于验证自研芯片算子实现的正确性，支持从单算子到完整图的渐进式比对。

核心特性：
- **四状态判定**：PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT
- **Golden 自检**：自动检测 Golden 数据有效性
- **多种比对模式**：精确比对、模糊比对

## 四状态判定模型

| DUT vs Golden | Golden 自检 | 判定状态 | 含义 |
|---------------|-------------|----------|------|
| PASS | PASS | **PASS** | DUT 正确，Golden 有效 |
| PASS | FAIL | **GOLDEN_SUSPECT** | DUT 匹配，但 Golden 可疑 |
| FAIL | PASS | **DUT_ISSUE** | Golden 有效，DUT 有问题 |
| FAIL | FAIL | **BOTH_SUSPECT** | 都可疑，需人工排查 |

## 快速开始

### 使用 Compare API

```python
from aidevtools.compare import (
    CompareEngine,
    CompareConfig,
    CompareStatus,
)

# 1. 创建配置
config = CompareConfig(
    fuzzy_min_qsnr=30.0,      # 最小 QSNR 阈值
    fuzzy_min_cosine=0.999,   # 最小余弦相似度
    sanity_min_qsnr=20.0,     # Golden 自检 QSNR 阈值
)

# 2. 创建引擎
engine = CompareEngine(config)

# 3. 执行比对
result = engine.compare(
    dut_output=dut,           # DUT 输出
    golden_pure=golden_fp32,  # 纯 fp32 Golden
    golden_qnt=golden_qnt,    # 量化感知 Golden (可选)
    name="matmul_0",
)

# 4. 查看结果
print(f"Status: {result.status.value}")
print(f"DUT Passed: {result.dut_passed}")
print(f"Golden Valid: {result.golden_valid}")
```

### 便捷函数

```python
from aidevtools.compare import compare_full

# 一行代码完成比对
result = compare_full(
    dut_output=dut,
    golden_pure=golden_fp32,
    golden_qnt=golden_qnt,
    name="conv_0",
)
```

### 仅精确比对

```python
engine = CompareEngine()
result = engine.compare_exact_only(dut, golden, name="test")
```

### 仅模糊比对

```python
engine = CompareEngine(config)
result = engine.compare_fuzzy_only(dut, golden, name="test")
```

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

## Golden 自检项

| 检查项 | 说明 | 失败原因 |
|--------|------|----------|
| non_zero | 数据非全零 | Golden 可能未正确生成 |
| no_nan_inf | 无 NaN/Inf | 数值溢出或异常 |
| range_valid | 数值范围合理 | 数据可能是常数 |
| qsnr_valid | 量化 QSNR 达标 | 量化误差过大 |

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

### 表格输出

```python
from aidevtools.compare.report import print_compare_table

print_compare_table(results)
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

## 工作流程

### 完整流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. 生成    │ ──> │  2. 执行    │ ──> │  3. 比对    │ ──> │  4. 报告    │
│  Golden     │     │  DUT        │     │  验证       │     │  分析       │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                   │
      v                   v                   v                   v
 golden_pure         dut_output         CompareResult         report.txt
 golden_qnt                                                   report.json
```

### 示例代码

```python
import numpy as np
from aidevtools import ops
from aidevtools.ops import _functional as F
from aidevtools.compare import CompareEngine, CompareConfig
from aidevtools.compare.report import print_compare_table

# 1. 生成 Golden
ops.clear()
x = np.random.randn(2, 8, 64).astype(np.float32)
w = np.random.randn(64, 128).astype(np.float32)

y_golden = F.matmul(x, w)

# 2. 模拟 DUT 输出 (带噪声)
y_dut = y_golden + np.random.randn(*y_golden.shape).astype(np.float32) * 0.001

# 3. 比对
config = CompareConfig(fuzzy_min_qsnr=30.0, fuzzy_min_cosine=0.99)
engine = CompareEngine(config)

results = []
for r in ops.get_records():
    result = engine.compare(
        dut_output=y_dut,
        golden_pure=r.golden,
        name=r.op_name,
    )
    results.append(result)

# 4. 输出报告
print_compare_table(results)
```

## 失败处理

### GOLDEN_SUSPECT

Golden 自检失败，但 DUT 匹配 Golden。

**处理方法**：
1. 检查 Golden 生成逻辑
2. 检查量化参数配置
3. 查看 sanity.messages 获取详细信息

### DUT_ISSUE

Golden 有效，但 DUT 不匹配。

**处理方法**：
1. 查看 max_abs 定位误差范围
2. 查看 qsnr 评估整体质量
3. 检查 DUT 算子实现

### BOTH_SUSPECT

Golden 和 DUT 都可疑。

**处理方法**：
1. 优先修复 Golden 问题
2. 重新生成 Golden 后再测试 DUT

## 命令行工具 (旧版)

```bash
# 生成 CSV
aidev trace run model.py -o workspace/

# 比数
aidev compare run model_compare.csv

# 归档
aidev compare archive model_compare.csv
```
