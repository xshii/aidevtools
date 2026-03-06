# 比数模块设计说明书

> 版本: 3.0
> 日期: 2026-03
> 模块: `aidevtools.compare`

---

## 1. 概述

### 1.1 目的

`aidevtools.compare` 提供 DUT (Device Under Test) 输出与 Golden 参考值的多层次精度比对，覆盖精确比对、模糊比对、Golden 自检、分块定位、Bit 级分析五大能力。

### 1.2 设计原则

1. **策略模式**: 每种比对能力封装为独立策略，通过引擎组合执行
2. **渐进式分析**: L1 快速筛选 → L2 中度诊断 → L3 深度定位，按需执行
3. **结果跟策略走**: 每个 Result 类型定义在对应的 Strategy 文件中
4. **性能优先**: 单次遍历合并计算、early exit 短路、`_PreparedPair` 缓存

---

## 2. 模块架构

### 2.1 目录结构

```
aidevtools/compare/
├── __init__.py              # 公共 API 统一导出
├── types.py                 # CompareConfig + _PreparedPair
├── metrics.py               # 指标计算 (QSNR, cosine, AllMetrics)
├── engine.py                # CompareEngine 执行引擎
├── model.py                 # ModelTieredAnalyzer 模型级分析
├── strategy/                # 策略
│   ├── __init__.py
│   ├── base.py              # CompareContext + CompareStrategy 接口
│   ├── exact.py             # ExactStrategy + ExactResult
│   ├── fuzzy.py             # FuzzyStrategy + FuzzyResult
│   ├── sanity.py            # SanityStrategy + SanityResult
│   ├── blocked.py           # BlockedStrategy + BlockResult
│   ├── bit_analysis.py      # BitAnalysisStrategy + BitAnalysisResult
│   ├── composite.py         # CompositeStrategy 自定义组合
│   └── tiered.py            # TieredStrategy + ProgressiveStrategy
└── report/                  # 报告
    ├── __init__.py
    ├── text_report.py       # 文字表格 + joint report
    ├── visualizer.py        # pyecharts 基础底座
    └── model_visualizer.py  # 模型级可视化
```

### 2.2 依赖关系

```
report/text_report ──→ strategy/* (消费各 Result 类型)
report/visualizer  ←── strategy/* (策略调用 Visualizer 生成图表)

strategy/*         ──→ types (CompareConfig, _PreparedPair)
strategy/*         ──→ metrics (指标计算)

engine             ──→ types + strategy
model              ──→ engine + strategy
```

关键点: **无循环依赖**。`types.py` 只含基础设施 (`CompareConfig`, `_PreparedPair`)，每个 Result 类型跟着自己的 Strategy 走。

---

## 3. 核心数据结构

### 3.1 CompareConfig (types.py)

```python
@dataclass
class CompareConfig:
    # 精确比对
    exact_max_abs: float = 0.0        # 允许的最大绝对误差 (0=bit级精确)
    exact_max_count: int = 0          # 允许超阈值的元素个数

    # 模糊比对
    fuzzy_atol: float = 1e-5          # 绝对容差
    fuzzy_rtol: float = 1e-3          # 相对容差
    fuzzy_min_qsnr: float = 30.0      # 最低 QSNR (dB)
    fuzzy_min_cosine: float = 0.999   # 最低余弦相似度
    fuzzy_max_exceed_ratio: float = 0.0

    # Golden 自检
    sanity_min_qsnr: float = 20.0
    sanity_max_nan_ratio: float = 0.0
    sanity_max_inf_ratio: float = 0.0
    sanity_min_nonzero_ratio: float = 0.01
```

### 3.2 _PreparedPair (types.py)

跨策略共享的预处理缓存，消除重复 `astype(float64).flatten()`:

```python
@dataclass
class _PreparedPair:
    g: np.ndarray        # float64 flattened golden
    r: np.ndarray        # float64 flattened result
    diff: np.ndarray     # g - r
    abs_err: np.ndarray  # |diff|
    g_abs: np.ndarray    # |g|
    total: int
```

### 3.3 Result 类型 (各 strategy 文件)

| Result | 定义位置 | 关键字段 |
|--------|----------|----------|
| `ExactResult` | `strategy/exact.py` | `passed`, `mismatch_count`, `max_abs`, `diff_bits`, `total_bits`, `diff_bit_ratio` |
| `FuzzyResult` | `strategy/fuzzy.py` | `passed`, `qsnr`, `cosine`, `max_abs`, `mean_abs`, `max_rel`, `exceed_count` |
| `SanityResult` | `strategy/sanity.py` | `valid`, `checks`, `messages`, `non_zero`, `no_nan_inf`, `range_valid`, `qsnr_valid` |
| `BlockResult` | `strategy/blocked.py` | `offset`, `size`, `qsnr`, `cosine`, `max_abs`, `passed` |
| `BitAnalysisResult` | `strategy/bit_analysis.py` | `fmt`, `summary`, `warnings`, `has_critical` |

---

## 4. 策略详细设计

### 4.1 ExactStrategy

精确比对 + bit 级 popcount 统计，单次 O(n) 扫描。

- `max_abs=0`: view 为 `uint8` 逐字节比较
- `max_abs>0`: 复用 `_PreparedPair.abs_err` 计数超阈值元素
- 同时计算 XOR popcount (`diff_bits` / `total_bits`)
- 有 `raw_golden`/`raw_dut` 时优先比源格式字节

### 4.2 FuzzyStrategy

统计指标比对，内部委托 `calc_all_metrics_early_exit`:

判定条件 (三条全满足才 PASS):
1. 超阈值比例 <= `max_exceed_ratio`
2. QSNR >= `min_qsnr`
3. cosine >= `min_cosine`

支持 `use_golden_qnt` 参数切换纯 fp32 / 量化感知比对。

### 4.3 SanityStrategy

Golden 数据有效性自检:

| 检查项 | 说明 |
|--------|------|
| `non_zero` | 非零元素比例 >= 阈值 |
| `no_nan_inf` | NaN/Inf 比例 <= 阈值 |
| `range_valid` | 数据非常数 (采样前 1000 元素) |
| `qsnr_valid` | golden_qnt vs golden_pure QSNR >= 阈值 |

### 4.4 BlockedStrategy

将大张量按 `block_size` 分块，每块独立计算 QSNR/cosine/max_abs，定位误差集中区域。支持文本热力图和 pyecharts 可视化。

### 4.5 BitAnalysisStrategy

逐 bit 语义分析 (sign/exponent/mantissa)，生成分级告警:
- **CRITICAL**: 符号位翻转 / 指数偏移 >= 2
- **WARNING**: 指数偏移 = 1
- **INFO**: 仅尾数差异

支持 `FloatFormat` (FP32/FP16/BFLOAT16) 和自定义 `BitLayout`。

---

## 5. 渐进式分级架构

### 5.1 ProgressiveStrategy

三级分析，按需执行:

```
L1_quick:   ExactStrategy
              ↓ exact.passed? → 停止 (deep=False 时)
L2_medium:  FuzzyStrategy(pure) + FuzzyStrategy(qnt) + SanityStrategy
              ↓ fuzzy.passed? → 停止 (deep=False 时)
L3_deep:    BitAnalysisStrategy + BlockedStrategy
```

`deep=True` 时三级全部执行，不做早停。

### 5.2 结果格式

```python
{
    "exact": ExactResult,
    "fuzzy_pure": FuzzyResult,
    "fuzzy_qnt": FuzzyResult,
    "sanity": SanityResult,
    "bit_analysis_fp32": BitAnalysisResult,   # L3
    "blocked_64": [BlockResult, ...],          # L3
    "_executed_levels": ["L1_quick", "L2_medium"],
    "_stopped_at": "L2_medium",
}
```

### 5.3 ModelTieredAnalyzer

模型级全局协调分析 (几十到几百个算子):

```
L1: 所有算子快速检查 (Exact)
    ↓ 通过率 >= 90%? → 停止
L2: 失败算子中度分析 (Fuzzy + Sanity)
    ↓ 通过率 >= 80%? → 停止
L3: 仍失败的深度分析 (BitAnalysis + Blocked)
```

---

## 6. 报告系统

### 6.1 文字报告 (text_report.py)

| 函数 | 说明 |
|------|------|
| `print_strategy_table(results_list, names)` | 多算子汇总表格 |
| `print_joint_report(results, name)` | 单算子联合报告 (表格 + bit 统计 + 分析详情 + 热力图) |
| `format_strategy_results(results, name)` | 格式化单行 |
| `generate_strategy_json(results, name)` | JSON 格式输出 |

`print_joint_report` 自动遍历 results 字典，对每种策略结果调用对应的打印方法。

### 6.2 可视化 (visualizer.py)

pyecharts 封装底座，提供 `create_pie`, `create_bar`, `create_heatmap`, `create_radar`, `create_sankey`, `create_line` 等工厂方法。

`visualize_joint_report(results, name)` 将所有策略图表合并到一个 Page。

---

## 7. CLI 命令接口

```bash
compare diff --golden=a.txt --result=b.txt [--qtype=bfp8] [--shape=2,16,64] [--engine=standard|deep]
compare convert --golden=a.bin --target_dtype=float16
compare qtypes
```

`--engine=standard`: 默认，ProgressiveStrategy(deep=False)，早停
`--engine=deep`: ProgressiveStrategy(deep=True)，三级全执行

文件名自动解读: `{op}_{qtype}_{NxMxK}.txt`

---

## 8. 指标计算 (metrics.py)

### 8.1 核心函数

| 函数 | 说明 |
|------|------|
| `calc_all_metrics(golden, result, atol, rtol)` | 单次遍历计算全部指标 |
| `calc_all_metrics_early_exit(...)` | 带 early exit: cosine → exceed → QSNR |
| `calc_qsnr(golden, result)` | QSNR (dB) |
| `calc_cosine(a, b)` | 余弦相似度 |
| `check_nan_inf(data)` | NaN/Inf 计数 |
| `check_nonzero(data)` | 非零元素统计 |

### 8.2 Early Exit 优化

```
1. flatten + float64 (一次)
2. cosine (最便宜，复用 g_sq_sum)
   └─ cosine < min_cosine → early exit
3. exceed_count (中等开销)
   └─ exceed_ratio > max → early exit
4. QSNR + rel_error (最贵)
5. 返回 AllMetrics
```

---

## 9. 测试

| 文件 | 说明 |
|------|------|
| `tests/ut/compare/test_compare.py` | 核心比对逻辑 |
| `tests/ut/compare/test_bitwise.py` | Bit 级分析 |
| `tests/ut/compare/test_optimizations.py` | 指标优化验证 |
| `tests/ut/compare/test_visualization.py` | 可视化 |
| `tests/st/test_compare_diff_cmd.py` | CLI 集成测试 |
| `tests/ut/test_qa_weights.py` | 量化权重比对 |

当前: 698 ut + 29 st 全过。
