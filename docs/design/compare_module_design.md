# 比数功能模块软件设计文档

> 版本: 2.0
> 日期: 2026-02
> 模块: `aidevtools.compare`
> 状态: Released

---

## 目录

1. [概述](#1-概述)
2. [模块架构](#2-模块架构)
3. [核心数据结构](#3-核心数据结构)
4. [子模块详细设计](#4-子模块详细设计)
5. [比对引擎流程](#5-比对引擎流程)
6. [外部依赖与接口](#6-外部依赖与接口)
7. [CLI 命令接口](#7-cli-命令接口)
8. [使用示例](#8-使用示例)
9. [测试策略](#9-测试策略)
10. [附录](#10-附录)

---

## 1. 概述

### 1.1 目的

本文档描述 `aidevtools.compare` 比数功能模块的软件设计。该模块负责对 DUT (Device Under Test) 输出与 Golden 参考值进行多层次精度比对，涵盖精确比对、模糊比对、Golden 自检、分块定位、Bit 级分析五大能力，以及报告生成。

### 1.2 功能范围

| 能力 | 说明 |
|------|------|
| 精确比对 | Bit 级或允许小误差的逐元素比较 |
| 模糊比对 | 基于 QSNR、余弦相似度、超阈值比例的统计比对 |
| Golden 自检 | 验证 Golden 数据有效性 (非零、无 NaN/Inf、QSNR) |
| 分块定位 | 将大张量分块比对，热力图定位误差集中区域 |
| Bit 级分析 | 逐 bit 对比 sign/exponent/mantissa，生成告警与可视化 |
| 状态判定 | 综合 DUT 比对 + Golden 自检，四态判定 |
| 报告生成 | 文本表格、文本报告、JSON 报告 |

### 1.3 术语定义

| 术语 | 定义 |
|------|------|
| DUT | Device Under Test，被测硬件/仿真器输出 |
| Golden | 软件参考实现的正确输出 |
| golden_pure | 纯 fp32 精度计算的 Golden |
| golden_qnt | 量化感知 (QA-aware) 计算的 Golden |
| QSNR | Quantization Signal-to-Noise Ratio，量化信噪比 (dB) |
| BFP | Block Floating Point，块浮点格式 (共享指数) |
| BitLayout | 通用 bit 分布配置，通过字母模板描述任意数据类型 |

### 1.4 设计原则

1. **分层解耦**: 指标计算 → 比对判定 → 引擎协调 → 报告输出，各层独立可测
2. **性能优先**: 单次遍历合并计算、early exit 短路、向量化 per-bit 统计
3. **格式透明**: 通过 `FloatFormat` / `BitLayout` 抽象，支持任意浮点与定点格式
4. **四态判定**: DUT 比对与 Golden 自检交叉，避免 Golden 异常导致误判

---

## 2. 模块架构

### 2.1 文件结构

```
aidevtools/compare/
├── __init__.py          # 公共 API 统一导出
├── types.py             # 核心数据结构 (CompareConfig, CompareResult, ...)
├── metrics.py           # 指标计算 (QSNR, cosine, abs/rel error, ...)
├── exact.py             # 精确比对
├── fuzzy.py             # 模糊比对
├── sanity.py            # Golden 自检
├── engine.py            # 比对引擎 (协调精确/模糊/自检)
├── blocked.py           # 分块比对 + 热力图
├── bitwise.py           # Bit 级分析 + 告警 + SVG 可视化
└── report.py            # 报告生成 (文本/JSON)
```

### 2.2 模块依赖关系

```
                    ┌─────────────┐
                    │  __init__   │  (统一导出)
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     ┌────▼────┐     ┌────▼────┐     ┌─────▼─────┐
     │ engine  │     │ blocked │     │  bitwise   │
     └────┬────┘     └────┬────┘     └─────┬─────┘
          │               │               │
    ┌─────┼─────┐         │               │
    │     │     │         │               │
┌───▼─┐ ┌▼───┐ ┌▼─────┐  │     ┌─────────▼─────────┐
│exact│ │fuzzy│ │sanity│  │     │formats.quantize    │
└──┬──┘ └──┬─┘ └──┬───┘  │     │(simulate_quantize) │
   │       │      │      │     └────────────────────┘
   └───────┴──────┴──────┘
                  │
           ┌──────▼──────┐      ┌──────────┐
           │   metrics   │      │  report   │
           └─────────────┘      └──────────┘
                  │
           ┌──────▼──────┐
           │    types     │
           └─────────────┘
```

### 2.3 外部模块交互

```
┌─────────────────────────────────────────────────────┐
│                   用户层                             │
│  demos/  │  CLI (commands/compare.py)  │  用户脚本   │
└──────────┴────────────┬────────────────┴────────────┘
                        │
              ┌─────────▼──────────┐
              │  aidevtools.compare │  ← 本模块
              └─────────┬──────────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
┌───────▼───────┐ ┌─────▼──────┐ ┌──────▼───────┐
│formats.base   │ │formats.    │ │  datagen     │
│(load/load_dir)│ │quantize    │ │(Model DSL)   │
└───────────────┘ └────────────┘ └──────────────┘
```

---

## 3. 核心数据结构

### 3.1 types.py — 类型定义

#### CompareStatus (Enum)

四态判定结果:

| 状态 | DUT vs Golden | Golden 自检 | 含义 |
|------|:---:|:---:|------|
| `PASS` | PASS | PASS | DUT 匹配 Golden，且 Golden 有效 |
| `GOLDEN_SUSPECT` | PASS | FAIL | DUT 匹配，但 Golden 自检失败 |
| `DUT_ISSUE` | FAIL | PASS | Golden 有效，但 DUT 不匹配 |
| `BOTH_SUSPECT` | FAIL | FAIL | 都可疑，需人工排查 |

#### ExactResult

```python
@dataclass
class ExactResult:
    passed: bool              # 是否通过
    mismatch_count: int       # 不匹配元素数
    first_diff_offset: int    # 第一个差异位置 (-1=无差异)
    max_abs: float            # 最大绝对误差
    total_elements: int       # 总元素数
```

#### FuzzyResult

```python
@dataclass
class FuzzyResult:
    passed: bool              # 是否通过
    max_abs: float            # 最大绝对误差
    mean_abs: float           # 平均绝对误差
    max_rel: float            # 最大相对误差
    qsnr: float               # 量化信噪比 (dB)
    cosine: float             # 余弦相似度
    total_elements: int       # 总元素数
    exceed_count: int         # 超阈值元素数
```

#### SanityResult

```python
@dataclass
class SanityResult:
    valid: bool                           # Golden 是否有效
    checks: Dict[str, bool]              # 各检查项结果
    messages: List[str]                  # 告警消息
    non_zero: bool = True                # 非全零
    no_nan_inf: bool = True              # 无 NaN/Inf
    range_valid: bool = True             # 数值范围合理
    qsnr_valid: bool = True              # golden_qnt vs golden_pure QSNR
```

#### CompareResult

```python
@dataclass
class CompareResult:
    name: str                             # 算子/比对名称
    op_id: int                            # 算子 ID
    exact: Optional[ExactResult]          # 精确比对结果
    fuzzy_pure: Optional[FuzzyResult]     # 模糊比对 (纯 fp32)
    fuzzy_qnt: Optional[FuzzyResult]      # 模糊比对 (量化感知)
    sanity: Optional[SanityResult]        # Golden 自检
    status: CompareStatus                 # 最终状态
```

属性方法:
- `dut_passed`: DUT 是否通过 (精确或模糊任一通过)
- `golden_valid`: Golden 是否有效
- `determine_status()`: 根据比对结果判定四态

#### CompareConfig

```python
@dataclass
class CompareConfig:
    # 精确比对阈值
    exact_max_abs: float = 0.0           # 允许的最大绝对误差 (0=bit级精确)
    exact_max_count: int = 0             # 允许超阈值的元素个数

    # 模糊比对阈值
    fuzzy_atol: float = 1e-5             # 绝对容差
    fuzzy_rtol: float = 1e-3             # 相对容差
    fuzzy_min_qsnr: float = 30.0         # 最低 QSNR (dB)
    fuzzy_min_cosine: float = 0.999      # 最低余弦相似度
    fuzzy_max_exceed_ratio: float = 0.0  # 最大超阈值比例

    # Golden 自检阈值
    sanity_min_qsnr: float = 20.0        # golden_qnt vs golden_pure QSNR
    sanity_max_nan_ratio: float = 0.0    # 最大 NaN 比例
    sanity_max_inf_ratio: float = 0.0    # 最大 Inf 比例
    sanity_min_nonzero_ratio: float = 0.01  # 最低非零比例
```

### 3.2 bitwise.py — Bit 级类型定义

#### FloatFormat (Enum)

标准浮点格式:

| 枚举值 | bit 分布 |
|--------|---------|
| `FLOAT32` | 1 + 8 + 23 (sign + exp + mantissa) |
| `FLOAT16` | 1 + 5 + 10 |
| `BFLOAT16` | 1 + 8 + 7 |

#### BitLayout (dataclass)

通用 bit 分布配置，通过字母模板承载任意数据类型:

```python
@dataclass
class BitLayout:
    sign_bits: int = 0             # 符号位数
    exponent_bits: int = 0         # per-element 指数位数
    mantissa_bits: int = 0         # 尾数/数据位数
    name: str = ""                 # 格式名称
    shared_exponent_bits: int = 0  # 共享指数位数 (BFP)
    block_size: int = 1            # 共享指数的 block 大小
    template: str = ""             # 字母模板
```

**模板语法**:
- 普通格式: `"SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM"` (float32)
- BFP 格式: `"EEEEEEEE(SMMMMMMM)*16"` — 括号外为共享 bit，括号内为 per-element，`*N` 为 block 大小
- 字母含义: `S`=sign, `E`=exponent, `M`=mantissa, `I`=integer, `P`=parity, `F`=flag, `D`=data

**预定义实例**:

| 实例 | 模板 | 说明 |
|------|------|------|
| `FP32` | `SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM` | 32-bit float |
| `FP16` | `SEEEEEMMMMMMMMMM` | 16-bit float |
| `BFP16` | `EEEEEEEE(SMMMMMMMMMMMMMMM)*16` | Block FP16 |
| `BFP8` | `EEEEEEEE(SMMMMMMM)*16` | Block FP8 |
| `BFP4` | `EEEEEEEE(SMMM)*16` | Block FP4 |
| `INT8` | `SIIIIIII` | 有符号 8-bit 整数 |
| `UINT8` | `IIIIIIII` | 无符号 8-bit 整数 |

**关键属性**:
- `total_bits`: per-element 总 bit 数
- `display_name`: 显示名称
- `bit_template`: MSB→LSB 字母模板
- `bit_template_spaced`: 带空格分隔的模板 (按分组)
- `bit_group_labels`: 分组字母 → 语义标签映射
- `group_colors()`: 分组字母 → SVG 颜色映射

#### BitDiffSummary

```python
@dataclass
class BitDiffSummary:
    total_elements: int            # 总元素数
    diff_elements: int             # 有 bit 差异的元素数
    sign_flip_count: int           # 符号位翻转数
    exponent_diff_count: int       # 指数域差异元素数
    mantissa_diff_count: int       # 尾数域差异元素数
    per_bit_error_count: np.ndarray  # 每个 bit 位置的错误计数
    max_exponent_diff: int         # 最大指数偏移量
```

#### BitWarning / WarnLevel

```python
class WarnLevel(Enum):
    CRITICAL = "CRITICAL"   # 符号位翻转 / 指数大偏移 (>=2)
    WARNING  = "WARNING"    # 指数小偏移 (±1)
    INFO     = "INFO"       # 仅尾数差异 (量化正常损失)

@dataclass
class BitWarning:
    level: WarnLevel
    message: str
    count: int
    indices: np.ndarray     # 前 N 个问题索引
```

#### BitAnalysisResult / ModelBitAnalysis

```python
@dataclass
class BitAnalysisResult:
    fmt: Union[FloatFormat, BitLayout]
    summary: BitDiffSummary
    warnings: List[BitWarning]

@dataclass
class ModelBitAnalysis:
    per_op: Dict[str, BitAnalysisResult]    # 逐算子分析
    global_result: Optional[BitAnalysisResult]  # 全局分析
```

---

## 4. 子模块详细设计

### 4.1 metrics.py — 指标计算

**设计要点**: 单次遍历合并计算 + early exit 短路优化。

#### AllMetrics

将 QSNR、cosine、abs/rel error、exceed count 封装为单一数据类，避免重复 flatten + float64 转换。

#### 函数清单

| 函数 | 签名 | 说明 |
|------|------|------|
| `calc_all_metrics` | `(golden, result, atol, rtol) → AllMetrics` | 单次遍历计算全部指标，大数组 2-3x 提速 |
| `calc_all_metrics_early_exit` | `(golden, result, atol, rtol, min_qsnr, min_cosine, max_exceed_ratio) → AllMetrics` | 带 early exit 版本: cosine → exceed → QSNR 顺序检查，失败 case 额外 2-3x 提速 |
| `calc_qsnr` | `(golden, result) → float` | 量化信噪比 (dB) |
| `calc_cosine` | `(a, b) → float` | 余弦相似度 |
| `calc_abs_error` | `(golden, result) → (max_abs, mean_abs, abs_errors)` | 绝对误差统计 |
| `calc_rel_error` | `(golden, result) → (max_rel, mean_rel, rel_errors)` | 相对误差统计 |
| `calc_exceed_count` | `(golden, result, atol, rtol) → int` | 超阈值元素数 |
| `check_nan_inf` | `(data) → (nan_count, inf_count, total)` | NaN/Inf 检查 |
| `check_nonzero` | `(data) → (nonzero_count, total, ratio)` | 非零元素检查 |

#### 计算优化策略

```
calc_all_metrics_early_exit 流程:

1. flatten + float64 (一次)
2. diff, abs_err, g_sq_sum (一次)
3. cosine ← 最便宜 (复用 g_sq_sum + dot)
   ├─ cosine < min_cosine → early exit (填充剩余指标)
4. exceed_count ← 中等开销
   ├─ exceed_ratio > max → early exit
5. QSNR + relative error ← 最贵
6. 返回完整 AllMetrics
```

### 4.2 exact.py — 精确比对

#### compare_exact

```python
def compare_exact(golden, result, max_abs=0.0, max_count=0) → ExactResult
```

**行为**:
- `max_abs=0`: Bit 级精确比对 — view 为 `uint8` 逐字节比较
- `max_abs>0`: 允许小误差 — `|golden - result| > max_abs` 计数
- `max_count`: 允许超阈值的元素个数

#### compare_bit

```python
def compare_bit(golden: bytes, result: bytes) → bool
```

原始字节级对比，完全一致返回 `True`。

### 4.3 fuzzy.py — 模糊比对

#### compare_fuzzy

```python
def compare_fuzzy(golden, result, config=None) → FuzzyResult
```

**判定条件** (三条全部满足才 PASS):
1. 超阈值比例 ≤ `max_exceed_ratio`
2. QSNR ≥ `min_qsnr`
3. cosine ≥ `min_cosine`

内部委托 `calc_all_metrics_early_exit` 实现单次遍历 + early exit。

#### compare_isclose

```python
def compare_isclose(golden, result, atol=1e-5, rtol=1e-3, max_exceed_ratio=0.0) → FuzzyResult
```

类似 `numpy.isclose` 的简化接口，判定条件: `|result - golden| ≤ atol + rtol * |golden|`。

### 4.4 sanity.py — Golden 自检

#### check_golden_sanity

```python
def check_golden_sanity(golden_pure, golden_qnt=None, config=None) → SanityResult
```

**检查项**:

| 检查项 | 说明 | 阈值来源 |
|--------|------|----------|
| `non_zero` | 非零元素比例 ≥ 阈值 | `sanity_min_nonzero_ratio` |
| `no_nan_inf` | NaN/Inf 比例 ≤ 阈值 | `sanity_max_nan_ratio`, `sanity_max_inf_ratio` |
| `range_valid` | 数据非常数 (采样前 1000 个元素) | — |
| `qsnr_valid` | golden_qnt vs golden_pure 的 QSNR ≥ 阈值 | `sanity_min_qsnr` |

任一检查项失败 → `valid = False`。

#### check_data_sanity

通用数据自检 (无 Golden 对比)，检查非零 + NaN/Inf。

### 4.5 engine.py — 比对引擎

#### CompareEngine

协调精确比对、模糊比对、Golden 自检三个阶段，输出最终四态判定。

```python
class CompareEngine:
    def __init__(self, config: CompareConfig = None)
    def compare(self, dut_output, golden_pure, golden_qnt=None, ...) → CompareResult
    def compare_exact_only(self, dut_output, golden, ...) → CompareResult
    def compare_fuzzy_only(self, dut_output, golden, ...) → CompareResult
```

**`compare()` 执行流程**:

```
输入: dut_output, golden_pure, golden_qnt (可选)
     │
     ├─ 1. compare_exact(golden_pure, dut)     → ExactResult
     ├─ 2. compare_fuzzy(golden_pure, dut)     → FuzzyResult (fuzzy_pure)
     ├─ 3. compare_fuzzy(golden_qnt, dut)      → FuzzyResult (fuzzy_qnt)
     └─ 4. check_golden_sanity(golden_pure, golden_qnt) → SanityResult
                    │
                    ▼
     CompareResult.determine_status()
                    │
     ┌──────────────┼──────────────┐
     │              │              │
     ▼              ▼              ▼
  dut_passed?   golden_valid?   → CompareStatus (四态)
```

#### 便捷函数

```python
def compare_full(dut_output, golden_pure, golden_qnt=None, config=None, name="") → CompareResult
def determine_status(exact, fuzzy_pure, fuzzy_qnt, sanity) → CompareStatus
```

### 4.6 blocked.py — 分块比对

#### compare_blocked

```python
def compare_blocked(golden, result, block_size=1024, min_qsnr=30.0,
                    min_cosine=0.999, atol=1e-5, rtol=1e-3) → List[BlockResult]
```

将大张量按 `block_size` 分块，每块独立计算 QSNR/cosine/max_abs/exceed_count，用于定位误差集中区域。

#### BlockResult

```python
@dataclass
class BlockResult:
    offset: int       # 块起始偏移
    size: int         # 块大小
    qsnr: float       # 块 QSNR
    cosine: float     # 块 cosine
    max_abs: float    # 块最大绝对误差
    exceed_count: int # 块超阈值数
    passed: bool      # 块是否通过
```

#### 可视化

| 函数 | 说明 |
|------|------|
| `print_block_heatmap(blocks, cols=40)` | 文本热力图: `.`≥40dB, `o`≥20dB, `X`≥10dB, `#`<10dB |
| `find_worst_blocks(blocks, top_n=5)` | 找出最差的 N 个 block |

### 4.7 bitwise.py — Bit 级分析

#### compare_bitwise

```python
def compare_bitwise(golden, result, fmt=None, max_warning_indices=10) → BitAnalysisResult
```

**核心流程**:

```
输入: golden (np.ndarray), result (np.ndarray), fmt (FloatFormat|BitLayout)
     │
     ├─ 1. 格式检测 (fmt=None → 自动检测 dtype)
     │
     ├─ 2. BFP + fp32 自动量化
     │     fmt 是 BFP 且输入为 fp32?
     │     ├─ Yes: simulate_quantize → 量化精度损失 → fp32 bit pattern
     │     │       analysis_fmt = FP32 (用 1+8+23 分析)
     │     └─ No:  直接按 fmt 处理
     │
     ├─ 3. _to_uint: 转为无符号整型 (保留 bit pattern)
     │     fp32 → view(uint32), fp16 → view(uint16), bfloat16 → >>16
     │
     ├─ 4. XOR: g_uint ^ r_uint → 差异 bit mask
     │
     ├─ 5. 分域分析 (根据 sign_bits/exp_bits/mant_bits):
     │     ├─ sign:     xor >> (total-1) & 1 → sign_flip_count
     │     ├─ exponent: (uint >> mant_bits) & exp_mask → exp_diff
     │     └─ mantissa: uint & mant_mask → mant_diff
     │
     ├─ 6. per-bit 错误统计 (向量化):
     │     np.unpackbits(xor.view(uint8)) → reshape → sum(axis=0)
     │
     └─ 7. 生成告警:
           ├─ CRITICAL: 符号位翻转
           ├─ CRITICAL: 指数偏移 ≥ 2 (量级严重偏离)
           ├─ WARNING:  指数偏移 = 1 (约 2x 偏差)
           └─ INFO:     仅尾数差异 (量化正常损失)
```

#### BFP + fp32 自动量化机制

当用户传入 BFP 格式 (如 `BFP8`) 但数据实际为 fp32 时:

1. `_to_uint` 检测到 `fmt.shared_exponent_bits > 0` 且 `arr.dtype == float32`
2. 调用 `simulate_quantize(arr, qtype)` 模拟量化精度损失
3. 量化后 arr 仍为 fp32，使用 fp32 bit pattern (view uint32) 比较
4. `compare_bitwise` 将 `analysis_fmt` 设为 `FP32`，确保 sign/exp/mant 按 1+8+23 分析

这避免了直接将 fp32 的 IEEE 754 bit pattern 当作 8-bit 格式处理导致的静默错误。

#### compare_model_bitwise

```python
def compare_model_bitwise(per_op_pairs, fmt=None, final_pair=None) → ModelBitAnalysis
```

一键式模型级 bit 分析: 逐算子 + 全局。

#### 可视化函数

| 函数 | 说明 |
|------|------|
| `print_bit_template(fmt)` | 打印 bit 模板 (字母标识 S/E/M/I 等) |
| `print_bit_analysis(result, name)` | 打印 bit 级分析报告 (含模板 + 统计 + 告警) |
| `print_bit_heatmap(golden, result, fmt, block_size, cols)` | 文本热力图 (`.`/`o`/`X`/`#`) |
| `print_model_bit_analysis(result, name)` | 打印模型级逐算子 bit 表 |
| `gen_bit_heatmap_svg(golden, result, output_path, ...)` | SVG 热力图 (绿→黄→橙→红) |
| `gen_perbit_bar_svg(result, output_path, ...)` | SVG per-bit 错误分布条形图 |

### 4.8 report.py — 报告生成

| 函数 | 签名 | 说明 |
|------|------|------|
| `print_compare_table` | `(results: List[CompareResult])` | 打印比对结果表格 (exact/fuzzy/sanity/status) |
| `generate_text_report` | `(results, output_path=None) → str` | 生成文本报告 |
| `generate_json_report` | `(results, output_path=None) → dict` | 生成 JSON 报告 |

---

## 5. 比对引擎流程

### 5.1 完整比对时序

```
用户代码                CompareEngine              各子模块
  │                        │                         │
  │ compare_full(dut,      │                         │
  │   golden_pure,         │                         │
  │   golden_qnt)          │                         │
  │───────────────────────>│                         │
  │                        │ compare_exact()          │
  │                        │────────────────────────>│ exact.py
  │                        │<────────────────────────│ ExactResult
  │                        │                         │
  │                        │ compare_fuzzy(pure)      │
  │                        │────────────────────────>│ fuzzy.py
  │                        │<────────────────────────│ FuzzyResult
  │                        │                         │
  │                        │ compare_fuzzy(qnt)       │
  │                        │────────────────────────>│ fuzzy.py
  │                        │<────────────────────────│ FuzzyResult
  │                        │                         │
  │                        │ check_golden_sanity()    │
  │                        │────────────────────────>│ sanity.py
  │                        │<────────────────────────│ SanityResult
  │                        │                         │
  │                        │ determine_status()       │
  │                        │  ┌──────────────┐       │
  │                        │  │ dut_passed?  │       │
  │                        │  │ golden_valid?│       │
  │                        │  │ → 四态判定    │       │
  │                        │  └──────────────┘       │
  │<───────────────────────│                         │
  │ CompareResult          │                         │
```

### 5.2 状态判定矩阵

```
                  Golden 自检
                 PASS    FAIL
             ┌────────┬───────────────┐
DUT    PASS  │  PASS  │ GOLDEN_SUSPECT│
比对        ├────────┼───────────────┤
       FAIL  │DUT_ISSUE│ BOTH_SUSPECT │
             └────────┴───────────────┘
```

DUT 通过判定逻辑: `exact.passed` **或** `fuzzy_qnt.passed` 任一为 True。

### 5.3 Bit 级分析流程

```
用户代码                compare_bitwise            内部函数
  │                        │                         │
  │ compare_bitwise(       │                         │
  │   golden, dut,         │                         │
  │   fmt=BFP8)            │                         │
  │───────────────────────>│                         │
  │                        │                         │
  │                        │ BFP + fp32?              │
  │                        │ ├─ Yes: analysis_fmt=FP32│
  │                        │ └─ No: analysis_fmt=fmt  │
  │                        │                         │
  │                        │ _to_uint(golden, fmt)    │
  │                        │────────────────────────>│
  │                        │   [BFP+fp32: simulate_  │
  │                        │    quantize → view u32] │
  │                        │<────────────────────────│
  │                        │                         │
  │                        │ _to_uint(result, fmt)    │
  │                        │────────────────────────>│
  │                        │<────────────────────────│
  │                        │                         │
  │                        │ xor = g_uint ^ r_uint   │
  │                        │                         │
  │                        │ 分域分析 (sign/exp/mant) │
  │                        │                         │
  │                        │ _per_bit_count(xor)      │
  │                        │────────────────────────>│
  │                        │   [unpackbits → reshape │
  │                        │    → sum(axis=0)]       │
  │                        │<────────────────────────│
  │                        │                         │
  │                        │ _generate_warnings()     │
  │                        │────────────────────────>│
  │                        │<────────────────────────│
  │                        │                         │
  │<───────────────────────│                         │
  │ BitAnalysisResult      │                         │
```

---

## 6. 外部依赖与接口

### 6.1 formats 模块接口

| 接口 | 模块 | 用途 |
|------|------|------|
| `load(path, fmt, qtype, shape)` | `formats.base` | 加载 bin 文件，自动反量化为 fp32 |
| `load_dir(directory, bm)` | `formats.base` | 自动扫描目录加载所有 DUT bin |
| `simulate_quantize(data, qtype)` | `formats.quantize` | 模拟量化精度损失 (quantize → dequantize) |
| `quantize(data, qtype)` | `formats.quantize` | 量化数据 |
| `dequantize(data, qtype, meta)` | `formats.quantize` | 反量化数据 |
| `generate_fake_dut(ref, qtype, noise)` | `formats.quantize` | 生成模拟 DUT 数据 |

### 6.2 datagen 模块接口

| 接口 | 模块 | 用途 |
|------|------|------|
| `Model(seed, qtype, precision, data_dir, bm)` | `datagen` | Model DSL，自动生成权重 |
| `Model.export(output_dir, bm=...)` | `datagen` | 导出为 DUT 格式 bin 文件 |
| `Model.input(shape)` | `datagen` | 定义输入 (data_dir 模式从文件加载) |
| `Model.linear/gelu/softmax/...` | `datagen` | 算子调用，自动生成权重并计算 golden |
| `DataGenerator.generate_four_track(...)` | `datagen` | 生成四种比数 golden |

### 6.3 文件命名约定

导出文件: `{bm}_{name}_{NxMxK}.{qtype}.bin`

示例: `encoder_linear_0_weight_64x64.bfp4.bin`

| 字段 | 说明 |
|------|------|
| `bm` | benchmark 前缀 (如 `encoder`) |
| `name` | tensor 名 (如 `linear_0_weight`) |
| `NxMxK` | shape 维度 (如 `64x64`) |
| `qtype` | 量化类型 (如 `bfp4`, `bfp8`, `float32`) |

`load()`/`load_dir()` 从文件名自动推断 qtype 和 shape，实现全自动加载。

---

## 7. CLI 命令接口

模块: `aidevtools.commands.compare`

```
compare <action> [options]
```

### 7.1 子命令

| 子命令 | 说明 | 关键参数 |
|--------|------|----------|
| `single` | 单次比对两个文件 | `--golden`, `--result`, `--dtype`, `--shape` |
| `fuzzy` | 模糊比对 | `--golden`, `--result`, `--dtype`, `--shape` |
| `bitwise` | Bit 级分析 + 热力图 + SVG | `--golden`, `--result`, `--dtype`, `--output` |
| `convert` | 量化类型转换 | `--golden`, `--target_dtype`, `--output` |
| `qtypes` | 列出支持的量化类型 | — |
| `dump` | 导出 Golden 数据 | `--output` |
| `clear` | 清空 Golden 记录 | — |
| `xlsx *` | xlsx 相关子命令 | `--xlsx`, `--ops` |

### 7.2 bitwise 命令输出

执行 `compare bitwise` 后输出:
1. Bit 级分析报告 (文本)
2. Bit 热力图 (文本)
3. `{output}/bit_heatmap.svg` — SVG 热力图
4. `{output}/perbit_bar.svg` — per-bit 错误分布条形图

返回码: `0` = 无 CRITICAL, `1` = 有 CRITICAL。

---

## 8. 使用示例

### 8.1 基础比对

```python
from aidevtools.compare import compare_full, CompareConfig

# 默认配置比对
result = compare_full(dut_output, golden_pure, golden_qnt)
print(f"Status: {result.status.value}")  # PASS / DUT_ISSUE / ...

# 自定义阈值
config = CompareConfig(fuzzy_min_qsnr=25.0, fuzzy_min_cosine=0.99)
result = compare_full(dut, golden, config=config)
```

### 8.2 分块定位

```python
from aidevtools.compare import compare_blocked, print_block_heatmap, find_worst_blocks

blocks = compare_blocked(golden, dut, block_size=1024)
print_block_heatmap(blocks)
worst = find_worst_blocks(blocks, top_n=3)
for b in worst:
    print(f"offset={b.offset}, QSNR={b.qsnr:.1f} dB")
```

### 8.3 Bit 级分析

```python
from aidevtools.compare import compare_bitwise, print_bit_analysis, FP32, BFP8
from aidevtools.compare.bitwise import gen_bit_heatmap_svg, gen_perbit_bar_svg

# 标准 float32 比对
result = compare_bitwise(golden, dut, fmt=FP32)
print_bit_analysis(result, name="Linear_0")

# BFP8 + fp32 输入 (自动量化)
result = compare_bitwise(golden_fp32, dut_fp32, fmt=BFP8)

# 生成 SVG
gen_bit_heatmap_svg(golden, dut, "heatmap.svg", fmt=FP32)
gen_perbit_bar_svg(result, "perbit.svg")
```

### 8.4 Model DSL + 数据回放

```python
from aidevtools.datagen import Model
from aidevtools.compare.bitwise import compare_bitwise, FP32

# 生成 + 导出
with Model(seed=42, qtype="bfp8", bm="encoder") as m:
    x = m.input((2, 16, 64))
    y = m.linear(x, out_features=64)
    y = m.gelu(y)
    m.export("./golden/", bm="encoder")

# 数据回放
with Model(seed=42, qtype="bfp8", data_dir="./golden/", bm="encoder") as m2:
    x2 = m2.input((2, 16, 64))
    y2 = m2.linear(x2, out_features=64)
    y2 = m2.gelu(y2)

# Bit 级比对
result = compare_bitwise(y.golden, y2.golden, fmt=FP32)
```

### 8.5 逐算子模型比对

```python
from aidevtools.compare.bitwise import compare_model_bitwise, print_model_bit_analysis, FP32

pairs = {
    "Q_proj": (golden_q, dut_q),
    "K_proj": (golden_k, dut_k),
    "Attention": (golden_attn, dut_attn),
}
result = compare_model_bitwise(pairs, fmt=FP32, final_pair=(golden_out, dut_out))
print_model_bit_analysis(result, name="Encoder")
```

---

## 9. 测试策略

### 9.1 测试文件

| 文件 | 说明 | 用例数 |
|------|------|--------|
| `tests/ut/compare/test_compare.py` | 核心比对逻辑 (exact/fuzzy/sanity/engine/report) | — |
| `tests/ut/compare/test_bitwise.py` | Bit 级分析全覆盖 | — |
| `tests/ut/compare/test_optimizations.py` | 指标计算优化验证 | — |
| `tests/ut/test_model_data_dir.py` | Model data_dir 数据回放 | 8 |
| `tests/st/test_compare_cmd.py` | CLI 命令集成测试 | — |

### 9.2 关键测试场景

| 场景 | 验证点 |
|------|--------|
| bit-exact 数据 | compare_exact/compare_bitwise 应报告 0 差异 |
| 量化精度损失 | compare_fuzzy 应检测到 QSNR 下降，compare_bitwise 应报告 mantissa-only diff |
| 符号位翻转 | compare_bitwise 应生成 CRITICAL 告警 |
| 指数偏移 | 偏移 ≥2 → CRITICAL，偏移 =1 → WARNING |
| Golden 全零 | check_golden_sanity 应报告 non_zero 失败 |
| Golden 含 NaN | check_golden_sanity 应报告 no_nan_inf 失败 |
| BFP + fp32 输入 | _to_uint 应自动 simulate_quantize 而非静默错误 |
| data_dir 回放确定性 | 两次 data_dir 加载应产生 bit-exact 输出 |
| 损坏权重检测 | 修改权重后 data_dir 回放应产生不同输出 |

### 9.3 注意事项

- **量化损失测试**: 不要对比原始 fp32 与 export→reload 的数据 (BFP4 仅 2 位尾数，损失极大)。应对比两次 reload (确定性一致)。
- **`hash()` 不确定性**: Python 3 默认 PYTHONHASHSEED 随机化。Demo 中使用 `int.from_bytes(name.encode(), 'little') % 2**31` 替代 `hash()` 确保跨进程确定性。

---

## 10. 附录

### 10.1 公共 API 清单

`aidevtools.compare.__all__` 导出:

**类型**:
`CompareConfig`, `CompareResult`, `CompareStatus`, `ExactResult`, `FuzzyResult`, `SanityResult`

**引擎**:
`CompareEngine`, `compare_full`, `determine_status`

**精确比对**:
`compare_exact`, `compare_bit`

**模糊比对**:
`compare_fuzzy`, `compare_isclose`

**Golden 自检**:
`check_golden_sanity`, `check_data_sanity`

**指标计算 (优化版)**:
`AllMetrics`, `calc_all_metrics`, `calc_all_metrics_early_exit`

**指标计算 (独立函数)**:
`calc_qsnr`, `calc_cosine`, `calc_abs_error`, `calc_rel_error`, `calc_exceed_count`, `check_nan_inf`, `check_nonzero`

**分块比对**:
`BlockResult`, `compare_blocked`, `print_block_heatmap`, `find_worst_blocks`

**Bit 级分析**:
`FloatFormat`, `BitLayout`, `BFP8`, `BFP4`, `INT8`, `UINT8`, `WarnLevel`, `BitDiffSummary`, `BitWarning`, `BitAnalysisResult`, `compare_bitwise`, `print_bit_template`, `print_bit_analysis`, `print_bit_heatmap`, `gen_bit_heatmap_svg`, `gen_perbit_bar_svg`

**报告**:
`print_compare_table`, `generate_text_report`, `generate_json_report`

### 10.2 Demo 索引

| Demo | 路径 | 说明 |
|------|------|------|
| 06 | `demos/compare/06_encoder_bfp8/` | Encoder BFP8 全流程比对 |
| 07 | `demos/compare/07_qa_encoder_bfp8/` | 量化感知 Encoder 比对 |
| 08 | `demos/compare/08_bitwise_encoder_bfp8/` | Bit 级分析 + SVG 可视化 |
| 09 | `demos/compare/09_autoload_replay/` | 数据导出 + 自动回放比对 |

### 10.3 SVG 颜色编码

**热力图** (`gen_bit_heatmap_svg`):

| 颜色 | 含义 |
|------|------|
| 绿 `#4caf50` | bit-exact (0% diff) |
| 浅绿 `#8bc34a` | < 0.1% diff |
| 黄 `#ffeb3b` | < 1% diff |
| 橙 `#ff9800` | < 10% diff |
| 红 `#f44336` | ≥ 10% diff |

**per-bit 条形图** (`gen_perbit_bar_svg`):

| 颜色 | 分组 |
|------|------|
| 红 `#f44336` | sign |
| 橙 `#ff9800` | exponent |
| 蓝 `#2196f3` | mantissa / integer / data |
| 绿/紫/灰 | 其他自定义分组 |
