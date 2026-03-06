# 比对套件 API 使用手册

> 版本: 3.0
> 日期: 2026-03
> 适用于: aidevtools >= 2.0.0

## 快速开始

```python
from aidevtools.compare import CompareEngine, CompareConfig

# 创建引擎 (渐进式，默认早停)
engine = CompareEngine.progressive(config=CompareConfig(
    fuzzy_min_qsnr=30.0,
    fuzzy_min_cosine=0.999,
))

# 执行比对
results = engine.run(dut=dut_output, golden=golden_fp32)

# 查看结果
exact = results.get("exact")
if exact:
    print(f"Exact: {'PASS' if exact.passed else 'FAIL'}")
    print(f"Diff bits: {exact.diff_bits}/{exact.total_bits} ({exact.diff_bit_ratio:.4%})")

fuzzy = results.get("fuzzy_pure")
if fuzzy:
    print(f"QSNR: {fuzzy.qsnr:.2f} dB, Cosine: {fuzzy.cosine:.6f}")
```

---

## 引擎 API

### CompareEngine.progressive()

```python
engine = CompareEngine.progressive(
    config=None,        # CompareConfig, 可选
    block_size=64,      # L3 分块大小
    deep=False,         # True=三级全执行, False=早停
)
results = engine.run(
    dut=dut_array,           # DUT 输出 (np.ndarray)
    golden=golden_array,     # Golden 参考 (np.ndarray)
    golden_qnt=None,         # 量化感知 Golden (可选)
    metadata=None,           # 额外元数据 (可选)
)
```

**分级流程**:

| 级别 | 策略 | 早停条件 (deep=False) |
|------|------|----------------------|
| L1 | Exact | exact.passed → 停止 |
| L2 | Fuzzy(pure) + Fuzzy(qnt) + Sanity | fuzzy.passed → 停止 |
| L3 | BitAnalysis + Blocked | 最后一级 |

`deep=True` 时忽略早停条件，三级全部执行。

### CompareEngine.model_progressive()

```python
analyzer = CompareEngine.model_progressive(config=config)
results = analyzer.progressive_analyze(
    pairs={"op_name": (golden, dut), ...},
    l1_threshold=0.9,   # L1 通过率阈值
    l2_threshold=0.8,   # L2 通过率阈值
)
analyzer.print_summary(results)
```

适用于大模型场景 (几十到几百个算子)，根据整体通过率决定是否深入下一级。

### 自定义策略

```python
from aidevtools.compare.strategy import CompositeStrategy, ExactStrategy, FuzzyStrategy

engine = CompareEngine(
    strategy=CompositeStrategy([
        ExactStrategy(),
        FuzzyStrategy(use_golden_qnt=False),
    ]),
    config=config,
)
```

---

## 结果格式

`engine.run()` 返回 `Dict[str, Any]`:

```python
{
    "exact": ExactResult,                    # L1
    "fuzzy_pure": FuzzyResult,              # L2
    "fuzzy_qnt": FuzzyResult,               # L2 (如有 golden_qnt)
    "sanity": SanityResult,                  # L2
    "bit_analysis_fp32": BitAnalysisResult, # L3
    "blocked_64": [BlockResult, ...],        # L3
    "_executed_levels": ["L1_quick", ...],
    "_stopped_at": "L1_quick",
}
```

---

## Result 类型参考

### ExactResult

```python
from aidevtools.compare.strategy.exact import ExactResult

exact.passed            # bool
exact.mismatch_count    # int - 不匹配元素数
exact.first_diff_offset # int - 第一个差异位置 (-1=无差异)
exact.max_abs           # float - 最大绝对误差
exact.total_elements    # int
exact.diff_bits         # int - XOR popcount
exact.total_bits        # int
exact.diff_bit_ratio    # float (property) - diff_bits / total_bits
```

### FuzzyResult

```python
from aidevtools.compare.strategy.fuzzy import FuzzyResult

fuzzy.passed          # bool
fuzzy.qsnr            # float (dB)
fuzzy.cosine          # float
fuzzy.max_abs         # float
fuzzy.mean_abs        # float
fuzzy.max_rel         # float
fuzzy.total_elements  # int
fuzzy.exceed_count    # int
```

### SanityResult

```python
from aidevtools.compare.strategy.sanity import SanityResult

sanity.valid        # bool - Golden 是否有效
sanity.checks       # Dict[str, bool] - 各检查项
sanity.messages     # List[str] - 告警消息
sanity.non_zero     # bool
sanity.no_nan_inf   # bool
sanity.range_valid  # bool
sanity.qsnr_valid   # bool
```

### BlockResult

```python
from aidevtools.compare.strategy.blocked import BlockResult

block.offset       # int - 块起始偏移
block.size         # int
block.qsnr         # float (dB)
block.cosine       # float
block.max_abs      # float
block.exceed_count # int
block.passed       # bool
```

### BitAnalysisResult

```python
from aidevtools.compare.strategy.bit_analysis import BitAnalysisResult

result.fmt                            # FloatFormat | BitLayout
result.summary.total_elements         # int
result.summary.diff_elements          # int
result.summary.sign_flip_count        # int
result.summary.exponent_diff_count    # int
result.summary.mantissa_diff_count    # int
result.summary.max_exponent_diff      # int
result.warnings                       # List[BitWarning]
result.has_critical                    # bool
```

---

## 策略静态方法

每个策略都可以独立使用，无需通过引擎:

### ExactStrategy

```python
from aidevtools.compare.strategy import ExactStrategy

result = ExactStrategy.compare(golden, dut, max_abs=0.0, max_count=0)
is_same = ExactStrategy.compare_bytes(golden_bytes, dut_bytes)
```

### FuzzyStrategy

```python
from aidevtools.compare.strategy import FuzzyStrategy

result = FuzzyStrategy.compare(golden, dut, config=CompareConfig())
result = FuzzyStrategy.compare_isclose(golden, dut, atol=1e-5, rtol=1e-3)
```

### SanityStrategy

```python
from aidevtools.compare.strategy import SanityStrategy

result = SanityStrategy.compare(golden_pure, golden_qnt, config)
result = SanityStrategy.check_data(data, name="input")
```

### BitAnalysisStrategy

```python
from aidevtools.compare.strategy import BitAnalysisStrategy, FP32

result = BitAnalysisStrategy.compare(golden, dut, fmt=FP32)
BitAnalysisStrategy.print_result(result, name="linear_0")
```

### BlockedStrategy

```python
from aidevtools.compare.strategy import BlockedStrategy

blocks = BlockedStrategy.compare(golden, dut, block_size=1024)
BlockedStrategy.print_heatmap(blocks)
```

---

## 报告 API

### 文字报告

```python
from aidevtools.compare.report.text_report import (
    print_joint_report,
    print_strategy_table,
    generate_strategy_json,
)

# 单算子联合报告 (表格 + bit 统计 + 分析详情 + 热力图)
print_joint_report(results, name="softmax")

# 多算子汇总表格
print_strategy_table([results1, results2], names=["softmax", "linear"])

# JSON 输出
json_data = generate_strategy_json(results, name="softmax")
```

### 可视化报告

```python
from aidevtools.compare.report.text_report import visualize_joint_report

page = visualize_joint_report(results, name="softmax")
page.render("report.html")
```

---

## CompareConfig 推荐阈值

```python
# BFP8 量化
config_bfp8 = CompareConfig(
    fuzzy_min_qsnr=15.0,
    fuzzy_min_cosine=0.98,
    fuzzy_max_exceed_ratio=0.05,
)

# FP16 量化
config_fp16 = CompareConfig(
    fuzzy_min_qsnr=30.0,
    fuzzy_min_cosine=0.999,
)

# FP32 计算
config_fp32 = CompareConfig(
    fuzzy_min_qsnr=60.0,
    fuzzy_min_cosine=0.9999,
)
```

---

## 精度指标参考

| 指标 | 公式 | 参考值 | 说明 |
|------|------|--------|------|
| max_abs | max(\|g-r\|) | < 1e-5 | 最大绝对误差 |
| mean_abs | mean(\|g-r\|) | < 1e-6 | 平均绝对误差 |
| QSNR | 10*log10(signal/noise) | > 30dB | 量化信噪比 |
| cosine | dot(g,r)/(norm*norm) | > 0.999 | 余弦相似度 |
| diff_bit_ratio | XOR popcount / total bits | < 1% | bit 级差异率 |

---

## CLI 命令

```bash
# 渐进式比对 (自动解读文件名)
compare diff --golden=softmax_bfp8_2x16x64.txt --result=softmax_bfp8_2x16x64_result.txt

# 指定参数
compare diff --golden=a.txt --result=b.txt --qtype=bfp8 --shape=2,16,64 --format=hex_text

# 深度模式 (三级全执行)
compare diff --golden=a.txt --result=b.txt --engine=deep

# 类型转换
compare convert --golden=a.bin --target_dtype=float16

# 列出量化类型
compare qtypes
```

---

## 自定义策略开发

```python
from aidevtools.compare.strategy import CompareStrategy, CompareContext

class MyStrategy(CompareStrategy):
    @property
    def name(self) -> str:
        return "my_custom"

    def run(self, ctx: CompareContext) -> dict:
        diff = np.abs(ctx.golden - ctx.dut)
        return {"max_diff": float(diff.max()), "passed": diff.max() < 0.01}
```

使用:

```python
engine = CompareEngine(strategy=CompositeStrategy([
    ExactStrategy(),
    MyStrategy(),
]))
```

---

## 从 v2.0 迁移

| v2.0 (已删除) | v3.0 |
|----------------|------|
| `CompareEngine.standard()` | `CompareEngine.progressive()` |
| `CompareEngine.quick()` | `CompareEngine.progressive()` (默认早停) |
| `CompareEngine.deep()` | `CompareEngine.progressive(deep=True)` |
| `CompareEngine.minimal()` | 自定义 `CompositeStrategy([FuzzyStrategy()])` |
| `CompareStatus` | 检查 `sanity.valid` + `fuzzy.passed` |
| `CompareResult` | 返回 `Dict[str, Any]` |
| `compare_full()` | `CompareEngine.progressive().run()` |
| `result.get("status")` | 通过 `sanity.valid` / `fuzzy.passed` 判断 |
| `BitXorStrategy` | 合并到 `ExactResult.diff_bits` |

---

## 参考

- [设计说明书](./design/compare_module_design.md)
- [Demo 示例](../demos/compare/)
- [源代码](../aidevtools/compare/)
