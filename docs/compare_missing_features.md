# Compare 模块缺失功能清单

> 最后更新: 2026-02-09
> 生成原因: 策略模式重构后，部分功能被移除

---

## 1. 可视化功能 (Visualization)

### 1.1 文本输出

**已移除函数**：
- `print_bit_analysis(result, name)` - 打印 bit 分析结果
- `print_bit_heatmap(golden, result, fmt, block_size, cols)` - 打印字符热力图

**功能描述**：
```python
# 旧代码示例
result = compare_bitwise(golden, dut, fmt=FP32)
print_bit_analysis(result, name="conv1")
# 输出：
# [conv1] Bit-Level Analysis (float32)
# Total: 1000, Diff: 25 (2.5%)
# Sign flips: 2 (0.2%) [!]
# Exponent diffs: 10 (1.0%)
# Mantissa diffs: 15 (1.5%)
```

**影响范围**：
- CLI 工具无法友好显示 bit 分析结果
- 调试时需要手动检查 `result.summary` 和 `result.warnings`

**预计恢复工作量**：2-3 小时

---

### 1.2 SVG 可视化

**已移除函数**：
- `gen_bit_heatmap_svg(golden, result, output_path, fmt, block_size)` - 生成热力图
- `gen_perbit_bar_svg(result, output_path)` - 生成 per-bit 误差分布条形图

**功能描述**：
- 热力图：按 block 显示误差密度，颜色深浅表示错误率
- 条形图：32个柱子显示每个 bit position 的错误计数

**影响范围**：
- 无法生成报告中的可视化图表
- 需要手动分析数值数据

**预计恢复工作量**：3-4 小时（需要 matplotlib 或纯 SVG 生成）

---

## 2. 统计功能 (Statistics)

### 2.1 Per-bit 错误计数

**已移除属性**：
- `BitAnalysisSummary.per_bit_error_count: List[int]` - 长度为 total_bits 的数组

**功能描述**：
```python
# 旧代码示例
result = compare_bitwise(golden, dut, fmt=FP32)
print(result.summary.per_bit_error_count)
# 输出: [100, 0, 5, 12, ...] (32个数字)
# 表示 bit0 有100个错误，bit1 有0个错误，等等
```

**影响范围**：
- 无法定位哪些 bit position 最容易出错
- 无法分析量化算法对不同 bit 的影响

**预计恢复工作量**：1 小时

---

### 2.2 差异比例

**已移除属性**：
- `BitAnalysisSummary.diff_ratio: float` - diff_elements / total_elements

**功能描述**：
```python
# 旧代码示例
ratio = result.summary.diff_ratio  # 0.025 表示 2.5% 不匹配
```

**影响范围**：
- 需要手动计算 `diff_elements / total_elements`
- 影响较小，容易绕过

**预计恢复工作量**：15 分钟（加一行计算即可）

---

## 3. Bit 模板系统 (Bit Template System)

### 3.1 模板属性

**已移除属性**：
- `BitLayout.bit_template: str` - 例如 "SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM"
- `BitLayout.bit_template_spaced: str` - 例如 "S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM"
- `BitLayout.shared_exponent_bits: int` - 共享指数位数（BFP 格式）
- `BitLayout.block_size: int` - BFP block 大小
- `BitLayout.display_name: str` - 自动生成的显示名（替换为简单的 `name`）

**功能描述**：
```python
# 旧代码示例
layout = BitLayout.from_template(
    "SEEEEMMMM",  # per-element
    name="fp8_e4m3",
    shared_template="EEEEEEEE",  # 共享指数
    block_size=16
)
print(layout.bit_template_spaced)  # "S EEEE MMMM"
```

**影响范围**：
- 无法可视化显示 bit 布局
- 无法支持复杂的 BFP 共享指数格式
- 现在只能手动指定 (sign_bits, exponent_bits, mantissa_bits)

**预计恢复工作量**：2-3 小时

---

### 3.2 模板构造器

**已移除方法**：
- `BitLayout.from_template(template, name, shared_template, block_size)` - 从字符串模板创建

**功能描述**：
```python
# 旧代码示例
layout = BitLayout.from_template("SEEEEMMMM", name="fp8_e4m3")
# 自动解析: sign=1, exp=4, mant=4
```

**影响范围**：
- 必须手动计算并传入数值
- 无法通过模板快速定义新格式

**预计恢复工作量**：1 小时

---

### 3.3 模板打印

**已移除函数**：
- `print_bit_template(fmt)` - 打印格式模板

**功能描述**：
```python
# 旧代码示例
print_bit_template(FP32)
# 输出：
# float32 (32 bits)
# S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM
# S=sign, E=exponent, M=mantissa
```

**影响范围**：
- 无法快速查看格式定义
- 需要手动打印属性

**预计恢复工作量**：30 分钟

---

## 4. 模型级分析 (Model-level Analysis)

### 4.1 模型比对函数

**已移除函数**：
- `compare_model_bitwise(per_op_pairs, fmt, final_pair)` - 一键式模型 bit 比对
- 返回类型：`ModelBitAnalysis`

**功能描述**：
```python
# 旧代码示例
result = compare_model_bitwise(
    per_op_pairs={
        "conv1": (golden1, dut1),
        "bn1": (golden2, dut2),
    },
    fmt=BFP8,
    final_pair=(golden_final, dut_final)
)
print(result.has_critical)  # 整体是否有 CRITICAL 告警
print_model_bit_analysis(result, name="MyModel")
```

**影响范围**：
- 无法批量比对多个算子
- 需要手动循环调用 `compare_bitwise()`
- 无整体统计汇总

**预计恢复工作量**：1-2 小时

---

### 4.2 模型结果类型

**已移除类型**：
- `ModelBitAnalysis` - 包含 per_op 和 global_result

**功能描述**：
```python
class ModelBitAnalysis:
    per_op: Dict[str, BitAnalysisResult]
    global_result: Optional[BitAnalysisResult]
    has_critical: bool  # 任一算子有 CRITICAL
```

**影响范围**：
- 见 4.1

**预计恢复工作量**：见 4.1

---

### 4.3 模型打印函数

**已移除函数**：
- `print_model_bit_analysis(result, name)` - 打印模型级别的 bit 分析汇总

**功能描述**：
```python
# 旧代码示例
print_model_bit_analysis(result, name="ResNet18")
# 输出：
# [ResNet18] Model Bit Analysis
#
# 逐算子比对：
#   conv1: 2 sign flips [!]
#   bn1: OK
#
# 合计: 2 CRITICAL, 0 WARNING
#
# Global Output: 5 sign flips [!]
```

**影响范围**：
- 见 4.1

**预计恢复工作量**：见 4.1

---

## 5. 整数类型支持 (Integer Types)

### 5.1 整数预设

**已移除类型**：
- `INT8` - BitLayout(1, 0, 7, "int8")
- `UINT8` - BitLayout(0, 0, 8, "uint8")

**功能描述**：
```python
# 旧代码示例
result = compare_bitwise(golden_int8, dut_int8, fmt=INT8)
```

**影响范围**：
- 无法直接比对 int8 数据
- 需要用 BFP8 或自定义 BitLayout 代替

**预计恢复工作量**：10 分钟（加两行定义即可）

---

## 6. 旧 Compare Engine API

### 6.1 独立比对函数

**已移除函数**：
- `compare_exact(golden, result, max_abs, max_count)` - 精确比对
- `compare_fuzzy(golden, result, config)` - 模糊比对
- `check_golden_sanity(golden_pure, golden_qnt, config)` - Golden 自检

**功能描述**：
```python
# 旧代码示例
exact_result = compare_exact(golden, dut, max_abs=1e-5)
fuzzy_result = compare_fuzzy(golden, dut, config)
sanity_result = check_golden_sanity(golden_pure, golden_qnt, config)
```

**影响范围**：
- 必须使用策略模式：`CompareEngine.run()`
- 70个测试用例需要重写

**预计恢复工作量**：
- 不建议恢复（设计决策：策略模式更灵活）
- 如需恢复：5-8 小时重写测试

---

### 6.2 Engine 便捷方法

**已移除方法**：
- `CompareEngine.compare(dut, golden, golden_qnt, name, op_id)` - 完整比对
- `CompareEngine.compare_exact_only(dut, golden, name)` - 仅精确比对
- `CompareEngine.compare_fuzzy_only(dut, golden, name)` - 仅模糊比对

**新 API**：
```python
# 新代码
engine = CompareEngine.standard()
result = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)

# 访问结果
exact = result.get('exact')
fuzzy_pure = result.get('fuzzy_pure')
status = result.get('status')
```

**影响范围**：
- 所有旧代码需要迁移
- 返回值从对象变为字典

**预计恢复工作量**：
- 不建议恢复（设计决策）
- 可添加兼容层：2-3 小时

---

## 7. 其他细节

### 7.1 BitAnalysisStrategy 参数

**已移除参数**：
- `max_warning_indices` - 现在固定为 10

**影响**：
- 无法自定义告警索引数量
- 影响极小

---

## 总结

### 按优先级分类

**优先级 1 - 高影响**：
1. 可视化功能 (1.1, 1.2) - 6 小时
2. Per-bit 统计 (2.1) - 1 小时
3. 模型级分析 (4.1-4.3) - 2 小时

**优先级 2 - 中影响**：
1. Bit 模板系统 (3.1-3.3) - 4 小时

**优先级 3 - 低影响**：
1. diff_ratio (2.2) - 15 分钟
2. 整数类型 (5.1) - 10 分钟

**不建议恢复**：
- 旧 Compare Engine API (6.1, 6.2) - 设计决策

### 总工作量估算

- **核心功能恢复**：9 小时（优先级 1）
- **完整恢复**：13.5 小时（优先级 1-3）

---

## 替代方案

### 可视化
- 使用 Jupyter Notebook 手动绘图
- 使用第三方工具（如 pandas profiling）

### 模型级分析
```python
# 手动实现简单版本
results = {}
for name, (golden, dut) in per_op_pairs.items():
    results[name] = BitAnalysisStrategy.compare(golden, dut, fmt=fmt)

has_critical = any(r.has_critical for r in results.values())
```

### 统计功能
```python
# 手动计算 diff_ratio
diff_ratio = result.summary.diff_elements / result.summary.total_elements
```

---

生成时间: 2026-02-09 15:30
