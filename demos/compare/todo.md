# Compare Demo 实现状态

## 总览

**所有 Demo 已完成实现 ✅** (共 7 个)

- **Phase 1 (No DUT)**: Demo 01-03 ✓
- **Phase 2 (With DUT)**: Demo 04, 06-08 ✓
- **已移除**: Demo 05（不需要单独错误注入 demo）

---

## Phase 1: No DUT（数据生成策略验证）

### ✅ Demo 01: Encoder - Pure vs FP16
- **文件**: `compare_01_encoder_pure_vs_fp16/run.py`
- **状态**: ✅ 已完成
- **模型**: Encoder (batch=2, seq=16, hidden=64, ffn=256, 10个算子)
- **对比**: fp32 计算 vs fp16 计算
- **目标**: 测试 fp16 计算引入的精度损失
- **预期结果**:
  - QSNR > 60 dB
  - Cosine > 0.999
  - Sanity 全部通过（无 NaN/Inf）

### ✅ Demo 02: Encoder - Pure vs FuzzQ
- **文件**: `compare_02_encoder_pure_vs_fuzzq/run.py`
- **状态**: ✅ 已完成
- **模型**: Encoder (10个算子)
- **对比**: 纯随机数据 vs bfp8 量化反量化数据
- **目标**: 测试 FuzzQ（量化反量化）引入的数据误差
- **预期结果**:
  - QSNR ~17 dB（bfp8 低精度）
  - Cosine ~0.987
  - 展示 FuzzQ 特点

### ✅ Demo 03: Encoder - QA Pure vs QA FuzzQ
- **文件**: `compare_03_encoder_fuzzq_vs_qa/run.py`
- **状态**: ✅ 已完成
- **模型**: Encoder (10个算子)
- **对比**: QA 数据 vs QA 数据 + bfp8 量化
- **目标**: 测试 QA 策略对量化误差的优化效果
- **预期结果**:
  - QSNR ~32.1 dB（比 Pure 提升 15.1 dB）
  - Cosine ~0.9999
  - 证明 QA 策略有效性
- **QA 参数**: center=1.0, amplitude=0.02

---

## Phase 2: With DUT（三路执行 + 四比数 + 四态判定）

### ✅ Demo 04: MatMul - Pure vs DUT
- **文件**: `compare_04_matmul_pure_vs_dut/run.py`
- **状态**: ✅ 已完成
- **模型**: 单个 MatMul 算子 (64×64 @ 64×128)
- **精度**: Pure fp32
- **三路执行**:
  - PyTorch Golden: 纯 fp32 计算
  - CPU Golden: cpu_golden 后端执行
  - DUT: CPU Golden + 小量随机噪声 (1e-5)
- **四比数**: Track 1-4 (Pure, Local, HW, QA)
- **四态判定**: PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT
- **验证**:
  - PyTorch vs CPU Golden: bit-exact 一致
  - PyTorch vs DUT: Fuzzy PASS, QSNR >60 dB
  - 带完整断言验证

### ✅ Demo 06: Encoder - bfp8 x bfp4 混合精度（FuzzQ）
- **文件**: `compare_06_encoder_bfp8_fuzzq/run.py`
- **状态**: ✅ 已完成
- **模型**: Encoder (10个算子)
- **精度**: MatMul(bfp8 x bfp4), 其他(bfp8 x bfp8)
- **策略**: FuzzQ（纯随机 + 量化，qa_aware=False）
- **三路执行**:
  - PyTorch Golden: 模拟 bfp8/bfp4 量化计算
  - CPU Golden: 使用 `set_cpu_golden_dtype()` 混合精度
  - DUT: CPU Golden + 噪声 (5e-3)
- **四比数**: Track 1-4
- **特点**: 完整的三路执行 + 逐算子四态判定

### ✅ Demo 07: Encoder - bfp8 x bfp4 混合精度 + QA
- **文件**: `compare_07_encoder_bfp8_qa/run.py`
- **状态**: ✅ 已完成
- **模型**: Encoder (10个算子)
- **精度**: MatMul(bfp8 x bfp4), 其他(bfp8 x bfp8)
- **策略**: QA（qa_aware=True, center=1.0, amplitude=0.02）
- **四比数**: Track 1-4
- **特点**:
  - 展示 Pure vs HW QSNR
  - 证明 QA 对混合精度量化的优化
  - 预期 QSNR 比 Demo 06 提升

### ✅ Demo 08: Transformer - bfp8 x bfp4 混合精度 + QA
- **文件**: `compare_08_transformer_bfp8_qa/run.py`
- **状态**: ✅ 已完成
- **模型**: 小 Transformer (2 layers, 20个算子)
- **精度**: MatMul(bfp8 x bfp4), 其他(bfp8 x bfp8), QA
- **三路执行** + **四比数** + **四态判定**
- **特点**:
  - 大模型场景完整流程
  - 采样分析：每层首尾算子 (L0/L1 各2个)
  - 全局统计：所有 MatMul 的 Pure vs HW QSNR（平均/范围）
  - 最复杂的综合 demo

---

## 技术要点

### 混合精度配置

**MatMul 算子**:
```python
PRECISION_MATMUL = PrecisionConfig(
    input_dtype="bfp8",
    weight_dtype="bfp4",
    compute_dtype="bfp8",
    output_dtype="bfp8",
    qa_aware=True,  # Demo 07-08
)
```

**其他算子** (LayerNorm, GELU, Softmax):
```python
PRECISION_OTHER = PrecisionConfig(
    input_dtype="bfp8",
    weight_dtype="bfp8",
    compute_dtype="bfp8",
    output_dtype="bfp8",
    qa_aware=True,  # Demo 07-08
)
```

**CPU Golden 设置**:
```python
from aidevtools.ops.cpu_golden import set_cpu_golden_dtype

set_cpu_golden_dtype(
    dtype="bfp8",
    dtype_matmul_a="bfp8",
    dtype_matmul_b="bfp4",
    dtype_matmul_out="bfp8",
)
```

### 三路执行

1. **PyTorch Golden**: 使用 PyTorch 模拟量化计算
2. **CPU Golden**: 使用 cpu_golden 后端执行（支持混合精度）
3. **DUT**: CPU Golden 结果 + 后处理（噪声/量化/错误注入）

### 四比数（Four Track Golden）

使用 `DataGenerator.generate_four_track()` 生成：
- **Track 1 (golden_pure)**: 纯 fp32 计算
- **Track 2 (golden_local)**: 本地格式（fp16/int8）量化→反量化
- **Track 3 (golden_hw)**: 硬件格式（bfp8/bfp4）量化→反量化
- **Track 4 (golden_qa)**: 量化感知随机权重

### 四态判定

基于 DUT 比对结果和 Golden 自检的交叉判定：
- **PASS**: DUT 正确，Golden 有效
- **GOLDEN_SUSPECT**: DUT 匹配，但 Golden 可疑
- **DUT_ISSUE**: Golden 有效，DUT 有问题
- **BOTH_SUSPECT**: 都可疑，需人工排查

---

## 运行说明

### 环境要求

```bash
pip install numpy torch
```

### 运行单个 Demo

```bash
python3 demos/compare/compare_01_encoder_pure_vs_fp16/run.py
python3 demos/compare/compare_02_encoder_pure_vs_fuzzq/run.py
python3 demos/compare/compare_03_encoder_fuzzq_vs_qa/run.py
python3 demos/compare/compare_04_matmul_pure_vs_dut/run.py
python3 demos/compare/compare_06_encoder_bfp8_fuzzq/run.py
python3 demos/compare/compare_07_encoder_bfp8_qa/run.py
python3 demos/compare/compare_08_transformer_bfp8_qa/run.py
```

### 批量运行

```bash
for demo in demos/compare/compare_{01,02,03,04,06,07,08}_*/run.py; do
    echo "=== Running $demo ==="
    python3 "$demo" || echo "FAILED: $demo"
done
```

---

## 代码规范

1. **统一架构**: 所有 Demo 遵循相同的三路执行 + 四比数 + 四态判定架构
2. **策略系统**: 使用 `CompareEngine.standard()` + StandardStrategy
3. **标准报告**: 使用 `print_strategy_table()` 统一输出（Phase 1）
4. **断言验证**: 每个 demo 包含完整的结果验证
5. **可复现性**: 使用固定随机种子 (SEED=42)

---

## 开发时间线

- **2026-02-08**: Demo 01-03 实现（No DUT）
- **2026-02-09**: Demo 04, 06-08 实现（With DUT + 混合精度）
- **2026-02-09**: 清理重复目录，移除 Demo 05

---

最后更新: 2026-02-09
