# Compare Demo 完整实现

## 概览

本目录包含完整的比对系统 demo，展示从单算子到完整 Transformer 的**三路执行 + 四比数 + 四态判定**流程。

**实现状态**: 所有 7 个 Demo 已完成 ✅
- Phase 1 (No DUT): Demo 01-03
- Phase 2 (With DUT): Demo 04, 06-08

**Demo 05 已移除**（不需要单独的错误注入 demo）

## 核心概念

### 三路执行
1. **PyTorch Golden**: 使用 PyTorch 模拟量化计算
2. **CPU Golden**: 使用 cpu_golden 后端执行（支持混合精度）
3. **DUT**: CPU Golden 结果 + 后处理（噪声/量化/错误注入）

### 四比数（Four Track Golden）
使用 `generate_four_track()` 生成四种 golden：
- **Track 1 (golden_pure)**: 纯 fp32 计算
- **Track 2 (golden_local)**: 本地格式（fp16/int8）量化→反量化
- **Track 3 (golden_hw)**: 硬件格式（bfp8/bfp4）量化→反量化
- **Track 4 (golden_qa)**: 量化感知随机权重

### 四态判定
基于 DUT 比对结果和 Golden 自检结果的交叉判定：

| DUT vs Golden | Golden 自检 | 判定状态 |
|:---:|:---:|------|
| PASS | PASS | **PASS** - DUT 正确，Golden 有效 |
| PASS | FAIL | **GOLDEN_SUSPECT** - DUT 匹配，但 Golden 可疑 |
| FAIL | PASS | **DUT_ISSUE** - Golden 有效，DUT 有问题 |
| FAIL | FAIL | **BOTH_SUSPECT** - 都可疑，需人工排查 |

## Demo 列表

### Phase 1: No DUT（验证数据生成策略）

#### ✅ Demo 01: Encoder - Pure vs FP16
- **精度**: Pure fp32 vs fp16 计算
- **目标**: 测试 fp16 计算引入的精度损失
- **结果**: QSNR ~67 dB, Cosine ~1.0

#### ✅ Demo 02: Encoder - Pure vs FuzzQ
- **精度**: Pure fp32 vs bfp8 量化反量化
- **目标**: 测试量化反量化（FuzzQ）引入的数据误差
- **结果**: QSNR ~17 dB (bfp8 低精度格式)

#### ✅ Demo 03: Encoder - QA Pure vs QA FuzzQ
- **精度**: QA 数据（amplitude=0.02）vs bfp8 量化
- **目标**: 测试 QA 数据对 bfp8 量化的适应性
- **结果**: QSNR 32.1 dB（比 Pure 提升 15.1 dB）
- **优化**: amplitude 0.5→0.2→0.1→0.05→**0.02**

### Phase 2: With DUT（三路执行 + 四比数 + 四态判定）

#### ✅ Demo 04: MatMul - Pure vs DUT
- **文件**: `compare_04_matmul_pure_vs_dut/run.py`
- **模型**: 单个 MatMul 算子 (64×64 @ 64×128)
- **精度**: Pure fp32
- **三路执行**: PyTorch Golden, CPU Golden, DUT
- **四比数**: Track 1-4 (Pure, Local, HW, QA)
- **验证**:
  - PyTorch vs CPU Golden: bit-exact 一致
  - PyTorch vs DUT: Fuzzy PASS, QSNR >60 dB
  - Golden 数据无异常
- **关键特性**: 完整的三路执行流程示例，带断言验证

#### ✅ Demo 06: Encoder - bfp8 x bfp4 混合精度（FuzzQ）
- **文件**: `compare_06_encoder_bfp8_fuzzq/run.py`
- **模型**: Encoder (10个算子，hidden=64, ffn=256)
- **精度**: MatMul(bfp8 x bfp4), 其他(bfp8 x bfp8)
- **策略**: FuzzQ（纯随机 + 量化，qa_aware=False）
- **特点**:
  - 使用 cpu_golden 混合精度支持 (`set_cpu_golden_dtype`)
  - 三路执行: PyTorch Golden + CPU Golden + DUT
  - 逐算子四态判定结果输出
  - 完整的四比数生成流程

#### ✅ Demo 07: Encoder - bfp8 x bfp4 混合精度 + QA
- **文件**: `compare_07_encoder_bfp8_qa/run.py`
- **模型**: Encoder (10个算子)
- **精度**: MatMul(bfp8 x bfp4), 其他(bfp8 x bfp8)
- **策略**: QA（qa_aware=True, qa_center=1.0, qa_amplitude=0.02）
- **特点**:
  - QA 策略优化量化误差
  - 对比展示 Pure vs HW QSNR
  - 证明 QA 对 bfp8/bfp4 量化的优化效果
  - 简化实现，聚焦四比数生成

#### ✅ Demo 08: Transformer - bfp8 x bfp4 混合精度 + QA 四态判定
- **文件**: `compare_08_transformer_bfp8_qa/run.py`
- **模型**: 小 Transformer (2 layers, 20个算子)
- **精度**: MatMul(bfp8 x bfp4), 其他(bfp8 x bfp8), QA
- **特点**:
  - 大模型场景完整流程
  - 采样分析：每层首尾算子 (L0/L1 首尾各1个)
  - 全局统计：所有 MatMul 的 Pure vs HW QSNR 平均/范围
  - 完整的四比数 + 四态判定流程

## 混合精度配置

### MatMul 算子
```python
PRECISION_MATMUL = PrecisionConfig(
    input_dtype="bfp8",
    weight_dtype="bfp4",
    compute_dtype="bfp8",
    output_dtype="bfp8",
    qa_aware=True,  # Demo 07-08
)
```

### 其他算子（LayerNorm, GELU, Softmax）
```python
PRECISION_OTHER = PrecisionConfig(
    input_dtype="bfp8",
    weight_dtype="bfp8",
    compute_dtype="bfp8",
    output_dtype="bfp8",
    qa_aware=True,  # Demo 07-08
)
```

### CPU Golden 混合精度设置
```python
from aidevtools.ops.cpu_golden import set_cpu_golden_dtype

set_cpu_golden_dtype(
    dtype="bfp8",
    dtype_matmul_a="bfp8",
    dtype_matmul_b="bfp4",
    dtype_matmul_out="bfp8",
)
```

## 目录结构

```
demos/compare/
├── README.md                              # 本文档
├── todo.md                                # 实现状态文档
├── compare_01_encoder_pure_vs_fp16/       # Phase 1: fp32 vs fp16
├── compare_02_encoder_pure_vs_fuzzq/      # Phase 1: Pure vs FuzzQ
├── compare_03_encoder_fuzzq_vs_qa/        # Phase 1: FuzzQ vs QA
├── compare_04_matmul_pure_vs_dut/         # Phase 2: 单算子三路执行
├── compare_06_encoder_bfp8_fuzzq/         # Phase 2: 混合精度 FuzzQ
├── compare_07_encoder_bfp8_qa/            # Phase 2: 混合精度 QA
└── compare_08_transformer_bfp8_qa/        # Phase 2: Transformer 综合
```

## 环境依赖

运行 demos 需要安装以下依赖：

```bash
pip install numpy torch
```

## 运行 Demo

```bash
# Phase 1: No DUT（验证数据生成策略）
python3 demos/compare/compare_01_encoder_pure_vs_fp16/run.py
python3 demos/compare/compare_02_encoder_pure_vs_fuzzq/run.py
python3 demos/compare/compare_03_encoder_fuzzq_vs_qa/run.py

# Phase 2: With DUT（三路执行 + 四比数 + 四态判定）
python3 demos/compare/compare_04_matmul_pure_vs_dut/run.py
python3 demos/compare/compare_06_encoder_bfp8_fuzzq/run.py
python3 demos/compare/compare_07_encoder_bfp8_qa/run.py
python3 demos/compare/compare_08_transformer_bfp8_qa/run.py
```

批量运行所有 demos：
```bash
# 使用批量脚本（推荐）
cd demos/compare && ./run_all.sh

# 或手动运行
for demo in demos/compare/compare_{01,02,03,04,06,07,08}_*/run.py; do
    echo "=== Running $demo ==="
    python3 "$demo" || echo "FAILED: $demo"
done
```

## 运行结果

所有 7 个 demos 已通过测试（2026-02-09）：

**Phase 1 (No DUT):**
- ✅ Demo 01: QSNR ~67 dB, Cosine ~1.0 (fp16 精度损失很小)
- ✅ Demo 02: QSNR ~17 dB, Cosine ~0.987 (bfp8 量化误差明显)
- ✅ Demo 03: QSNR ~32 dB, Cosine ~0.9999 (QA 策略提升 15 dB)

**Phase 2 (With DUT):**
- ✅ Demo 04: 三路执行验证成功
- ✅ Demo 06: 混合精度 + FuzzQ 成功
- ✅ Demo 07: 混合精度 + QA 成功
- ✅ Demo 08: Transformer 大模型场景成功

**关键发现：**
- QA 策略显著优于 Pure 数据（32 dB vs 17 dB）
- 混合精度配置（bfp8 x bfp4）成功生效
- 三路执行流程完整可用

## 关键技术点

1. **cpu_golden 混合精度支持**: 通过 `set_cpu_golden_dtype()` 配置 MatMul 的 input/weight/output 精度
2. **四比数生成**: 使用 `DataGenerator.generate_four_track()` 一键生成四种 golden
3. **四态判定**: 使用 `CompareEngine.standard()` 进行完整比对
4. **QA 策略**: 通过受控动态范围（amplitude=0.02）优化量化误差

## 设计原则

1. **统一架构**: 所有 Demo 遵循相同的三路执行 + 四比数 + 四态判定架构
2. **混合精度**: MatMul 使用 bfp8 x bfp4，其他算子使用 bfp8 x bfp8
3. **策略系统**: 使用 CompareEngine + Strategy 系统，不手动调用 compare()
4. **标准报告**: 使用 print_strategy_table() 统一输出格式
5. **断言验证**: 每个 demo 包含完整的结果验证

---

最后更新: 2026-02-09
