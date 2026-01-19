# 02 统一工作流示例

演示新架构的完整流程：
- 全局配置 (golden_mode, precision, seed)
- 统一 Tensor 格式 (fp32 + quantized)
- 三列比对 (exact, fuzzy_pure, fuzzy_qnt)
- 状态判定 (PERFECT → PASS → QUANT_ISSUE → FAIL)

## 运行

```bash
cd demos/02_unified_workflow
python run.py
```

## 核心概念

### 1. 全局配置

```python
set_config(
    golden_mode="python",  # python | cpp
    precision="quant",     # pure | quant
    seed=42,
)
```

### 2. Tensor 格式

```python
# 同时包含 fp32 和量化后的数据
x = generate_random(shape=(2, 4, 64), qtype="bfp8", seed=42)
print(x.fp32)        # 最高精度
print(x.quantized)   # 量化后数据
```

### 3. 三列比对

| 列 | 说明 |
|----|------|
| exact | 精确比对 (bit-level) |
| fuzzy_pure | vs 纯 fp32 Golden |
| fuzzy_qnt | vs 量化感知 Golden |

### 4. 状态判定

| 状态 | 条件 |
|------|------|
| PERFECT | exact ✓ |
| PASS | fuzzy_qnt ✓ |
| QUANT_ISSUE | fuzzy_pure ✓, fuzzy_qnt ✗ |
| FAIL | 都 ✗ |
