# 03 Transformer 模型示例

使用 ops API 构建完整 Transformer 模型，演示：
- 使用 `ops.matmul`, `ops.softmax`, `ops.layernorm` 等
- 自动生成输入和权重
- 使用 cpp golden (via subprocess)
- 三列比对：exact / fuzzy_pure / fuzzy_qnt

## 运行

```bash
cd demos/03_transformer
python run.py
```

## 文件结构

| 文件 | 用途 |
|------|------|
| run.py | 运行入口，使用 ops API 构建模型 |

## 设计说明

- 使用新的 ops API，自动处理：
  - 输入/权重生成
  - bfp 量化
  - cpp golden 计算
  - reference (fp64) 计算
- 三列比对：exact / fuzzy_pure / fuzzy_qnt

## 量化配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| cpp golden | gfp16 | 16-bit gfloat 格式 |
| 输入量化 | bfp8 | 4-bit mantissa |

## 模型结构 (简化版单层 Transformer)

```
Input Linear
    ↓
┌───────────────────────────────────┐
│  Self-Attention Block              │
│  ├─ Q/K/V MatMul                  │
│  ├─ Attention Scores MatMul       │
│  ├─ Softmax                       │
│  ├─ Attention Output MatMul       │
│  └─ O MatMul                      │
└───────────────────────────────────┘
    ↓
LayerNorm
    ↓
┌───────────────────────────────────┐
│  FFN Block                         │
│  ├─ FFN Up MatMul                 │
│  ├─ Softmax (代替 GELU)           │
│  └─ FFN Down MatMul               │
└───────────────────────────────────┘
    ↓
LayerNorm
    ↓
Output
```

## 输出

运行后会在 `./workspace` 目录生成：
- `*_golden.bin` - cpp golden 输出
- `*_reference.bin` - fp64 高精度参考
- `*_input.bin` - 输入数据
- `*_weight.bin` - 权重数据
