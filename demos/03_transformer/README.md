# 03 Transformer 模型示例

完整 Transformer 模型示例，展示实际项目的组织方式。

## 运行

```bash
cd demos/03_transformer
python run.py
```

## 文件结构

| 文件 | 用途 |
|------|------|
| operators.py | 导入核心库算子 (直接使用 `aidevtools.ops.nn`) |
| model.py | 模型组合 + BFP 量化策略 |
| run.py | 运行入口，导出 golden 数据 |

## 设计说明

- 算子 golden 实现在核心库 `aidevtools.ops.nn` 中统一定义
- demo 只负责模型组合和量化策略，不重复定义算子

## 量化策略

| 操作 | 量化格式 | 说明 |
|------|----------|------|
| matmul/linear | bfp4 | 2-bit mantissa, 极端量化 |
| 其他 | bfp8 | 4-bit mantissa, 保持精度 |

## 模型结构

```
input_ids
    ↓
embedding (bfp8)
    ↓
┌───────────────────────────────────┐
│  Self-Attention Block              │
│  ├─ Q/K/V projection (bfp4)       │
│  ├─ Attention scores (bfp4)       │
│  ├─ Softmax (bfp8)                │
│  └─ Output projection (bfp4)      │
└───────────────────────────────────┘
    ↓ + residual (bfp8)
LayerNorm (bfp8)
    ↓
┌───────────────────────────────────┐
│  FFN Block                         │
│  ├─ Up projection (bfp4)          │
│  ├─ GELU (bfp8)                   │
│  └─ Down projection (bfp4)        │
└───────────────────────────────────┘
    ↓ + residual (bfp8)
LayerNorm (bfp8)
    ↓
output
```

## 输出

运行后会在 `./workspace` 目录生成：
- `*_golden.bin` - Golden 输出 (reference)
- `*_result.bin` - 用户 golden 实现输出 (如已注册)
- `*_input.bin` - 输入数据
- `*_weight.bin` - 权重数据
- `transformer_compare.csv` - 比对配置文件
