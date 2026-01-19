# 05 xlsx Transformer 示例

从 Excel 配置生成 Transformer 模型并运行。

## 文件

| 文件 | 说明 |
|------|------|
| transformer_config.xlsx | Transformer 模型配置 (13 个算子) |
| run.py | 运行示例 |

## 运行

```bash
cd demos/05_xlsx_transformer
python run.py
```

## 工作流程

```
1. 创建 xlsx 模板
        ↓
2. 在 xlsx 中定义 Transformer 算子序列
        ↓
3. 从 xlsx 生成 Python 代码
        ↓
4. 运行并比对结果
```

## 模型结构 (简化版 1 层 Transformer)

```
input_ids
    ↓
embedding (bfp8)
    ↓
Q/K/V projection (bfp4)
    ↓
attention (bfp8)
    ↓
O projection (bfp4)
    ↓ + residual
LayerNorm (bfp8)
    ↓
FFN_up (bfp4) → GELU (bfp8) → FFN_down (bfp4)
    ↓ + residual
LayerNorm (bfp8)
    ↓
output
```

## xlsx 算子配置

| ID | 算子 | 量化 | 说明 |
|----|------|------|------|
| 0 | embedding | bfp8 | Token Embedding |
| 1-3 | linear | bfp4 | Q/K/V projection |
| 4 | attention | bfp8 | Attention |
| 5 | linear | bfp4 | O projection |
| 6 | add | bfp8 | Residual 1 |
| 7 | layernorm | bfp8 | LayerNorm 1 |
| 8 | linear | bfp4 | FFN up |
| 9 | gelu | bfp8 | GELU |
| 10 | linear | bfp4 | FFN down |
| 11 | add | bfp8 | Residual 2 |
| 12 | layernorm | bfp8 | Output LayerNorm |

## 量化策略

| 操作 | 量化格式 | 说明 |
|------|----------|------|
| linear (matmul) | bfp4 | 2-bit mantissa, 极端量化 |
| 其他 | bfp8 | 4-bit mantissa, 保持精度 |

## 输出

运行后在 workspace 目录生成：
- `transformer_config.xlsx` - Transformer 配置文件
- `generated_transformer.py` - 生成的 Python 代码
- `*.bin` - Golden 数据

## 依赖

```bash
pip install openpyxl
```
