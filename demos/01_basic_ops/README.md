# 01 基础算子示例

演示如何使用 `aidevtools.ops.nn` 中的基础算子。

## 运行

```bash
cd demos/01_basic_ops
python run.py
```

## 包含算子

- `linear` - 线性层 y = x @ W + b
- `relu` - ReLU 激活
- `gelu` - GELU 激活
- `softmax` - Softmax
- `layernorm` - Layer Normalization
- `attention` - Scaled Dot-Product Attention
- `embedding` - Embedding 查表

## 输出

运行后会在 `./workspace` 目录生成：
- `*_golden.bin` - Golden 输出
- `*_input.bin` - 输入数据
- `*_weight.bin` - 权重数据
