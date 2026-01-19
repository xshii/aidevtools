#!/usr/bin/env python
"""Transformer Golden 数据生成示例

使用方法:
    cd demos/03_transformer
    python run.py
"""
import numpy as np
import sys
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aidevtools.trace import trace, dump, clear

# 导入本地算子和模型
from operators import (
    linear, relu, gelu, softmax_safe, layernorm, attention, embedding
)
from model import (
    TransformerConfig, init_weights,
    self_attention_block, ffn_block, transformer_layer, transformer_forward
)


def run_single_operators():
    """运行单算子测试"""
    print("=" * 50)
    print("单算子 Golden 计算")
    print("=" * 50)

    # Linear: y = ax + b
    print("\n[1] Linear (ax+b)")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    w = np.random.randn(64, 128).astype(np.float32)
    b = np.random.randn(128).astype(np.float32)
    y = linear(x, w, b)
    print(f"    input: {x.shape} -> output: {y.shape}")

    # ReLU
    print("\n[2] ReLU")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    y = relu(x)
    print(f"    input: {x.shape} -> output: {y.shape}")

    # GELU
    print("\n[3] GELU")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    y = gelu(x)
    print(f"    input: {x.shape} -> output: {y.shape}")

    # Softmax Safe
    print("\n[4] Softmax Safe")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    y = softmax_safe(x)
    print(f"    input: {x.shape} -> output: {y.shape}")
    print(f"    sum(axis=-1): {y.sum(axis=-1)[0, 0]:.6f} (should be 1.0)")

    # LayerNorm
    print("\n[5] LayerNorm")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    gamma = np.ones(64, dtype=np.float32)
    beta = np.zeros(64, dtype=np.float32)
    y = layernorm(x, gamma, beta)
    print(f"    input: {x.shape} -> output: {y.shape}")
    print(f"    mean: {y.mean():.6f}, std: {y.std():.6f}")

    # Attention
    print("\n[6] Attention")
    q = np.random.randn(2, 8, 64).astype(np.float32)
    k = np.random.randn(2, 8, 64).astype(np.float32)
    v = np.random.randn(2, 8, 64).astype(np.float32)
    y = attention(q, k, v)
    print(f"    Q: {q.shape}, K: {k.shape}, V: {v.shape} -> output: {y.shape}")

    # Embedding
    print("\n[7] Embedding")
    input_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    embed_table = np.random.randn(100, 64).astype(np.float32)
    y = embedding(input_ids, embed_table)
    print(f"    input_ids: {input_ids.shape} -> output: {y.shape}")


def run_transformer_model():
    """运行 Transformer 模型测试"""
    print("\n" + "=" * 50)
    print("Transformer 模型 Golden 计算")
    print("=" * 50)

    # 小型配置用于测试
    config = TransformerConfig(
        vocab_size=1000,
        hidden_size=64,
        num_heads=4,
        intermediate_size=256,
        num_layers=2,
        max_seq_len=32,
    )

    print(f"\n配置:")
    print(f"    vocab_size: {config.vocab_size}")
    print(f"    hidden_size: {config.hidden_size}")
    print(f"    num_heads: {config.num_heads}")
    print(f"    intermediate_size: {config.intermediate_size}")
    print(f"    num_layers: {config.num_layers}")

    # 初始化权重
    weights = init_weights(config)
    print(f"\n权重初始化完成")

    # 输入数据
    batch_size = 2
    seq_len = 16
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"\n输入: input_ids {input_ids.shape}")

    # 前向传播
    output = transformer_forward(input_ids, weights, config)
    print(f"输出: {output.shape}")
    print(f"输出均值: {output.mean():.6f}, 标准差: {output.std():.6f}")


def main():
    # 清空之前的记录
    clear()

    # 运行单算子
    run_single_operators()

    # 运行模型
    run_transformer_model()

    # 导出
    output_dir = Path(__file__).parent / "workspace"
    print("\n" + "=" * 50)
    print("导出 Golden 数据")
    print("=" * 50)

    dump(str(output_dir), format="raw")

    print(f"\n完成! 下一步:")
    print(f"  1. 运行仿真器生成 result 数据")
    print(f"  2. 运行比数: aidev compare {output_dir}")


if __name__ == "__main__":
    main()
