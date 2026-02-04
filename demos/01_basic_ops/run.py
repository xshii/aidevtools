#!/usr/bin/env python
"""基础算子示例

演示通过 PyTorch 劫持使用 golden 实现。

用法:
    import aidevtools.golden  # 导入即启用劫持

    import torch.nn.functional as F
    y = F.linear(x, w)  # 自动走 golden
"""
import torch

import aidevtools.golden as golden  # 导入即启用劫持
from aidevtools import ops
import torch.nn.functional as F


def main():
    print("=" * 60)
    print("基础算子 Golden 示例 (PyTorch 劫持模式)")
    print("=" * 60)

    # 清空之前的记录
    golden.clear()

    # 1. Linear
    print("\n[1] Linear: y = x @ W.T + b")
    x = torch.randn(2, 4, 64)
    # PyTorch 格式: weight [out_features, in_features]
    w = torch.randn(128, 64)
    b = torch.randn(128)
    y = F.linear(x, w, b)
    print(f"    input: {x.shape}, weight: {w.shape} -> output: {y.shape}")

    # 2. ReLU
    print("\n[2] ReLU: y = max(0, x)")
    x = torch.randn(2, 4, 64)
    y = F.relu(x)
    print(f"    input: {x.shape} -> output: {y.shape}")
    print(f"    负值数量: {(x < 0).sum().item()} -> {(y < 0).sum().item()}")

    # 3. GELU
    print("\n[3] GELU")
    x = torch.randn(2, 4, 64)
    y = F.gelu(x)
    print(f"    input: {x.shape} -> output: {y.shape}")

    # 4. Softmax
    print("\n[4] Softmax")
    x = torch.randn(2, 4, 64)
    y = F.softmax(x, dim=-1)
    print(f"    input: {x.shape} -> output: {y.shape}")
    print(f"    sum(dim=-1): {y.sum(dim=-1)[0, 0].item():.6f} (should be 1.0)")

    # 5. LayerNorm (需要 normalized_shape)
    print("\n[5] LayerNorm")
    x = torch.randn(2, 4, 64)
    y = F.layer_norm(x, normalized_shape=(64,))
    print(f"    input: {x.shape} -> output: {y.shape}")
    print(f"    mean: {y.mean().item():.6f}, std: {y.std().item():.6f}")

    # 6. Attention
    print("\n[6] Scaled Dot-Product Attention")
    q = torch.randn(2, 4, 8, 64)  # [B, H, L, D]
    k = torch.randn(2, 4, 8, 64)
    v = torch.randn(2, 4, 8, 64)
    y = F.scaled_dot_product_attention(q, k, v)
    print(f"    Q: {q.shape}, K: {k.shape}, V: {v.shape} -> output: {y.shape}")

    # 打印报告
    print("\n" + "=" * 60)
    print("Golden 比对报告")
    print("=" * 60)
    golden.report()

    print("\n完成!")


if __name__ == "__main__":
    main()
