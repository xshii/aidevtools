#!/usr/bin/env python
"""基础算子示例

演示 PyTorch 风格的 functional API。

用法:
    from aidevtools import F
    y = F.linear(x, weight, bias)
    y = F.relu(y)
"""
import numpy as np
import sys
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aidevtools import F, ops
from aidevtools.ops import get_records


def main():
    print("=" * 60)
    print("基础算子 Golden 示例 (PyTorch 风格 API)")
    print("=" * 60)

    # 清空之前的记录
    ops.clear()

    # 1. Linear
    print("\n[1] Linear: y = x @ W.T + b")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    # PyTorch 格式: weight [out_features, in_features]
    w = np.random.randn(128, 64).astype(np.float32)
    b = np.random.randn(128).astype(np.float32)
    y = F.linear(x, w, b)
    print(f"    input: {x.shape}, weight: {w.shape} -> output: {y.shape}")

    # 2. ReLU
    print("\n[2] ReLU: y = max(0, x)")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    y = F.relu(x)
    print(f"    input: {x.shape} -> output: {y.shape}")
    print(f"    负值数量: {np.sum(x < 0)} -> {np.sum(y < 0)}")

    # 3. GELU
    print("\n[3] GELU")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    y = F.gelu(x)
    print(f"    input: {x.shape} -> output: {y.shape}")

    # 4. Softmax
    print("\n[4] Softmax")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    y = F.softmax(x, dim=-1)
    print(f"    input: {x.shape} -> output: {y.shape}")
    print(f"    sum(dim=-1): {y.sum(axis=-1)[0, 0]:.6f} (should be 1.0)")

    # 5. LayerNorm
    print("\n[5] LayerNorm")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    y = F.layer_norm(x, normalized_shape=(64,))
    print(f"    input: {x.shape} -> output: {y.shape}")
    print(f"    mean: {y.mean():.6f}, std: {y.std():.6f}")

    # 6. Attention
    print("\n[6] Scaled Dot-Product Attention")
    q = np.random.randn(2, 4, 8, 64).astype(np.float32)  # [B, H, L, D]
    k = np.random.randn(2, 4, 8, 64).astype(np.float32)
    v = np.random.randn(2, 4, 8, 64).astype(np.float32)
    y = F.scaled_dot_product_attention(q, k, v)
    print(f"    Q: {q.shape}, K: {k.shape}, V: {v.shape} -> output: {y.shape}")

    # 7. Embedding
    print("\n[7] Embedding")
    input_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    embed_table = np.random.randn(100, 64).astype(np.float32)
    y = F.embedding(input_ids, embed_table)
    print(f"    input_ids: {input_ids.shape}, table: {embed_table.shape} -> output: {y.shape}")

    # 导出
    print("\n" + "=" * 60)
    print("导出 Golden 数据")
    print("=" * 60)

    output_dir = Path(__file__).parent / "workspace"
    ops.dump(str(output_dir), format="raw")

    records = get_records()
    print(f"\n共记录 {len(records)} 个算子:")
    for r in records:
        print(f"  - {r['name']}")

    print(f"\n输出目录: {output_dir}")
    print("完成!")


if __name__ == "__main__":
    main()
