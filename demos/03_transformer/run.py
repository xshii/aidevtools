#!/usr/bin/env python
"""Transformer Demo - 使用 ops API 构建完整 Transformer

演示使用新的 ops API 构建 Transformer 模型：
1. 使用 ops.matmul, ops.softmax, ops.layernorm 等
2. 自动生成输入和权重
3. 使用 cpp golden (via subprocess)
4. 三列比对：exact / fuzzy_pure / fuzzy_qnt

使用方法:
    cd demos/03_transformer
    python run.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from aidevtools import ops
from aidevtools.ops.base import get_records, set_golden_mode, clear
from aidevtools.tools.compare.diff import compare_3col, print_compare_table
from aidevtools.formats.quantize import generate_fake_dut

# 设置 cpu golden dtype 并使用 cpp golden
from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
set_cpu_golden_dtype("gfp16")
set_golden_mode("cpp")


def run_single_layer_transformer():
    """
    运行单层 Transformer (简化版)

    结构:
        Input -> MatMul(Q) -> MatMul(K) -> MatMul(V)
              -> Transpose(K) -> MatMul(QK) -> Softmax -> MatMul(V)
              -> MatMul(O) -> LayerNorm
              -> MatMul(FFN_up) -> Softmax -> MatMul(FFN_down) -> LayerNorm
    """
    ops.seed(42)
    clear()

    # 配置
    batch, seq, hidden = 2, 16, 64
    ffn_hidden = 256
    dtype = "bfp8"

    print(f"配置: batch={batch}, seq={seq}, hidden={hidden}, ffn={ffn_hidden}")
    print(f"量化: {dtype} (input) + gfp16 (cpp golden)")

    # ========== Self-Attention ==========
    print("\n[Self-Attention]")

    # Input projection (用 matmul 代替 linear，因为 linear 没有 cpp golden)
    x = ops.matmul((batch, seq, hidden), (hidden, hidden), dtype=dtype)  # Input
    print(f"  Input: {x.shape}")

    # Q, K, V projections
    q = ops.matmul(x, (hidden, hidden), dtype=dtype)  # Q
    k = ops.matmul(x, (hidden, hidden), dtype=dtype)  # K
    v = ops.matmul(x, (hidden, hidden), dtype=dtype)  # V
    print(f"  Q/K/V: {q.shape}")

    # Attention: Q @ K^T -> Softmax -> @ V
    # 为了简化，这里直接用 matmul 模拟 attention
    # 实际 attention 需要 transpose K，这里用 (seq, seq) 模拟 attention scores
    attn_scores = ops.matmul(q, (hidden, seq), dtype=dtype)  # 模拟 Q @ K^T
    attn_weights = ops.softmax(attn_scores, dtype=dtype)
    print(f"  Attention weights: {attn_weights.shape}")

    attn_out = ops.matmul(attn_weights, (seq, hidden), dtype=dtype)  # 模拟 @ V
    print(f"  Attention output: {attn_out.shape}")

    # Output projection
    o = ops.matmul(attn_out, (hidden, hidden), dtype=dtype)
    print(f"  O projection: {o.shape}")

    # LayerNorm 1
    ln1 = ops.layernorm(o, dtype=dtype)  # gamma/beta 自动生成
    print(f"  LayerNorm 1: {ln1.shape}")

    # ========== FFN ==========
    print("\n[FFN]")

    # FFN up
    ffn_up = ops.matmul(ln1, (hidden, ffn_hidden), dtype=dtype)
    print(f"  FFN up: {ffn_up.shape}")

    # Activation (用 softmax 代替 GELU，因为 GELU 没有 cpp golden)
    ffn_act = ops.softmax(ffn_up, dtype=dtype)
    print(f"  FFN activation: {ffn_act.shape}")

    # FFN down
    ffn_down = ops.matmul(ffn_act, (ffn_hidden, hidden), dtype=dtype)
    print(f"  FFN down: {ffn_down.shape}")

    # LayerNorm 2
    output = ops.layernorm(ffn_down, dtype=dtype)  # gamma/beta 自动生成
    print(f"  Output: {output.shape}")

    return get_records()


def main():
    print(f"""
{'=' * 70}
  Transformer Demo - 使用 ops API
{'=' * 70}
  golden_mode: cpp (via subprocess)
  quantization: gfp16 (cpp) + bfp8 (input)
""")

    # 1. 运行模型
    print("[1] 运行 Transformer 模型")
    print("-" * 50)
    records = run_model()

    print(f"\n共 {len(records)} 个算子:")
    for r in records:
        print(f"  {r['name']}: {r['golden'].shape}")

    # 2. 生成假 DUT
    print(f"\n[2] 生成假的 DUT 数据")
    print("-" * 50)
    print("流程: reference → bfp8 量化/反量化 → 加小噪声")
    np.random.seed(123)
    dut_outputs = [generate_fake_dut(r["reference"], qtype="bfp8", noise_level=0.001) for r in records]

    # 3. 比对
    print(f"\n[3] 三列比对")
    print("-" * 50)
    results = []
    for i, r in enumerate(records):
        result = compare_3col(
            op_name=r["op"],
            op_id=i,
            result=dut_outputs[i],
            golden_pure=r["reference"],
            golden_qnt=r["golden"],
        )
        results.append(result)

    print_compare_table(results)

    # 4. 导出
    print(f"\n[4] 导出 bin 文件")
    print("-" * 50)
    output_dir = Path(__file__).parent / "workspace"
    ops.dump(str(output_dir))
    print(f"输出目录: {output_dir}")


def run_model():
    """运行模型的包装函数"""
    return run_single_layer_transformer()


if __name__ == "__main__":
    main()
