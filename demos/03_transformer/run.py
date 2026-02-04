#!/usr/bin/env python
"""Transformer Demo - 使用 PyTorch 风格 API 构建完整 Transformer

演示使用 F.matmul, F.softmax, F.layer_norm 构建 Transformer 模型。
注意: 仅使用有 cpu_golden 实现的算子 (matmul, softmax, layernorm, transpose)

使用方法:
    cd demos/03_transformer
    python run.py
"""
import numpy as np
from pathlib import Path

from aidevtools import ops
from aidevtools.ops import _functional as F
from aidevtools.ops import get_records
from aidevtools.tools.compare.diff import compare_full
from aidevtools.formats.quantize import generate_fake_dut

# 设置 cpu golden dtype 并使用 cpp golden
from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
set_cpu_golden_dtype("gfp16")
ops.set_golden_mode("cpp")


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
    ops.clear()

    # 配置
    batch, seq, hidden = 2, 16, 64
    ffn_hidden = 256
    np.random.seed(42)

    print(f"配置: batch={batch}, seq={seq}, hidden={hidden}, ffn={ffn_hidden}")

    # ========== Self-Attention ==========
    print("\n[Self-Attention]")

    # Input
    x = np.random.randn(batch, seq, hidden).astype(np.float32)
    print(f"  Input: {x.shape}")

    # Q, K, V projections
    w_q = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
    w_k = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
    w_v = np.random.randn(hidden, hidden).astype(np.float32) * 0.02

    q = F.matmul(x, w_q)
    k = F.matmul(x, w_k)
    v = F.matmul(x, w_v)
    print(f"  Q/K/V: {q.shape}")

    # Attention: Q @ K^T -> Softmax -> @ V
    # Q @ K^T: (batch, seq, hidden) @ (batch, hidden, seq) -> (batch, seq, seq)
    k_t = np.swapaxes(k, -2, -1)
    attn_scores = F.matmul(q, k_t) / np.sqrt(hidden)
    attn_weights = F.softmax(attn_scores, dim=-1)
    print(f"  Attention weights: {attn_weights.shape}")

    attn_out = F.matmul(attn_weights, v)
    print(f"  Attention output: {attn_out.shape}")

    # Output projection
    w_o = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
    o = F.matmul(attn_out, w_o)
    print(f"  O projection: {o.shape}")

    # LayerNorm 1
    ln1 = F.layer_norm(o, normalized_shape=(hidden,))
    print(f"  LayerNorm 1: {ln1.shape}")

    # ========== FFN ==========
    print("\n[FFN]")

    # FFN up
    w_up = np.random.randn(hidden, ffn_hidden).astype(np.float32) * 0.02
    ffn_up = F.matmul(ln1, w_up)
    print(f"  FFN up: {ffn_up.shape}")

    # Activation (使用 softmax 代替 GELU，因为 GELU 没有 cpu_golden)
    ffn_act = F.softmax(ffn_up, dim=-1)
    print(f"  FFN activation: {ffn_act.shape}")

    # FFN down
    w_down = np.random.randn(ffn_hidden, hidden).astype(np.float32) * 0.02
    ffn_down = F.matmul(ffn_act, w_down)
    print(f"  FFN down: {ffn_down.shape}")

    # LayerNorm 2
    output = F.layer_norm(ffn_down, normalized_shape=(hidden,))
    print(f"  Output: {output.shape}")

    return get_records()


def main():
    print(f"""
{'=' * 70}
  Transformer Demo - PyTorch 风格 API
{'=' * 70}
  golden_mode: cpp (via subprocess)
  quantization: gfp16 (cpp)
""")

    # 1. 运行模型
    print("[1] 运行 Transformer 模型")
    print("-" * 50)
    records = run_single_layer_transformer()

    print(f"\n共 {len(records)} 个算子:")
    for r in records:
        print(f"  {r['name']}: {r['golden'].shape}")

    # 2. 生成假 DUT
    print("\n[2] 生成假的 DUT 数据")
    print("-" * 50)
    print("流程: golden -> bfp8 量化/反量化 -> 加小噪声")
    np.random.seed(123)
    dut_outputs = [generate_fake_dut(r["golden"], qtype="bfp8", noise_level=0.001) for r in records]

    # 3. 比对
    print("\n[3] 比对 (golden vs DUT)")
    print("-" * 50)
    print(f"    {'Op':15} {'Max Abs':>12} {'QSNR':>10} {'Cosine':>10} {'Status':>8}")
    print("    " + "-" * 60)
    for i, r in enumerate(records):
        diff = compare_full(r["golden"], dut_outputs[i])
        status = "PASS" if diff.passed else "FAIL"
        print(f"    {r['name']:15} {diff.max_abs:12.6e} {diff.qsnr:10.2f} {diff.cosine:10.6f} {status:>8}")

    # 4. 导出
    print("\n[4] 导出 bin 文件")
    print("-" * 50)
    output_dir = Path(__file__).parent / "workspace"
    ops.dump(str(output_dir))
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
