#!/usr/bin/env python3
"""
Transformer 模型时延分析 Demo

演示如何使用 PyTorch 风格 F API 定义模型，自动生成 OpProfile 进行 Paper Analysis。

有 cpu_golden 实现的算子:
  - 矩阵运算: matmul, linear
  - 归一化: layernorm, softmax
  - 激活函数: relu, gelu, sigmoid, tanh, silu
  - 逐元素: add, mul, div
  - 其他: transpose

使用方式:
1. 调用 F.matmul, F.layer_norm, F.softmax, F.gelu 等有 cpu_golden 的算子
2. 调用 ops.get_profiles() 获取自动生成的 profiles
3. 使用 PaperAnalyzer 分析时延

Usage:
    python demos/08_paper_analysis/main.py
"""

import numpy as np
from pathlib import Path

from aidevtools import ops
from aidevtools.ops import _functional as F
from aidevtools.analysis import (
    PaperAnalyzer,
    PassConfig,
    PassPreset,
    export_xlsx,
    export_csv,
    export_json,
    load_chip_spec,
    list_chips,
)


def transformer_layer_ops(
    batch: int = 4,
    seq: int = 512,
    hidden: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = 3072,
):
    """
    使用 PyTorch 风格 F API 定义 Transformer Layer

    调用完成后，可通过 ops.get_profiles() 获取自动生成的 profiles
    """
    head_dim = hidden // num_heads

    # 创建输入张量
    x = np.random.randn(batch, seq, hidden).astype(np.float32)

    # ============================================================
    # Self-Attention Block
    # ============================================================

    # LayerNorm
    x_norm = F.layer_norm(x, normalized_shape=(hidden,))

    # Q/K/V 投影
    w_q = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
    w_k = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
    w_v = np.random.randn(hidden, hidden).astype(np.float32) * 0.02

    q = F.matmul(x_norm, w_q)
    k = F.matmul(x_norm, w_k)
    v = F.matmul(x_norm, w_v)

    # Reshape for multi-head attention
    q = q.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Attention (分解为 matmul + softmax，因为 attention 没有 cpu_golden)
    # Q @ K^T / sqrt(d_k)
    k_t = np.swapaxes(k, -2, -1)
    attn_scores = F.matmul(q, k_t) / np.sqrt(head_dim)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_out = F.matmul(attn_weights, v)

    # Reshape back
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, hidden)

    # Output projection
    w_o = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
    attn_out = F.matmul(attn_out, w_o)

    # Residual
    x = x + attn_out

    # ============================================================
    # FFN Block
    # ============================================================

    # LayerNorm
    x_norm = F.layer_norm(x, normalized_shape=(hidden,))

    # FFN1: hidden -> ffn_hidden
    w_ffn1 = np.random.randn(hidden, ffn_hidden).astype(np.float32) * 0.02
    h = F.matmul(x_norm, w_ffn1)

    # GELU Activation
    h = F.gelu(h)

    # FFN2: ffn_hidden -> hidden
    w_ffn2 = np.random.randn(ffn_hidden, hidden).astype(np.float32) * 0.02
    h = F.matmul(h, w_ffn2)

    # Residual
    x = x + h

    return x


def print_profiles_info(profiles: list):
    """打印 profile 信息"""
    print("\n" + "=" * 80)
    print("Transformer Layer Operator Profiles (Auto-generated from F API)")
    print("=" * 80)

    total_flops = 0
    total_bytes = 0

    for p in profiles:
        total_flops += p.flops
        total_bytes += p.total_bytes
        print(f"{p.name:20s} | {p.op_type:12s} | {p.compute_unit:6s} | "
              f"FLOPs: {p.flops/1e9:8.2f}G | Bytes: {p.total_bytes/1e6:8.2f}MB | "
              f"AI: {p.arithmetic_intensity:6.1f}")

    print("-" * 80)
    print(f"{'Total':20s} | {'':12s} | {'':6s} | "
          f"FLOPs: {total_flops/1e9:8.2f}G | Bytes: {total_bytes/1e6:8.2f}MB")
    print("=" * 80 + "\n")


def main():
    print("\n" + "=" * 80)
    print("Transformer Model Paper Analysis Demo")
    print("Using PyTorch-style F API with auto profile generation")
    print("=" * 80)

    # 显示可用芯片
    print(f"\nAvailable chips: {list_chips()}")

    # 加载芯片规格
    chip = load_chip_spec("npu_910")
    print(f"\nChip: {chip.name}")
    print(f"  Cube FP16: {chip.cube.fp16_tflops} TFLOPS")
    print(f"  Vector FP16: {chip.vector.fp16_gflops} GFLOPS")
    print(f"  HBM Bandwidth: {chip.memory.hbm.bandwidth_gbps} GB/s")

    # 模型配置
    model_config = {
        "batch": 4,
        "seq": 512,
        "hidden": 768,
        "num_heads": 12,
        "ffn_hidden": 3072,
    }

    print("\nModel Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")

    # ============================================================
    # 使用 F API 定义模型 (自动生成 profiles)
    # ============================================================

    print("\nDefining transformer layer using F API...")
    ops.clear()  # 清空之前的记录和 profiles
    np.random.seed(42)

    # 调用 F API 定义模型
    output = transformer_layer_ops(**model_config)
    print(f"Output shape: {output.shape}")

    # 获取自动生成的 profiles
    profiles = ops.get_profiles()
    print(f"\nAuto-generated {len(profiles)} profiles from F API calls")

    # 打印 profile 信息
    print_profiles_info(profiles)

    # ============================================================
    # Paper Analysis
    # ============================================================

    print("\nRunning Paper Analysis...")

    # 使用标准优化配置
    pass_config = PassConfig.from_preset(PassPreset.STANDARD)
    print(f"Pass Configuration: {pass_config.preset.value}")

    # 创建分析器
    analyzer = PaperAnalyzer(chip="npu_910", pass_config=pass_config)

    # 添加 profiles (来自 ops.get_profiles())
    analyzer.add_profiles(profiles)

    # 执行分析
    result = analyzer.analyze()

    # 打印摘要
    analyzer.print_summary()

    # ============================================================
    # 详细结果
    # ============================================================

    print("\n" + "=" * 80)
    print("Detailed Operator Latency Breakdown")
    print("=" * 80)
    print(f"{'Op Name':20s} | {'Unit':6s} | {'Compute':>10s} | {'Memory':>10s} | "
          f"{'Roofline':>10s} | {'Total':>10s} | {'Bottleneck':10s}")
    print("-" * 80)

    for bd in result.breakdowns:
        print(f"{bd.profile.name:20s} | {bd.profile.compute_unit:6s} | "
              f"{bd.timing.compute_us:10.2f} | {bd.timing.memory_us:10.2f} | "
              f"{bd.timing.roofline_us:10.2f} | {bd.timing.total_us:10.2f} | "
              f"{bd.bottleneck:10s}")

    print("=" * 80)

    # ============================================================
    # 导出
    # ============================================================

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    xlsx_path = output_dir / "transformer_analysis_npu910.xlsx"
    csv_path = output_dir / "transformer_analysis_npu910.csv"
    json_path = output_dir / "transformer_analysis_npu910.json"

    print(f"\nExporting results to {output_dir}/")
    export_xlsx(result, str(xlsx_path))
    export_csv(result, str(csv_path))
    export_json(result, str(json_path))

    print("\nDemo completed successfully!")
    print(f"Output files saved to: {output_dir}")

    return result


if __name__ == "__main__":
    main()
