#!/usr/bin/env python3
"""
Transformer 模型时延分析 Demo

演示如何使用 Paper Analysis 模块分析一个 3D 1MB 级别矩阵的 Transformer 模型
在 NPU 910 上的时延表现。

Usage:
    python transformer_analysis_demo.py
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aidevtools.analysis import (
    PaperAnalyzer,
    OpProfile,
    MatMulDtypeConfig,
    PassConfig,
    PassPreset,
    profile_matmul,
    profile_layernorm,
    profile_softmax,
    profile_attention,
    profile_gelu,
    profile_add,
    export_xlsx,
    export_csv,
    export_json,
    load_chip_spec,
    list_chips,
)


def create_transformer_profiles(
    batch_size: int = 4,
    seq_len: int = 512,
    hidden_size: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = 3072,
    dtype: str = "fp16",
) -> list:
    """
    创建 Transformer 单层的算子 Profiles

    模型参数 (约 1MB 级别每个矩阵):
    - batch_size: 4
    - seq_len: 512
    - hidden_size: 768
    - num_heads: 12
    - ffn_hidden: 3072

    单个矩阵大小示例:
    - Q/K/V 投影权重: 768 x 768 x 2 bytes = 1.125 MB
    - FFN1 权重: 768 x 3072 x 2 bytes = 4.5 MB
    - FFN2 权重: 3072 x 768 x 2 bytes = 4.5 MB
    """
    profiles = []
    head_dim = hidden_size // num_heads
    bytes_per_elem = 2 if dtype == "fp16" else 4

    # ============================================================
    # 1. Self-Attention
    # ============================================================

    # 1.1 LayerNorm (pre-attention)
    x = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float16)
    gamma = np.zeros((hidden_size,), dtype=np.float16)
    beta = np.zeros((hidden_size,), dtype=np.float16)
    ln_profile = profile_layernorm(x, gamma, beta)
    ln_profile.name = "attn_ln"
    profiles.append(ln_profile)

    # 1.2 Q 投影: [B, S, H] @ [H, H] -> [B, S, H]
    input_q = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float16)
    weight_q = np.zeros((hidden_size, hidden_size), dtype=np.float16)
    q_profile = profile_matmul(input_q, weight_q)
    q_profile.name = "q_proj"
    profiles.append(q_profile)

    # 1.3 K 投影
    k_profile = profile_matmul(input_q, weight_q)
    k_profile.name = "k_proj"
    profiles.append(k_profile)

    # 1.4 V 投影
    v_profile = profile_matmul(input_q, weight_q)
    v_profile.name = "v_proj"
    profiles.append(v_profile)

    # 1.5 Attention: Q @ K^T @ V
    # 重塑为多头: [B, S, H] -> [B, num_heads, S, head_dim]
    q = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float16)
    k = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float16)
    v = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float16)
    attn_profile = profile_attention(q, k, v)
    attn_profile.name = "self_attn"
    profiles.append(attn_profile)

    # 1.6 Output 投影: [B, S, H] @ [H, H] -> [B, S, H]
    out_proj = profile_matmul(input_q, weight_q)
    out_proj.name = "out_proj"
    profiles.append(out_proj)

    # 1.7 残差连接 Add
    a = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float16)
    add_profile = profile_add(a, a)
    add_profile.name = "attn_residual"
    profiles.append(add_profile)

    # ============================================================
    # 2. FFN (Feed-Forward Network)
    # ============================================================

    # 2.1 LayerNorm (pre-FFN)
    ln2_profile = profile_layernorm(x, gamma, beta)
    ln2_profile.name = "ffn_ln"
    profiles.append(ln2_profile)

    # 2.2 FFN1: [B, S, H] @ [H, FFN] -> [B, S, FFN]
    ffn_input = np.zeros((batch_size, seq_len, hidden_size), dtype=np.float16)
    ffn1_weight = np.zeros((hidden_size, ffn_hidden), dtype=np.float16)
    ffn1_profile = profile_matmul(ffn_input, ffn1_weight)
    ffn1_profile.name = "ffn1"
    profiles.append(ffn1_profile)

    # 2.3 GELU 激活
    ffn_hidden_tensor = np.zeros((batch_size, seq_len, ffn_hidden), dtype=np.float16)
    gelu_profile = profile_gelu(ffn_hidden_tensor)
    gelu_profile.name = "ffn_gelu"
    profiles.append(gelu_profile)

    # 2.4 FFN2: [B, S, FFN] @ [FFN, H] -> [B, S, H]
    ffn2_weight = np.zeros((ffn_hidden, hidden_size), dtype=np.float16)
    ffn2_profile = profile_matmul(ffn_hidden_tensor, ffn2_weight)
    ffn2_profile.name = "ffn2"
    profiles.append(ffn2_profile)

    # 2.5 残差连接 Add
    add2_profile = profile_add(a, a)
    add2_profile.name = "ffn_residual"
    profiles.append(add2_profile)

    return profiles


def print_profiles_info(profiles: list):
    """打印 profile 信息"""
    print("\n" + "=" * 80)
    print("Transformer Layer Operator Profiles")
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
    print("Analyzing on NPU 910 (Ascend 910)")
    print("=" * 80)

    # 显示可用芯片
    print(f"\nAvailable chips: {list_chips()}")

    # 加载芯片规格
    chip = load_chip_spec("npu_910")
    print(f"\nChip: {chip.name}")
    print(f"  Cube FP16: {chip.cube.fp16_tflops} TFLOPS")
    print(f"  Vector FP16: {chip.vector.fp16_gflops} GFLOPS")
    print(f"  HBM Bandwidth: {chip.memory.hbm.bandwidth_gbps} GB/s")
    print(f"  HBM Capacity: {chip.memory.hbm.capacity_bytes / 1024**3:.0f} GB")

    # 创建模型配置 (约 1MB 级别矩阵)
    model_config = {
        "batch_size": 4,
        "seq_len": 512,
        "hidden_size": 768,
        "num_heads": 12,
        "ffn_hidden": 3072,
        "dtype": "fp16",
    }

    print(f"\nModel Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")

    # 创建算子 profiles
    profiles = create_transformer_profiles(**model_config)
    print_profiles_info(profiles)

    # ============================================================
    # 分析
    # ============================================================

    # 使用标准优化配置
    pass_config = PassConfig.from_preset(PassPreset.STANDARD)
    print(f"\nPass Configuration: {pass_config.preset.value}")
    print(f"  Roofline: {pass_config.roofline_enabled}")
    print(f"  Memory Efficiency: {pass_config.memory_efficiency_enabled}")
    print(f"  Forward Prefetch: {pass_config.forward_prefetch_enabled}")
    print(f"  Backward Prefetch: {pass_config.backward_prefetch_enabled}")
    print(f"  Cube/Vector Parallel: {pass_config.cube_vector_parallel_enabled}")
    print(f"  Overhead: {pass_config.overhead_enabled}")

    # 创建分析器
    analyzer = PaperAnalyzer(
        chip="npu_910",
        pass_config=pass_config,
    )

    # 添加 profiles
    analyzer.add_profiles(profiles)

    # 执行分析
    print("\nRunning analysis...")
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
              f"{bd.compute_time_us:10.2f} | {bd.memory_time_us:10.2f} | "
              f"{bd.roofline_time_us:10.2f} | {bd.total_time_us:10.2f} | "
              f"{bd.bottleneck:10s}")

    print("=" * 80)

    # ============================================================
    # 导出
    # ============================================================

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 导出 xlsx
    xlsx_path = output_dir / "transformer_analysis_npu910.xlsx"
    print(f"\nExporting to Excel: {xlsx_path}")
    export_xlsx(result, str(xlsx_path))

    # 导出 CSV
    csv_path = output_dir / "transformer_analysis_npu910.csv"
    print(f"Exporting to CSV: {csv_path}")
    export_csv(result, str(csv_path))

    # 导出 JSON
    json_path = output_dir / "transformer_analysis_npu910.json"
    print(f"Exporting to JSON: {json_path}")
    export_json(result, str(json_path))

    print("\nDemo completed successfully!")
    print(f"Output files saved to: {output_dir}")

    return result


if __name__ == "__main__":
    main()
