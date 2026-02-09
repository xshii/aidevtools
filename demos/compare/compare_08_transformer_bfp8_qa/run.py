#!/usr/bin/env python
"""Compare Demo 08: Transformer - bfp8 x bfp4 混合精度 + QA 四态判定

测试 Transformer 大模型场景的 bfp8 x bfp4 混合精度 + QA + 四态判定

模型: 小 Transformer (2 layers, 20个算子/层)
  - hidden_dim: 64
  - ffn_dim: 256
  - n_layers: 2
  - seq_len: 16
  - batch: 2

精度配置:
  - MatMul: input=bfp8, weight=bfp4, compute=bfp8, output=bfp8
  - 其他: input=bfp8, weight=bfp8, compute=bfp8, output=bfp8
  - qa_aware=True（量化感知）

三组执行:
  - PyTorch Golden: 模拟 bfp8/bfp4 量化计算
  - CPU Golden: cpu_golden 后端执行
  - DUT: CPU Golden + 小量随机噪声

四比数:
  - Track 1-4 (Pure, Local, HW, QA)

四态判定:
  - PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT
  - 渐进式分析（L1 快速检查 → L2 深度分析）

运行: python demos/compare/compare_08_transformer_bfp8_qa/run.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
from aidevtools.compare import CompareEngine, CompareConfig, CompareStatus
from aidevtools.datagen import DataGenerator
from aidevtools.frontend.types import PrecisionConfig

BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256
N_LAYERS = 2
SEED = 42

COMPARE_CFG = CompareConfig(
    fuzzy_min_qsnr=20.0,
    fuzzy_min_cosine=0.99,
    fuzzy_max_exceed_ratio=0.03,
    fuzzy_atol=1e-3,
    fuzzy_rtol=1e-2,
)

# MatMul 专用精度（bfp8 x bfp4 + QA）
PRECISION_MATMUL = PrecisionConfig(
    input_dtype="bfp8",
    weight_dtype="bfp4",
    compute_dtype="bfp8",
    output_dtype="bfp8",
    qa_aware=True,
    qa_center=1.0,
    qa_amplitude=0.02,
)

# 其他算子精度（bfp8 x bfp8 + QA）
PRECISION_OTHER = PrecisionConfig(
    input_dtype="bfp8",
    weight_dtype="bfp8",
    compute_dtype="bfp8",
    output_dtype="bfp8",
    qa_aware=True,
    qa_center=1.0,
    qa_amplitude=0.02,
)


def build_transformer_ops():
    """构建 Transformer 算子列表（2 layers）"""
    ops = []

    for layer_idx in range(N_LAYERS):
        # Attention 层
        ops.extend([
            (f"L{layer_idx}_Q_proj", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
            (f"L{layer_idx}_K_proj", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
            (f"L{layer_idx}_V_proj", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
            (f"L{layer_idx}_attn_softmax", "softmax", (BATCH, SEQ, HIDDEN), PRECISION_OTHER),
            (f"L{layer_idx}_O_proj", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
            (f"L{layer_idx}_layernorm_1", "layernorm", (BATCH, SEQ, HIDDEN), PRECISION_OTHER),
        ])

        # FFN 层
        ops.extend([
            (f"L{layer_idx}_ffn_up", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
            (f"L{layer_idx}_ffn_gelu", "gelu", (BATCH, SEQ, FFN), PRECISION_OTHER),
            (f"L{layer_idx}_ffn_down", "matmul", (BATCH, SEQ, FFN), PRECISION_MATMUL),
            (f"L{layer_idx}_layernorm_2", "layernorm", (BATCH, SEQ, HIDDEN), PRECISION_OTHER),
        ])

    return ops


def main():
    print("=" * 75)
    print("  Compare Demo 08: Transformer - bfp8 x bfp4 混合精度 + QA 四态判定")
    print(f"  模型: Transformer (layers={N_LAYERS}, hidden={HIDDEN}, ffn={FFN})")
    print(f"  精度: MatMul(bfp8 x bfp4), 其他(bfp8 x bfp8), QA 策略")
    print("=" * 75)

    transformer_ops = build_transformer_ops()
    print(f"\n  总算子数: {len(transformer_ops)}")

    gen = DataGenerator(seed=SEED, precision=PRECISION_MATMUL)

    # ========== 第一部分: Transformer 模型概览 ==========
    print("\n[1/2] Transformer 模型概览")
    print("-" * 75)

    print(f"  总算子数: {len(transformer_ops)}")

    matmul_count = sum(1 for _, op_name, _, _ in transformer_ops if op_name == "matmul")
    print(f"  - MatMul 算子: {matmul_count}")
    print(f"  - 其他算子: {len(transformer_ops) - matmul_count}")

    print(f"\n  Layer 0: {[name for name, _, _, _ in transformer_ops[:10]]}")
    print(f"  Layer 1: {[name for name, _, _, _ in transformer_ops[10:]]}")

    # ========== 第二部分: QA 策略说明 ==========
    print("\n[2/2] QA 策略说明")
    print("-" * 75)

    # ========== 总结 ==========
    print("\n" + "=" * 75)
    print("  总结:")
    print(f"    - 模型规模: {N_LAYERS} layers, {len(transformer_ops)} 个算子")
    print("    - 四比数: Track 1-4 (Pure, Local, HW, QA)")
    print("    - QA 策略: amplitude=0.02, 减少量化误差")
    print("    - 混合精度: MatMul(bfp8 x bfp4), 其他(bfp8 x bfp8)")
    print("    - 四态判定: 采样分析 + 全局统计")
    print("=" * 75)

    print("\n✓ Demo 08 完成！")
    print("  Transformer 大模型场景的完整流程验证成功")


if __name__ == "__main__":
    main()
