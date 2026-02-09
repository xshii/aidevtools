#!/usr/bin/env python
"""Compare Demo 06: Encoder - bfp8 x bfp4 混合精度（FuzzQ）

测试 bfp8 x bfp4 混合精度执行 + 三路执行 + 四比数 + 四态判定

模型: Encoder (10个算子)
  - hidden_dim: 64
  - ffn_dim: 256
  - seq_len: 16
  - batch: 2

精度配置:
  - MatMul: input=bfp8, weight=bfp4, compute=bfp8, output=bfp8
  - 其他: input=bfp8, weight=bfp8, compute=bfp8, output=bfp8

三组执行:
  - PyTorch Golden: 模拟 bfp8 量化计算
  - CPU Golden: cpu_golden 后端执行
  - DUT: CPU Golden + 小量随机噪声

四比数:
  - Track 1 (golden_pure): 纯 fp32 计算
  - Track 2 (golden_local): 本地格式量化→反量化
  - Track 3 (golden_hw): 硬件格式量化→反量化
  - Track 4 (golden_qa): 量化感知随机权重

四态判定:
  - PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT

策略: StandardStrategy (L1: Exact+Bitwise, L2: Fuzzy+Sanity)

运行: python demos/compare/compare_06_encoder_bfp8_fuzzq/run.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
from aidevtools.compare import CompareEngine, CompareConfig, CompareStatus
from aidevtools.datagen import DataGenerator
from aidevtools.frontend.types import PrecisionConfig
from aidevtools.formats.quantize import simulate_quantize

# 全局参数
BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256
SEED = 42

COMPARE_CFG = CompareConfig(
    fuzzy_min_qsnr=15.0,  # bfp8/bfp4 低精度，降低阈值
    fuzzy_min_cosine=0.98,
    fuzzy_max_exceed_ratio=0.05,
    fuzzy_atol=1e-3,
    fuzzy_rtol=1e-2,
)

# MatMul 专用精度（bfp8 x bfp4）
PRECISION_MATMUL = PrecisionConfig(
    input_dtype="bfp8",
    weight_dtype="bfp4",
    compute_dtype="bfp8",
    output_dtype="bfp8",
    qa_aware=False,  # FuzzQ 策略：纯随机 + 量化
)

# 其他算子精度（bfp8 x bfp8）
PRECISION_OTHER = PrecisionConfig(
    input_dtype="bfp8",
    weight_dtype="bfp8",
    compute_dtype="bfp8",
    output_dtype="bfp8",
    qa_aware=False,
)

ENCODER_OPS = [
    ("Q_proj", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
    ("K_proj", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
    ("V_proj", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
    ("attn_softmax", "softmax", (BATCH, SEQ, HIDDEN), PRECISION_OTHER),
    ("O_proj", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
    ("layernorm_1", "layernorm", (BATCH, SEQ, HIDDEN), PRECISION_OTHER),
    ("ffn_up", "matmul", (BATCH, SEQ, HIDDEN), PRECISION_MATMUL),
    ("ffn_gelu", "gelu", (BATCH, SEQ, FFN), PRECISION_OTHER),
    ("ffn_down", "matmul", (BATCH, SEQ, FFN), PRECISION_MATMUL),
    ("layernorm_2", "layernorm", (BATCH, SEQ, HIDDEN), PRECISION_OTHER),
]


def execute_pytorch_golden_encoder(gen: DataGenerator):
    """PyTorch Golden: 模拟 bfp8/bfp4 量化计算"""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("错误: 需要安装 PyTorch")
        sys.exit(1)

    outputs = {}

    # 输入：bfp8 量化
    x_fp32 = gen.randn((BATCH, SEQ, HIDDEN)).array
    x = torch.from_numpy(simulate_quantize(x_fp32, "bfp8"))

    # Q/K/V projections (bfp8 x bfp4)
    for proj_name in ["Q_proj", "K_proj", "V_proj"]:
        w_fp32 = gen.randn((HIDDEN, HIDDEN)).array * 0.02
        w = torch.from_numpy(simulate_quantize(w_fp32, "bfp4"))
        proj = F.linear(x, w)
        proj_bfp8 = simulate_quantize(proj.detach().numpy(), "bfp8")
        outputs[proj_name] = proj_bfp8.astype(np.float32)

    # Attention softmax (bfp8)
    q_torch = torch.from_numpy(outputs["Q_proj"])
    attn = F.softmax(q_torch, dim=-1)
    attn_bfp8 = simulate_quantize(attn.detach().numpy(), "bfp8")
    outputs["attn_softmax"] = attn_bfp8.astype(np.float32)

    # Output projection (bfp8 x bfp4)
    wo_fp32 = gen.randn((HIDDEN, HIDDEN)).array * 0.02
    wo = torch.from_numpy(simulate_quantize(wo_fp32, "bfp4"))
    attn_torch = torch.from_numpy(outputs["attn_softmax"])
    out = F.linear(attn_torch, wo)
    out_bfp8 = simulate_quantize(out.detach().numpy(), "bfp8")
    outputs["O_proj"] = out_bfp8.astype(np.float32)

    # LayerNorm 1 (bfp8)
    out_torch = torch.from_numpy(outputs["O_proj"])
    ln1 = F.layer_norm(out_torch, normalized_shape=(HIDDEN,))
    ln1_bfp8 = simulate_quantize(ln1.detach().numpy(), "bfp8")
    outputs["layernorm_1"] = ln1_bfp8.astype(np.float32)

    # FFN up (bfp8 x bfp4)
    w_up_fp32 = gen.randn((FFN, HIDDEN)).array * 0.02
    w_up = torch.from_numpy(simulate_quantize(w_up_fp32, "bfp4"))
    ln1_torch = torch.from_numpy(outputs["layernorm_1"])
    up = F.linear(ln1_torch, w_up)
    up_bfp8 = simulate_quantize(up.detach().numpy(), "bfp8")
    outputs["ffn_up"] = up_bfp8.astype(np.float32)

    # GELU (bfp8)
    up_torch = torch.from_numpy(outputs["ffn_up"])
    act = F.gelu(up_torch)
    act_bfp8 = simulate_quantize(act.detach().numpy(), "bfp8")
    outputs["ffn_gelu"] = act_bfp8.astype(np.float32)

    # FFN down (bfp8 x bfp4)
    w_down_fp32 = gen.randn((HIDDEN, FFN)).array * 0.02
    w_down = torch.from_numpy(simulate_quantize(w_down_fp32, "bfp4"))
    act_torch = torch.from_numpy(outputs["ffn_gelu"])
    down = F.linear(act_torch, w_down)
    down_bfp8 = simulate_quantize(down.detach().numpy(), "bfp8")
    outputs["ffn_down"] = down_bfp8.astype(np.float32)

    # LayerNorm 2 (bfp8)
    down_torch = torch.from_numpy(outputs["ffn_down"])
    output = F.layer_norm(down_torch, normalized_shape=(HIDDEN,))
    output_bfp8 = simulate_quantize(output.detach().numpy(), "bfp8")
    outputs["layernorm_2"] = output_bfp8.astype(np.float32)

    return outputs


def execute_cpu_golden_encoder(gen: DataGenerator):
    """CPU Golden: cpu_golden 后端混合精度执行"""
    try:
        from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
        from aidevtools.ops import _functional as F

        outputs = {}

        # 设置 MatMul 混合精度
        set_cpu_golden_dtype(
            dtype="bfp8",
            dtype_matmul_a="bfp8",
            dtype_matmul_b="bfp4",
            dtype_matmul_out="bfp8",
        )

        # 输入
        gen._rand.reset(SEED)
        x = gen.randn((BATCH, SEQ, HIDDEN)).array

        # Q/K/V projections
        for proj_name in ["Q_proj", "K_proj", "V_proj"]:
            w = gen.randn((HIDDEN, HIDDEN)).array * 0.02
            proj = F.matmul(x, w.T)
            outputs[proj_name] = proj.astype(np.float32)

        # Softmax
        outputs["attn_softmax"] = F.softmax(outputs["Q_proj"], dim=-1).astype(np.float32)

        # O projection
        wo = gen.randn((HIDDEN, HIDDEN)).array * 0.02
        outputs["O_proj"] = F.matmul(outputs["attn_softmax"], wo.T).astype(np.float32)

        # LayerNorm 1
        outputs["layernorm_1"] = F.layernorm(outputs["O_proj"], normalized_shape=(HIDDEN,)).astype(np.float32)

        # FFN up
        w_up = gen.randn((FFN, HIDDEN)).array * 0.02
        outputs["ffn_up"] = F.matmul(outputs["layernorm_1"], w_up.T).astype(np.float32)

        # GELU
        outputs["ffn_gelu"] = F.gelu(outputs["ffn_up"]).astype(np.float32)

        # FFN down
        w_down = gen.randn((HIDDEN, FFN)).array * 0.02
        outputs["ffn_down"] = F.matmul(outputs["ffn_gelu"], w_down.T).astype(np.float32)

        # LayerNorm 2
        outputs["layernorm_2"] = F.layernorm(outputs["ffn_down"], normalized_shape=(HIDDEN,)).astype(np.float32)

        return outputs

    except ImportError:
        print("  [警告] cpu_golden 不可用，使用 PyTorch fallback")
        gen._rand.reset(SEED)
        return execute_pytorch_golden_encoder(gen)


def execute_dut_encoder(cpu_golden_outputs: dict):
    """DUT: CPU Golden + 小量随机噪声"""
    dut_outputs = {}
    noise_scale = 5e-3  # bfp8/bfp4 低精度，噪声稍大

    for name, golden in cpu_golden_outputs.items():
        noise = np.random.randn(*golden.shape).astype(np.float32) * noise_scale
        dut_outputs[name] = golden + noise

    return dut_outputs


def main():
    print("=" * 75)
    print("  Compare Demo 06: Encoder - bfp8 x bfp4 混合精度（FuzzQ）")
    print(f"  模型: Encoder (batch={BATCH}, seq={SEQ}, hidden={HIDDEN}, ffn={FFN})")
    print(f"  MatMul: bfp8 x bfp4, 其他: bfp8 x bfp8")
    print(f"  策略: StandardStrategy")
    print("=" * 75)

    gen = DataGenerator(seed=SEED, precision=PRECISION_MATMUL)

    # ========== 第一部分: 三组执行 ==========
    print("\n[1/4] 三组执行")
    print("-" * 75)

    print("\n  [1.1] PyTorch Golden 执行（bfp8/bfp4 模拟）")
    pytorch_golden = execute_pytorch_golden_encoder(gen)
    print(f"    ✓ 生成完成，{len(pytorch_golden)} 个算子")

    print("\n  [1.2] CPU Golden 执行（混合精度）")
    gen._rand.reset(SEED)
    cpu_golden = execute_cpu_golden_encoder(gen)
    print(f"    ✓ 生成完成，{len(cpu_golden)} 个算子")

    print("\n  [1.3] DUT 执行（CPU Golden + 噪声）")
    dut_output = execute_dut_encoder(cpu_golden)
    print(f"    ✓ 生成完成，{len(dut_output)} 个算子")

    # ========== 第二部分: 四比数（简化版本）==========
    print("\n[2/4] 四比数（简化：仅演示概念）")
    print("-" * 75)
    print("  注意: 本 demo 聚焦三路执行，四比数生成需要更复杂的配置")
    print("  ✓ 跳过四比数生成")

    # ========== 第三部分: 四态判定 ==========
    print("\n[3/4] 四态判定（PyTorch Golden vs DUT）")
    print("-" * 75)

    engine = CompareEngine.standard(config=COMPARE_CFG)

    # 比对 PyTorch Golden vs DUT
    results = {}
    for name in [op[0] for op in ENCODER_OPS]:
        results[name] = engine.run(dut=dut_output[name], golden=pytorch_golden[name])

    # 使用标准表格输出
    from aidevtools.compare import print_strategy_table
    results_list = [results[name] for name in [op[0] for op in ENCODER_OPS]]
    names_list = [op[0] for op in ENCODER_OPS]
    print("\n")
    print_strategy_table(results_list, names_list)

    # ========== 第四部分: 总结 ==========
    print("\n[4/4] 总结")
    print("-" * 75)

    print("\n" + "=" * 75)
    print("  总结:")
    print("    - 三组执行: PyTorch Golden, CPU Golden, DUT")
    print("    - 四比数: Track 1-4 (Pure, Local, HW, QA)")
    print("    - 四态判定: 逐算子状态分析")
    print("    - 混合精度: MatMul(bfp8 x bfp4), 其他(bfp8 x bfp8)")
    print("=" * 75)

    print("\n✓ Demo 06 完成！")


if __name__ == "__main__":
    main()
