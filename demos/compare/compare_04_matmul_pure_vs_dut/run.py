#!/usr/bin/env python
"""Compare Demo 04: 单算子 MatMul - Pure vs DUT

测试完整的三路执行 + 四比数 + 四态判定流程

模型: 单个 MatMul 算子
前端: DataGenerator + generate_four_track

三组执行:
  - PyTorch Golden: 纯 fp32 计算
  - CPU Golden: cpu_golden 后端执行（假编译模拟）
  - DUT: CPU Golden + 小量随机噪声（模拟硬件误差）

四比数:
  - Track 1 (golden_pure): 纯 fp32 计算
  - Track 2 (golden_local): 本地格式量化→反量化
  - Track 3 (golden_hw): 硬件格式量化→反量化
  - Track 4 (golden_qa): 量化感知随机权重

四态判定:
  - PASS: DUT 正确，Golden 有效
  - GOLDEN_SUSPECT: DUT 匹配，但 Golden 可疑
  - DUT_ISSUE: Golden 有效，DUT 有问题
  - BOTH_SUSPECT: 都可疑，需人工排查

策略: StandardStrategy (L1: Exact+Bitwise, L2: Fuzzy+Sanity)

运行: python demos/compare/compare_04_matmul_pure_vs_dut/run.py
"""
import sys
import numpy as np

# 确保能找到 aidevtools
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aidevtools.compare import CompareEngine, CompareConfig, print_strategy_table, CompareStatus
from aidevtools.datagen import DataGenerator
from aidevtools.frontend.types import PrecisionConfig

# 全局参数
M, N, K = 64, 128, 64
SEED = 42

COMPARE_CFG = CompareConfig(
    fuzzy_min_qsnr=15.0,  # bfp8 量化，降低阈值
    fuzzy_min_cosine=0.98,
    fuzzy_max_exceed_ratio=0.05,
    fuzzy_atol=1e-3,
    fuzzy_rtol=1e-2,
)

# bfp8 精度配置
PRECISION = PrecisionConfig(
    input_dtype="bfp8",
    weight_dtype="bfp8",
    compute_dtype="bfp8",
    output_dtype="bfp8",
    qa_aware=False,  # 纯随机
)


def execute_pytorch_golden(gen: DataGenerator):
    """PyTorch Golden: 模拟 bfp8 量化计算"""
    try:
        import torch
    except ImportError:
        print("错误: 需要安装 PyTorch")
        print("  pip install torch")
        sys.exit(1)

    from aidevtools.formats.quantize import simulate_quantize

    # 生成输入数据（纯随机）
    x_fp32 = gen.randn((M, K)).array
    w_fp32 = gen.randn((K, N)).array

    # 模拟 bfp8 量化
    x = torch.from_numpy(simulate_quantize(x_fp32, "bfp8"))
    w = torch.from_numpy(simulate_quantize(w_fp32, "bfp8"))

    # PyTorch 计算
    y = torch.matmul(x, w)

    # 输出量化到 bfp8
    y_bfp8 = simulate_quantize(y.detach().numpy(), "bfp8")
    return y_bfp8.astype(np.float32)


def execute_cpu_golden(gen: DataGenerator):
    """CPU Golden: cpu_golden 后端执行（假编译模拟）

    使用 bfp8 精度（块浮点 8bit）
    """
    from aidevtools.ops import _functional as F
    from aidevtools.ops.cpu_golden import set_cpu_golden_dtype

    # 设置为 bfp8 精度（与其他 demos 一致）
    set_cpu_golden_dtype(dtype="bfp8")

    # 生成输入数据（与 PyTorch 相同的随机数）
    gen._rand.reset(SEED)  # 重置随机数生成器
    x = gen.randn((M, K)).array
    w = gen.randn((K, N)).array

    # cpu_golden 执行
    y = F.matmul(x, w)
    return y.astype(np.float32)


def execute_dut(cpu_golden_output: np.ndarray):
    """DUT: CPU Golden + 小量随机噪声（模拟硬件误差）"""
    noise_scale = 1e-5  # 噪声幅度
    noise = np.random.randn(*cpu_golden_output.shape).astype(np.float32) * noise_scale
    dut_output = cpu_golden_output + noise
    return dut_output


def main():
    print("=" * 75)
    print("  Compare Demo 04: 单算子 MatMul - bfp8 vs DUT")
    print(f"  模型: MatMul ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
    print(f"  精度: bfp8 (块浮点 8bit)")
    print(f"  策略: StandardStrategy (L1: Exact+Bitwise, L2: Fuzzy+Sanity)")
    print("=" * 75)

    # 创建数据生成器
    gen = DataGenerator(seed=SEED, precision=PRECISION)

    # ========== 第一部分: 三组执行 ==========
    print("\n[1/3] 三组执行")
    print("-" * 75)

    print("\n  [1.1] PyTorch Golden 执行")
    pytorch_golden = execute_pytorch_golden(gen)
    print(f"    ✓ 输出 shape: {pytorch_golden.shape}")
    print(f"    ✓ 数值范围: [{pytorch_golden.min():.6f}, {pytorch_golden.max():.6f}]")

    print("\n  [1.2] CPU Golden 执行（假编译）")
    cpu_golden = execute_cpu_golden(gen)
    print(f"    ✓ 输出 shape: {cpu_golden.shape}")
    print(f"    ✓ 数值范围: [{cpu_golden.min():.6f}, {cpu_golden.max():.6f}]")

    print("\n  [1.3] DUT 执行（CPU Golden + 噪声）")
    dut_output = execute_dut(cpu_golden)
    print(f"    ✓ 输出 shape: {dut_output.shape}")
    print(f"    ✓ 数值范围: [{dut_output.min():.6f}, {dut_output.max():.6f}]")

    # ========== 第二部分: 四比数 ==========
    print("\n[2/3] 四比数（Four Track Golden）")
    print("-" * 75)

    # 重置生成器，确保与三组执行使用相同的随机数
    gen._rand.reset(SEED)

    tracks = gen.generate_four_track(
        "matmul",
        input_shape=(M, K),
        precision=PRECISION,
        weight_shape=(K, N),
    )

    print(f"\n  Track 1 (golden_pure):  {tracks.golden_pure.shape}")
    print(f"  Track 2 (golden_local): {tracks.golden_local.shape if tracks.golden_local is not None else 'N/A'}")
    print(f"  Track 3 (golden_hw):    {tracks.golden_hw.shape if tracks.golden_hw is not None else 'N/A'}")
    print(f"  Track 4 (golden_qa):    {tracks.golden_qa.shape if tracks.golden_qa is not None else 'N/A'}")

    # ========== 第三部分: 四态判定 ==========
    print("\n[3/3] 四态判定（Four-State Comparison）")
    print("-" * 75)

    engine = CompareEngine.standard(config=COMPARE_CFG)

    # 比对 1: PyTorch vs CPU Golden（验证后端正确性）
    print("\n  [3.1] PyTorch Golden vs CPU Golden")
    result_pytorch_cpu = engine.run(dut=cpu_golden, golden=pytorch_golden)
    print(f"    状态: {result_pytorch_cpu.get('status')}")
    if result_pytorch_cpu.get('exact'):
        print(f"    Exact: {'PASS' if result_pytorch_cpu['exact'].passed else 'FAIL'}")
    if result_pytorch_cpu.get('fuzzy_pure'):
        print(f"    Fuzzy: {'PASS' if result_pytorch_cpu['fuzzy_pure'].passed else 'FAIL'}, "
              f"QSNR={result_pytorch_cpu['fuzzy_pure'].qsnr:.1f} dB")

    # 比对 2: PyTorch vs DUT（主比对）
    print("\n  [3.2] PyTorch Golden vs DUT")
    result_pytorch_dut = engine.run(dut=dut_output, golden=pytorch_golden)
    print(f"    状态: {result_pytorch_dut.get('status')}")
    if result_pytorch_dut.get('exact'):
        print(f"    Exact: {'PASS' if result_pytorch_dut['exact'].passed else 'FAIL'}")
    if result_pytorch_dut.get('fuzzy_pure'):
        print(f"    Fuzzy: {'PASS' if result_pytorch_dut['fuzzy_pure'].passed else 'FAIL'}, "
              f"QSNR={result_pytorch_dut['fuzzy_pure'].qsnr:.1f} dB")

    # 比对 3: CPU Golden vs DUT
    print("\n  [3.3] CPU Golden vs DUT")
    result_cpu_dut = engine.run(dut=dut_output, golden=cpu_golden)
    print(f"    状态: {result_cpu_dut.get('status')}")
    if result_cpu_dut.get('exact'):
        print(f"    Exact: {'PASS' if result_cpu_dut['exact'].passed else 'FAIL'}")
    if result_cpu_dut.get('fuzzy_pure'):
        print(f"    Fuzzy: {'PASS' if result_cpu_dut['fuzzy_pure'].passed else 'FAIL'}, "
              f"QSNR={result_cpu_dut['fuzzy_pure'].qsnr:.1f} dB")

    # 比对 4: Four Track - Pure vs HW/Local/QA
    print("\n  [3.4] Four Track Golden 比对")

    # Track 1 (Pure) vs Track 3 (HW)
    if tracks.golden_hw is not None:
        r_hw = engine.run(dut=tracks.golden_hw, golden=tracks.golden_pure)
        qsnr_hw = r_hw.get('fuzzy_pure').qsnr if r_hw.get('fuzzy_pure') else float('inf')
        print(f"    Pure vs HW: {r_hw.get('status')}, QSNR={qsnr_hw:.1f} dB")

    # Track 1 (Pure) vs Track 2 (Local)
    if tracks.golden_local is not None:
        r_local = engine.run(dut=tracks.golden_local, golden=tracks.golden_pure)
        qsnr_local = r_local.get('fuzzy_pure').qsnr if r_local.get('fuzzy_pure') else float('inf')
        print(f"    Pure vs Local: {r_local.get('status')}, QSNR={qsnr_local:.1f} dB")

    # Track 1 (Pure) vs Track 4 (QA)
    if tracks.golden_qa is not None:
        r_qa = engine.run(dut=tracks.golden_qa, golden=tracks.golden_pure)
        qsnr_qa = r_qa.get('fuzzy_pure').qsnr if r_qa.get('fuzzy_pure') else float('inf')
        print(f"    Pure vs QA: {r_qa.get('status')}, QSNR={qsnr_qa:.1f} dB")

    # ========== 总结 ==========
    print("\n" + "=" * 75)
    print("  总结:")
    print("    - 三组执行: PyTorch Golden, CPU Golden, DUT")
    print("    - 四比数: Track 1-4 (Pure, Local, HW, QA)")
    print("    - 四态判定: PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT")
    print("=" * 75)

    # ========== 验证 ==========
    print("\n[验证] 检查结果是否符合预期...")

    # 1. PyTorch vs CPU Golden 应该一致
    # 都使用 bfp8 量化，但不同实现可能有差异
    pytorch_cpu_exact = result_pytorch_cpu.get('exact')
    pytorch_cpu_fuzzy = result_pytorch_cpu.get('fuzzy_pure')
    if pytorch_cpu_exact and pytorch_cpu_fuzzy:
        if pytorch_cpu_exact.passed:
            print("  ✓ PyTorch vs CPU Golden: bit-exact 一致")
        else:
            # bfp8 量化计算，不同实现可能有差异，QSNR 预期在 5-20 dB
            assert pytorch_cpu_fuzzy.qsnr > 5, f"PyTorch vs CPU Golden QSNR 过低: {pytorch_cpu_fuzzy.qsnr:.1f} dB"
            print(f"  ✓ PyTorch vs CPU Golden: QSNR={pytorch_cpu_fuzzy.qsnr:.1f} dB (bfp8，不同实现有差异)")

    # 2. PyTorch vs DUT 应该有误差（bfp8 量化 + 噪声）
    pytorch_dut_exact = result_pytorch_dut.get('exact')
    pytorch_dut_fuzzy = result_pytorch_dut.get('fuzzy_pure')
    if pytorch_dut_exact and pytorch_dut_fuzzy:
        assert not pytorch_dut_exact.passed, "PyTorch vs DUT 应该有量化+噪声误差"
        # DUT 是 CPU Golden (bfp8) + 噪声，QSNR 预期在 5-20 dB
        assert pytorch_dut_fuzzy.qsnr > 5, f"QSNR 应该 >5 dB，实际 {pytorch_dut_fuzzy.qsnr:.1f} dB"
        print(f"  ✓ PyTorch vs DUT: QSNR={pytorch_dut_fuzzy.qsnr:.1f} dB (bfp8 + 噪声)")

    # 3. 不应该有 GOLDEN_SUSPECT（golden 是纯 fp32）
    bad_status = []
    for name, r in [("PyTorch vs CPU", result_pytorch_cpu),
                    ("PyTorch vs DUT", result_pytorch_dut),
                    ("CPU vs DUT", result_cpu_dut)]:
        if r.get('status') in [CompareStatus.GOLDEN_SUSPECT, CompareStatus.BOTH_SUSPECT]:
            bad_status.append(name)
    assert len(bad_status) == 0, f"不应该怀疑 golden 数据: {bad_status}"
    print("  ✓ Golden 数据无异常")

    print("\n✓ 所有验证通过！")


if __name__ == "__main__":
    main()
