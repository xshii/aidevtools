#!/usr/bin/env python
"""MiniTransformer Demo - 完整比对流程演示

演示完整的 golden 生成与比对流程：
1. 使用 ops API 定义算子序列
2. 执行 cpp golden (via subprocess) 和 reference (pure fp32)
3. 构造假的 DUT 数据（模拟 bfp 格式处理）
4. 三列比对：exact / fuzzy_pure / fuzzy_qnt

比对流程说明：
- reference: fp32/fp64 高精度计算（用于 fuzzy_pure 比对）
- golden: cpp golden 计算（带量化，用于 fuzzy_qnt 比对）
- DUT: 芯片输出（本 demo 用模拟数据）

三列比对：
- exact: DUT vs golden 的 bit 级精确比对
- fuzzy_pure: DUT vs reference (不考虑量化误差)
- fuzzy_qnt: DUT vs golden (量化感知比对)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from aidevtools import ops
from aidevtools.ops.base import get_records, set_golden_mode
from aidevtools.tools.compare.diff import compare_3col, print_compare_table
from aidevtools.formats.quantize import simulate_quantize

# 设置 cpu golden dtype 并使用 cpp golden
from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
set_cpu_golden_dtype("gfp16")  # 使用 gfp16 格式
set_golden_mode("cpp")


def run_model():
    """运行模型，返回每层的 golden 和 reference"""
    ops.seed(42)
    ops.clear()

    # 定义模型: MatMul -> LayerNorm -> Softmax
    # 使用有 cpp golden 的算子 (支持 batch)
    # MatMul: (2, 8, 64) @ (64, 32) -> (2, 8, 32)
    y = ops.matmul((2, 8, 64), (64, 32), dtype="bfp8")
    y = ops.layernorm(y, 32, dtype="bfp8")
    y = ops.softmax(y, dtype="bfp8")

    return get_records()


def generate_fake_dut(
    reference: np.ndarray,
    qtype: str = "bfp8",
    noise_level: float = 0.001,
) -> np.ndarray:
    """
    生成假的 DUT 数据，模拟真实 bfp 格式处理流程

    模拟流程：
    1. 从 reference (fp32 精确值) 开始
    2. 应用 bfp 量化/反量化，模拟 DUT 的 bfp 格式计算
    3. 添加小噪声，模拟 DUT 的计算误差

    Args:
        reference: fp32 精确计算结果 (reference)
        qtype: 量化格式 (bfp4, bfp8, bfp16)
        noise_level: 噪声水平

    Returns:
        模拟的 DUT 输出 (fp32)
    """
    # Step 1: 对 reference 进行 bfp 量化/反量化，模拟 DUT 的 bfp 格式处理
    dut_quantized = simulate_quantize(reference.astype(np.float32), qtype)

    # Step 2: 添加小噪声，模拟 DUT 计算误差
    noise = np.random.randn(*dut_quantized.shape).astype(np.float32) * noise_level
    dut_result = dut_quantized + noise

    return dut_result


def main():
    print(f"\n{'=' * 70}")
    print(f"  MiniTransformer Demo - 完整比对流程")
    print(f"{'=' * 70}")
    print(f"  golden_mode: cpp (via subprocess)")
    print(f"  quantization: gfp16 (cpp) + bfp8 (simulation)")

    # 1. 运行模型
    print("\n[1] 运行模型 (MatMul -> LayerNorm -> Softmax)")
    print("    框架自动执行:")
    print("    - reference: fp32/fp64 高精度计算 (用于 fuzzy_pure)")
    print("    - golden: cpp golden via subprocess (用于 fuzzy_qnt)")
    records = run_model()

    for r in records:
        print(f"    {r['name']}: input={r['input'].shape}, golden={r['golden'].shape}")

    # 2. 生成假的 DUT 数据
    print("\n[2] 生成假的 DUT 数据 (模拟 bfp8 格式处理)")
    print("    流程: reference → bfp8 量化/反量化 → 加小噪声")
    np.random.seed(123)
    dut_outputs = []
    for r in records:
        # 使用 reference (fp32 精确值) 生成假 DUT
        dut = generate_fake_dut(r["reference"], qtype="bfp8", noise_level=0.001)
        dut_outputs.append(dut)
        print(f"    {r['name']}: dut={dut.shape}")

    # 3. 使用框架比对工具
    print("\n[3] 比对 (使用框架 compare_3col)")
    results = []
    for i, r in enumerate(records):
        # golden_pure = reference (fp64), golden_qnt = golden (带量化)
        result = compare_3col(
            op_name=r["op"],
            op_id=i,
            result=dut_outputs[i],
            golden_pure=r["reference"],
            golden_qnt=r["golden"],
        )
        results.append(result)

    print_compare_table(results)

    # 4. 导出 bin
    print("[4] 导出 bin 文件")
    output_dir = Path(__file__).parent / "workspace"
    ops.dump(str(output_dir))

    # 同时导出 dut
    from aidevtools.formats.base import save as save_data
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(records):
        save_data(str(output_dir / f"{r['name']}_dut.bin"), dut_outputs[i])

    print(f"\n    输出目录: {output_dir}")
    print(f"    文件列表:")
    for f in sorted(output_dir.glob("*.bin")):
        print(f"      {f.name}")


if __name__ == "__main__":
    main()
