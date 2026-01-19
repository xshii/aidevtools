#!/usr/bin/env python
"""MiniTransformer Demo - 使用简化 ops API

演示：
1. 使用 ops API 定义算子序列（自动生成数据）
2. 执行 golden + reference
3. 构造假的 DUT 数据（模拟 bfp 格式处理流程）
4. 使用框架比对工具比对

真实流程：
- Golden: fp32 计算 → bfp8 量化 → fp32 反量化 (带精度损失)
- DUT: 以 bfp8 格式计算 → bfp8 输出 → fp32 反量化 (用于比对)

本 demo 的假 DUT 模拟：
- 对 fp32 golden 结果进行 bfp8 量化/反量化
- 添加小噪声模拟 DUT 计算误差
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from aidevtools import ops
from aidevtools.ops.base import get_records
from aidevtools.tools.compare.diff import compare_3col, print_compare_table
from aidevtools.formats.quantize import simulate_quantize


def run_model():
    """运行模型，返回每层的 golden 和 reference"""
    ops.seed(42)
    ops.clear()

    # 定义模型: Linear -> LayerNorm -> Softmax
    y = ops.linear((2, 8, 64), 32, dtype="bfp8")
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
    print(f"  MiniTransformer Demo - 模拟 bfp 格式比对流程")
    print(f"{'=' * 70}")

    # 1. 运行模型
    print("\n[1] 运行模型 (Linear -> LayerNorm -> Softmax)")
    print("    dtype=bfp8, 框架自动执行:")
    print("    - reference: fp32/fp64 精确计算")
    print("    - golden: fp32 → bfp8 量化 → fp32 反量化")
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
