#!/usr/bin/env python
"""Transpose Demo - matmul+bias 后 transpose

演示：
1. Linear (matmul + bias)
2. Transpose (4维矩阵转置)
3. 同时测试 bfp4/bfp8/bfp16 三种量化类型
4. 模拟真实 bfp 格式处理流程

真实流程：
- Golden: fp32 计算 → bfp 量化 → fp32 反量化 (带精度损失)
- DUT: 以 bfp 格式计算 → bfp 输出 → fp32 反量化 (用于比对)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from aidevtools import ops
from aidevtools.ops.base import get_records
from aidevtools.tools.compare.diff import compare_3col, print_compare_table
from aidevtools.formats.quantize import simulate_quantize


def run_with_dtype(dtype: str):
    """使用指定 dtype 运行 linear + transpose"""
    ops.seed(42)
    ops.clear()

    # Linear: (batch, heads, seq, d_k) -> (batch, heads, seq, d_v)
    # 输入: (2, 4, 8, 64), 输出: (2, 4, 8, 32)
    y = ops.linear((2, 4, 8, 64), 32, dtype=dtype)

    # Transpose: (2, 4, 8, 32) -> (2, 4, 32, 8)
    # 交换最后两个维度
    y = ops.transpose(y, axes=(0, 1, 3, 2), dtype=dtype)

    return get_records()


def generate_fake_dut(
    reference: np.ndarray,
    qtype: str = "bfp8",
    noise_level: float = 0.0005,
) -> np.ndarray:
    """
    生成假的 DUT 数据，模拟真实 bfp 格式处理流程

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
    print(f"\n{'=' * 80}")
    print(f"  Transpose Demo - Linear + Transpose (4D)")
    print(f"  模拟 bfp 格式比对流程")
    print(f"{'=' * 80}")

    # 测试三种 dtype
    dtypes = ["bfp4", "bfp8", "bfp16"]

    for dtype in dtypes:
        print(f"\n{'=' * 80}")
        print(f"  dtype = {dtype}")
        print(f"{'=' * 80}")

        # 运行模型
        print(f"\n[1] 运行模型 (Linear -> Transpose)")
        print(f"    框架自动执行: reference (fp32) 和 golden ({dtype} 量化)")
        records = run_with_dtype(dtype)

        for r in records:
            print(f"    {r['name']}: input={r['input'].shape}, output={r['golden'].shape}")

        # 生成假 DUT (使用 bfp 量化流程)
        print(f"\n[2] 生成假的 DUT 数据 (模拟 {dtype} 格式处理)")
        print(f"    流程: reference → {dtype} 量化/反量化 → 加小噪声")
        np.random.seed(123)
        dut_outputs = [generate_fake_dut(r["reference"], qtype=dtype, noise_level=0.0005) for r in records]

        for i, r in enumerate(records):
            print(f"    {r['name']}: dut={dut_outputs[i].shape}")

        # 比对
        print(f"\n[3] 比对结果")
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

        # 导出
        output_dir = Path(__file__).parent / "workspace" / dtype
        ops.dump(str(output_dir))
        print(f"    导出到: {output_dir}")


if __name__ == "__main__":
    main()
