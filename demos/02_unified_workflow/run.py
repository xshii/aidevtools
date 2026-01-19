#!/usr/bin/env python
"""
统一工作流 Demo

演示新的架构:
1. 全局配置 (golden_mode, precision, seed)
2. 统一的 Tensor 格式 (fp32 + quantized)
3. register_op 算子注册
4. 三列比对结果 (exact, fuzzy_pure, fuzzy_qnt)
"""
import numpy as np
import sys
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aidevtools.core import (
    set_config,
    get_config,
    reset_config,
    Tensor,
    generate_random,
    generate_weight,
    get_engine,
    clear,
    list_ops,
)
from aidevtools.core.op import run_golden
from aidevtools.tools.compare.diff import compare_3col, print_compare_table, FullCompareResult
from typing import List


def demo_config():
    """演示全局配置"""
    print("=" * 60)
    print("1. 全局配置演示")
    print("=" * 60)

    # 重置为默认配置
    reset_config()
    config = get_config()
    print(f"\n默认配置:")
    print(f"  golden_mode: {config.golden_mode}")
    print(f"  precision:   {config.precision}")
    print(f"  seed:        {config.seed}")
    print(f"  exact:       max_abs={config.exact.max_abs}, max_count={config.exact.max_count}")
    print(f"  fuzzy:       atol={config.fuzzy.atol}, rtol={config.fuzzy.rtol}, min_qsnr={config.fuzzy.min_qsnr}dB")

    # 修改配置
    set_config(
        golden_mode="python",
        precision="quant",
        seed=42,
    )
    config = get_config()
    print(f"\n修改后配置:")
    print(f"  golden_mode: {config.golden_mode}")
    print(f"  precision:   {config.precision}")
    print(f"  seed:        {config.seed}")


def demo_tensor():
    """演示 Tensor 格式"""
    print("\n" + "=" * 60)
    print("2. Tensor 格式演示")
    print("=" * 60)

    # 创建 fp32 Tensor
    print("\n从 fp32 创建 Tensor:")
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    t1 = Tensor.from_fp32(data, qtype="float32")
    print(f"  {t1}")
    print(f"  fp32: {t1.fp32}")
    print(f"  quantized: {t1.quantized}")

    # 创建带量化的 Tensor
    print("\n创建 bfp16 量化 Tensor:")
    t2 = Tensor.from_fp32(data, qtype="bfp16")
    print(f"  {t2}")
    print(f"  fp32: {t2.fp32}")
    print(f"  quantized shape: {t2.quantized.shape if t2.quantized is not None else None}")
    print(f"  meta: {t2.meta}")

    # 量化-反量化模拟精度损失
    print("\n量化-反量化 (模拟精度损失):")
    t3 = t2.quantize_dequantize()
    print(f"  原始 fp32:     {data}")
    print(f"  量化后还原:    {t3.fp32}")
    print(f"  误差:          {np.abs(data - t3.fp32)}")

    # 生成随机张量
    print("\n生成随机张量:")
    t4 = generate_random(shape=(2, 3), qtype="bfp8", seed=42)
    print(f"  {t4}")
    print(f"  fp32:\n{t4.fp32}")

    # 生成权重张量
    print("\n生成权重张量 (Xavier 初始化):")
    t5 = generate_weight(shape=(64, 128), qtype="bfp16", seed=42, init="xavier")
    print(f"  {t5}")
    print(f"  mean: {t5.fp32.mean():.6f}, std: {t5.fp32.std():.6f}")


def demo_ops():
    """演示算子注册和执行"""
    print("\n" + "=" * 60)
    print("3. 算子注册演示")
    print("=" * 60)

    # 列出已注册的算子
    print(f"\n已注册的算子: {list_ops()}")

    # 执行 golden
    print("\n执行 linear golden:")
    x = np.random.randn(2, 4, 64).astype(np.float32)
    w = np.random.randn(64, 128).astype(np.float32)
    y = run_golden("linear", x, w)
    print(f"  input:  {x.shape}")
    print(f"  weight: {w.shape}")
    print(f"  output: {y.shape}")

    # 执行 relu
    print("\n执行 relu golden:")
    y_relu = run_golden("relu", y)
    print(f"  input:  {y.shape}")
    print(f"  output: {y_relu.shape}")
    print(f"  负值数量: {np.sum(y < 0)} -> {np.sum(y_relu < 0)}")


def demo_compare():
    """演示三列比对"""
    print("\n" + "=" * 60)
    print("4. 三列比对演示")
    print("=" * 60)

    config = get_config()
    results: List[FullCompareResult] = []

    # 场景 1: 完美匹配 (PERFECT)
    print("\n场景 1: 完美匹配")
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    golden_pure = x1.copy()
    golden_qnt = x1.copy()
    result = x1.copy()  # DUT 输出完全一致

    r1 = compare_3col(
        op_name="linear", op_id=0,
        result=result,
        golden_pure=golden_pure,
        golden_qnt=golden_qnt,
        exact_max_abs=config.exact.max_abs,
        fuzzy_atol=config.fuzzy.atol,
        fuzzy_rtol=config.fuzzy.rtol,
        fuzzy_min_qsnr=config.fuzzy.min_qsnr,
        fuzzy_min_cosine=config.fuzzy.min_cosine,
    )
    results.append(r1)
    print(f"  status: {r1.status}")

    # 场景 2: 小误差 (PASS)
    print("\n场景 2: 小误差 (fuzzy 通过)")
    result2 = x1 + np.array([1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float32)

    r2 = compare_3col(
        op_name="relu", op_id=0,
        result=result2,
        golden_pure=golden_pure,
        golden_qnt=golden_qnt,
        exact_max_abs=config.exact.max_abs,
        fuzzy_atol=config.fuzzy.atol,
        fuzzy_rtol=config.fuzzy.rtol,
        fuzzy_min_qsnr=config.fuzzy.min_qsnr,
        fuzzy_min_cosine=config.fuzzy.min_cosine,
    )
    results.append(r2)
    print(f"  status: {r2.status}")
    print(f"  max_abs: {r2.fuzzy_qnt.max_abs:.2e}")

    # 场景 3: 量化误差 (QUANT_ISSUE)
    print("\n场景 3: 量化导致的误差")
    # 模拟: pure golden 和 DUT 接近，但 quant golden 和 DUT 差异大
    golden_qnt3 = x1 + np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)  # 量化引入的偏差
    result3 = x1 + np.array([1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float32)  # DUT 接近 pure

    r3 = compare_3col(
        op_name="softmax", op_id=0,
        result=result3,
        golden_pure=golden_pure,
        golden_qnt=golden_qnt3,
        exact_max_abs=config.exact.max_abs,
        fuzzy_atol=config.fuzzy.atol,
        fuzzy_rtol=config.fuzzy.rtol,
        fuzzy_min_qsnr=config.fuzzy.min_qsnr,
        fuzzy_min_cosine=config.fuzzy.min_cosine,
    )
    results.append(r3)
    print(f"  status: {r3.status}")
    print(f"  fuzzy_pure passed: {r3.fuzzy_pure.passed}")
    print(f"  fuzzy_qnt passed: {r3.fuzzy_qnt.passed}")

    # 场景 4: 大误差 (FAIL)
    print("\n场景 4: 大误差")
    result4 = x1 + np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # 很大的误差

    r4 = compare_3col(
        op_name="matmul", op_id=0,
        result=result4,
        golden_pure=golden_pure,
        golden_qnt=golden_qnt,
        exact_max_abs=config.exact.max_abs,
        fuzzy_atol=config.fuzzy.atol,
        fuzzy_rtol=config.fuzzy.rtol,
        fuzzy_min_qsnr=config.fuzzy.min_qsnr,
        fuzzy_min_cosine=config.fuzzy.min_cosine,
    )
    results.append(r4)
    print(f"  status: {r4.status}")
    print(f"  max_abs: {r4.fuzzy_qnt.max_abs:.2e}")
    print(f"  qsnr: {r4.fuzzy_qnt.qsnr:.1f} dB")

    # 打印汇总表格
    print("\n" + "=" * 60)
    print("比对结果汇总")
    print_compare_table(results)


def demo_full_workflow():
    """演示完整工作流"""
    print("\n" + "=" * 60)
    print("5. 完整工作流演示")
    print("=" * 60)

    # 设置全局配置
    set_config(
        golden_mode="python",
        precision="quant",
        seed=42,
    )

    # 清空引擎
    clear()
    engine = get_engine()

    # 定义模型: linear -> relu -> softmax
    print("\n执行模型: linear -> relu -> softmax")

    # 生成输入
    x = generate_random(shape=(2, 4, 64), qtype="bfp8", seed=42)
    w = generate_weight(shape=(64, 128), qtype="bfp8", seed=43, init="xavier")

    print(f"  输入 x: {x}")
    print(f"  权重 w: {w}")

    # 执行 linear (使用引擎)
    y1 = engine.run_op("linear", inputs=[x], weights=[w], qtype="bfp8")
    print(f"  linear 输出: {y1}")

    # 执行 relu
    y2 = engine.run_op("relu", inputs=[y1], qtype="bfp8")
    print(f"  relu 输出: {y2}")

    # 执行 softmax
    y3 = engine.run_op("softmax", inputs=[y2], qtype="bfp8")
    print(f"  softmax 输出: {y3}")

    # 查看记录
    print(f"\n执行记录: {len(engine.get_records())} 条")
    for r in engine.get_records():
        print(f"  {r.op_name}_{r.id}: qtype={r.qtype}, output_shape={r.output.shape}")

    # 模拟 DUT 输出并比对
    print("\n模拟 DUT 比对:")
    results = []
    for r in engine.get_records():
        # 模拟 DUT 输出 (加点小噪声，在容差范围内)
        noise = np.random.randn(*r.golden_quant.shape).astype(np.float32) * 1e-7
        dut_output = r.golden_quant + noise

        config = get_config()
        cmp_result = compare_3col(
            op_name=r.op_name,
            op_id=r.id,
            result=dut_output,
            golden_pure=r.golden_pure,
            golden_qnt=r.golden_quant,
            exact_max_abs=config.exact.max_abs,
            fuzzy_atol=config.fuzzy.atol,
            fuzzy_rtol=config.fuzzy.rtol,
            fuzzy_min_qsnr=config.fuzzy.min_qsnr,
            fuzzy_min_cosine=config.fuzzy.min_cosine,
        )
        results.append(cmp_result)

    print_compare_table(results)


def main():
    print()
    print("*" * 60)
    print("*          统一工作流 Demo                               *")
    print("*" * 60)

    demo_config()
    demo_tensor()
    demo_ops()
    demo_compare()
    demo_full_workflow()

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
