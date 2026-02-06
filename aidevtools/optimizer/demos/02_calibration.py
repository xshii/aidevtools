"""
Demo: ML 校准流程

展示如何使用 PyTorch 生成测试用例，导入实测数据并校准超参数

流程:
1. 使用 PyTorch 定义模型结构
2. 提取 Benchmark 并生成测试数据
3. 在 DUT 上运行获取实测时延
4. 导入实测结果到 MeasurementArchive
5. 使用 HyperCalibrator 校准超参数
"""

import torch
import torch.nn.functional as F

import aidevtools.golden as golden
from aidevtools.ops import clear as ops_clear
from aidevtools.optimizer import (
    extract_benchmark,
    BenchmarkSuite,
    MeasurementArchive,
    HyperCalibrator,
    OptimizeMethod,
    get_fusion_rules,
)


def demo_generate_test_cases():
    """演示用 PyTorch 生成测试用例"""
    print("=" * 60)
    print("1. 使用 PyTorch 生成测试用例")
    print("=" * 60)

    test_cases = []

    # 测试用例 1: 单个 matmul
    ops_clear()
    x = torch.randn(512, 768)
    w = torch.randn(768, 768)
    y = torch.matmul(x, w)
    bm1 = extract_benchmark("matmul_512")
    test_cases.append(("matmul_512", bm1))
    print(f"生成: matmul_512, ops={len(bm1.ops)}")

    # 测试用例 2: matmul + gelu
    ops_clear()
    x = torch.randn(512, 768)
    w = torch.randn(768, 768)
    y = torch.matmul(x, w)
    y = F.gelu(y)
    bm2 = extract_benchmark("matmul_gelu_512")
    test_cases.append(("matmul_gelu_512", bm2))
    print(f"生成: matmul_gelu_512, ops={len(bm2.ops)}")

    # 测试用例 3: FFN
    ops_clear()
    x = torch.randn(512, 768)
    w1 = torch.randn(3072, 768)
    w2 = torch.randn(768, 3072)
    y = F.linear(x, w1)
    y = F.gelu(y)
    y = F.linear(y, w2)
    bm3 = extract_benchmark("ffn_512")
    test_cases.append(("ffn_512", bm3))
    print(f"生成: ffn_512, ops={len(bm3.ops)}")

    print(f"\n总共 {len(test_cases)} 个测试用例")
    return test_cases


def demo_export_for_dut(test_cases):
    """演示导出数据用于 DUT 测试"""
    print("\n" + "=" * 60)
    print("2. 导出数据用于 DUT 测试")
    print("=" * 60)

    from aidevtools.ops import OpDataGenerator

    print("使用 OpDataGenerator 生成 DUT 测试数据:")

    gen = OpDataGenerator(seed=42, l2_base=0x100000, alignment=256)

    for name, bm in test_cases:
        print(f"\n{name}:")
        for op in bm.ops:
            # 从 Benchmark 的 OpSpec 提取 shape 信息
            shapes = op.shapes
            M = shapes.get("M", 1)
            N = shapes.get("N", 1)
            K = shapes.get("K", N)

            if op.op_type.value == "matmul":
                data = gen.generate("matmul", input_shape=(M, K), out_features=N)
            elif op.op_type.value == "gelu":
                data = gen.generate("gelu", input_shape=(M, N))
            else:
                continue

            for param, info in data.items():
                print(f"    {param}: L2=0x{info.l2_addr:X}, shape={info.shape}")

    print(f"\n总 L2 内存: {gen.memory_layout().total_size / 1024:.1f} KB")
    print("\n实际使用时调用 gen.export_dut('golden/') 导出")


def demo_import_results():
    """演示导入实测结果"""
    print("\n" + "=" * 60)
    print("3. 导入实测结果")
    print("=" * 60)

    archive = MeasurementArchive()
    suite = BenchmarkSuite()

    # 模拟从 DUT 获取的实测结果
    # 实际使用中，这些数据来自 DUT 运行
    measured_results = [
        ("bert_ffn_512", 125.5),
        ("bert_ffn_1024", 245.0),
        ("gpt_attention_512", 180.3),
        ("gpt_attention_1024", 350.2),
    ]

    count = archive.import_results(measured_results, suite)
    print(f"导入了 {count} 条记录")

    stats = archive.statistics()
    print(f"\n统计信息:")
    print(f"  - 样本数: {stats['count']}")
    print(f"  - 时延范围: {stats['latency_us_min']:.1f} ~ {stats['latency_us_max']:.1f} us")

    return archive


def demo_calibration(archive: MeasurementArchive):
    """演示超参数校准"""
    print("\n" + "=" * 60)
    print("4. 超参数校准")
    print("=" * 60)

    calibrator = HyperCalibrator(archive)

    # 查看当前超参数
    rules = get_fusion_rules()
    print("当前超参数:")
    params = rules.hyper_params
    print(f"  - decay_base: {params.decay_base}")
    print(f"  - speedup_scale: {params.speedup_scale}")

    # 执行校准
    print("\n开始校准 (Random Search, 20 iterations)...")
    result = calibrator.calibrate(
        method=OptimizeMethod.RANDOM_SEARCH,
        max_iterations=20,
        train_ratio=0.8,
    )

    print(f"\n校准结果:")
    print(f"  - R²: {result.r_squared:.4f}")
    print(f"  - RMSE: {result.rmse:.2f} us")
    print(f"  - MAPE: {result.mape:.2f}%")
    print(f"  - 改进: {result.improvement():.1f}%")

    return result


def demo_workflow():
    """演示完整工作流"""
    print("\n" + "=" * 60)
    print("5. 完整工作流")
    print("=" * 60)

    print("""
完整工作流:

1. 用 PyTorch 定义模型:
   import torch.nn.functional as F
   import aidevtools.golden as golden

   y = F.linear(x, w1)
   y = F.gelu(y)
   y = F.linear(y, w2)

2. 提取 Benchmark:
   bm = extract_benchmark("my_ffn")

3. 生成 DUT 测试数据:
   gen = OpDataGenerator(seed=42)
   gen.export_dut("golden/")

4. DUT 运行，获取实测时延

5. 导入实测数据并校准:
   archive = MeasurementArchive()
   archive.import_results(results, suite)
   calibrator = HyperCalibrator(archive)
   result = calibrator.calibrate()
   calibrator.apply(result)

6. 使用校准后的模型预测:
   evaluator = FusionEvaluator()
   prediction = evaluator.evaluate(bm)
""")


if __name__ == "__main__":
    # 1. 生成测试用例
    test_cases = demo_generate_test_cases()

    # 2. 导出数据
    demo_export_for_dut(test_cases)

    # 3. 导入实测结果
    archive = demo_import_results()

    # 4. 校准
    demo_calibration(archive)

    # 5. 完整工作流
    demo_workflow()

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
