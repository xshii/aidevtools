"""
Demo: ML 校准流程

展示如何导入实测数据并校准超参数
"""

from aidevtools.optimizer import (
    Benchmark,
    BenchmarkSuite,
    MeasurementArchive,
    MeasurementSource,
    HyperCalibrator,
    OptimizeMethod,
    get_fusion_rules,
)


def demo_import_results():
    """演示导入实测结果 (最简接口)"""
    print("=" * 60)
    print("1. 导入实测结果")
    print("=" * 60)

    # 创建归档
    archive = MeasurementArchive()

    # 定义 benchmarks
    benchmarks = {
        "mm_512": Benchmark("mm_512").add_op("mm", "matmul", M=512, N=768, K=768),
        "mm_gelu_512": (
            Benchmark("mm_gelu_512")
            .add_op("mm", "matmul", M=512, N=768, K=768)
            .add_op("gelu", "gelu", M=512, N=768)
        ),
        "ffn_512": (
            Benchmark("ffn_512")
            .add_op("mm1", "matmul", M=512, N=3072, K=768)
            .add_op("gelu", "gelu", M=512, N=3072)
            .add_op("mm2", "matmul", M=512, N=768, K=3072)
        ),
    }

    # 模拟实测结果: {bm_name: latency_us}
    results = {
        "mm_512": 125.5,
        "mm_gelu_512": 98.2,  # 融合后更快
        "ffn_512": 380.0,
    }

    # 导入
    count = archive.import_from_benchmarks(results, benchmarks)
    print(f"导入了 {count} 条记录")

    # 查看统计
    stats = archive.statistics()
    print(f"\n统计信息:")
    print(f"  - 样本数: {stats['count']}")
    print(f"  - 融合样本: {stats['fused_count']}")
    print(f"  - 未融合样本: {stats['unfused_count']}")
    print(f"  - 时延范围: {stats['latency_us_min']:.1f} ~ {stats['latency_us_max']:.1f} us")

    return archive


def demo_import_with_suite():
    """演示使用 BenchmarkSuite 导入"""
    print("\n" + "=" * 60)
    print("2. 使用 BenchmarkSuite 导入")
    print("=" * 60)

    archive = MeasurementArchive()
    suite = BenchmarkSuite()

    # 只需提供 (bm_name, latency_us) 列表
    results = [
        ("bert_ffn_512", 125.5),
        ("bert_ffn_1024", 245.0),
        ("gpt_attention_512", 180.3),
    ]

    count = archive.import_results(results, suite)
    print(f"导入了 {count} 条记录")

    return archive


def demo_calibration(archive: MeasurementArchive):
    """演示超参数校准"""
    print("\n" + "=" * 60)
    print("3. 超参数校准")
    print("=" * 60)

    # 创建校准器
    calibrator = HyperCalibrator(archive)

    # 查看当前超参数
    rules = get_fusion_rules()
    print("当前超参数 (部分):")
    params = rules.hyper_params
    print(f"  - decay_base: {params.decay_base}")
    print(f"  - speedup_scale: {params.speedup_scale}")
    print(f"  - op_submit_base: {params.op_submit_base}")

    # 执行校准 (使用随机搜索，快速演示)
    print("\n开始校准 (Random Search, 20 iterations)...")
    result = calibrator.calibrate(
        method=OptimizeMethod.RANDOM_SEARCH,
        max_iterations=20,
        train_ratio=0.8,
    )

    print(f"\n校准结果:")
    print(f"  - 方法: {result.method.value}")
    print(f"  - 训练损失: {result.train_loss:.4f}")
    print(f"  - 验证损失: {result.val_loss:.4f}" if result.val_loss else "  - 验证损失: N/A")
    print(f"  - R²: {result.r_squared:.4f}")
    print(f"  - RMSE: {result.rmse:.2f} us")
    print(f"  - MAPE: {result.mape:.2f}%")
    print(f"  - 改进: {result.improvement():.1f}%")
    print(f"  - 耗时: {result.duration_seconds:.2f}s")

    # 应用结果
    if calibrator.apply(result, validate=True):
        print("\n新超参数已应用!")
        new_params = rules.hyper_params
        print(f"  - decay_base: {new_params.decay_base:.4f}")
        print(f"  - speedup_scale: {new_params.speedup_scale:.4f}")

    return result


def demo_export_import():
    """演示数据导出和导入"""
    print("\n" + "=" * 60)
    print("4. 数据导出/导入")
    print("=" * 60)

    archive = MeasurementArchive()

    # 添加一些数据
    benchmarks = {
        "test1": Benchmark("test1").add_op("mm", "matmul", M=256, N=256, K=256),
        "test2": Benchmark("test2").add_op("mm", "matmul", M=512, N=512, K=512),
    }
    archive.import_from_benchmarks({"test1": 50.0, "test2": 200.0}, benchmarks)

    # 导出为 numpy
    X, y, feature_names, sample_ids = archive.to_numpy()
    print(f"Numpy 格式:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - 特征名: {feature_names[:5]}...")

    # 导出为 pandas
    df = archive.to_pandas()
    print(f"\nPandas DataFrame:")
    print(f"  - shape: {df.shape}")
    print(f"  - columns: {list(df.columns)[:5]}...")

    # 保存到 JSON
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "archive.json")
        archive.save(json_path)
        print(f"\n已保存到: {json_path}")

        # 重新加载
        archive2 = MeasurementArchive(json_path)
        print(f"重新加载: {len(archive2)} 条记录")


if __name__ == "__main__":
    # 1. 导入实测结果
    archive = demo_import_results()

    # 2. 使用 Suite 导入
    demo_import_with_suite()

    # 3. 校准超参数
    demo_calibration(archive)

    # 4. 导出/导入
    demo_export_import()

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
