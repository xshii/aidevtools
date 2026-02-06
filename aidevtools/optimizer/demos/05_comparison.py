"""
Demo: 理论分析 vs 工程化方法对比

展示如何对比两种预测方法的精度

工作流:
1. 理论分析 = 使用默认/优化前的超参数
2. 工程化 = 使用 ML 校准后的超参数
3. 自动对比并输出校准后的时延预测
"""

from aidevtools.optimizer import (
    Benchmark,
    MeasurementArchive,
    compare_methods,
    calibrate_and_compare,
    MethodComparator,
    PredictMethod,
)


def demo_simple_compare():
    """演示简化对比接口"""
    print("=" * 70)
    print("1. 简化对比接口")
    print("=" * 70)

    # 模拟数据: 5 个 benchmark 的预测和实测结果
    theoretical = {
        "matmul_512": 120.0,
        "matmul_1024": 450.0,
        "ffn_512": 280.0,
        "ffn_1024": 1050.0,
        "attention_512": 350.0,
    }

    empirical = {
        "matmul_512": 115.0,
        "matmul_1024": 430.0,
        "ffn_512": 265.0,
        "ffn_1024": 980.0,
        "attention_512": 340.0,
    }

    actual = {
        "matmul_512": 118.0,
        "matmul_1024": 440.0,
        "ffn_512": 270.0,
        "ffn_1024": 1000.0,
        "attention_512": 345.0,
    }

    # 对比
    result = compare_methods(theoretical, empirical, actual)
    print(result.summary())


def demo_detailed_analysis():
    """演示详细分析"""
    print("\n" + "=" * 70)
    print("2. 详细分析")
    print("=" * 70)

    # 更多样本
    benchmarks = [
        "mm_256", "mm_512", "mm_1024", "mm_2048",
        "gelu_256", "gelu_512", "gelu_1024",
        "ffn_256", "ffn_512", "ffn_1024",
    ]

    # 理论分析: 倾向于略微高估 (保守)
    theoretical = {
        "mm_256": 32.0, "mm_512": 125.0, "mm_1024": 480.0, "mm_2048": 1900.0,
        "gelu_256": 8.0, "gelu_512": 30.0, "gelu_1024": 120.0,
        "ffn_256": 75.0, "ffn_512": 290.0, "ffn_1024": 1100.0,
    }

    # 工程化: 经过 ML 校准，更接近实测
    empirical = {
        "mm_256": 28.0, "mm_512": 112.0, "mm_1024": 445.0, "mm_2048": 1780.0,
        "gelu_256": 7.5, "gelu_512": 28.0, "gelu_1024": 110.0,
        "ffn_256": 68.0, "ffn_512": 268.0, "ffn_1024": 1020.0,
    }

    # 实测
    actual = {
        "mm_256": 30.0, "mm_512": 115.0, "mm_1024": 450.0, "mm_2048": 1800.0,
        "gelu_256": 7.8, "gelu_512": 29.0, "gelu_1024": 112.0,
        "ffn_256": 70.0, "ffn_512": 275.0, "ffn_1024": 1050.0,
    }

    result = compare_methods(theoretical, empirical, actual)

    # 查看详细结果
    print("\n各 Benchmark 详情:")
    print(f"{'Benchmark':<15} {'实测':>10} {'理论':>10} {'工程化':>10} {'理论误差%':>12} {'工程化误差%':>12}")
    print("-" * 75)

    for detail in result.details:
        bm = detail["benchmark"]
        act = detail["actual_us"]
        theo = detail["theoretical_us"]
        emp = detail["empirical_us"]
        t_err = detail["theoretical_error_pct"]
        e_err = detail["empirical_error_pct"]

        winner = "←" if t_err < e_err else "→" if e_err < t_err else "="
        print(f"{bm:<15} {act:>10.1f} {theo:>10.1f} {emp:>10.1f} {t_err:>11.1f}% {e_err:>11.1f}% {winner}")

    print("\n指标汇总:")
    print(result.summary())


def demo_scenario_analysis():
    """演示场景分析"""
    print("\n" + "=" * 70)
    print("3. 场景分析")
    print("=" * 70)

    # 不同场景的对比
    scenarios = {
        "小 Shape (计算密集)": {
            "theoretical": {"bm1": 50, "bm2": 55, "bm3": 52},
            "empirical": {"bm1": 48, "bm2": 52, "bm3": 50},
            "actual": {"bm1": 49, "bm2": 53, "bm3": 51},
        },
        "大 Shape (访存密集)": {
            "theoretical": {"bm1": 500, "bm2": 520, "bm3": 510},
            "empirical": {"bm1": 480, "bm2": 495, "bm3": 490},
            "actual": {"bm1": 490, "bm2": 505, "bm3": 498},
        },
        "融合算子": {
            "theoretical": {"bm1": 200, "bm2": 210, "bm3": 195},
            "empirical": {"bm1": 175, "bm2": 182, "bm3": 170},
            "actual": {"bm1": 180, "bm2": 188, "bm3": 175},
        },
    }

    print(f"{'场景':<25} {'理论MAPE%':>12} {'工程化MAPE%':>12} {'更优':>10}")
    print("-" * 65)

    for scenario_name, data in scenarios.items():
        result = compare_methods(
            data["theoretical"],
            data["empirical"],
            data["actual"],
        )
        t_mape = result.theoretical.mape
        e_mape = result.empirical.mape
        winner = "理论" if t_mape < e_mape else "工程化" if e_mape < t_mape else "相同"

        print(f"{scenario_name:<25} {t_mape:>12.2f} {e_mape:>12.2f} {winner:>10}")

    print("\n结论:")
    print("  - 小 Shape: 两种方法接近")
    print("  - 大 Shape: 工程化方法更优 (底噪建模更准)")
    print("  - 融合算子: 工程化方法明显更优 (融合加速比经过校准)")


def demo_generate_echarts():
    """演示生成 ECharts 图表"""
    print("\n" + "=" * 70)
    print("4. 生成 ECharts 对比图")
    print("=" * 70)

    theoretical = {"bm1": 100, "bm2": 200, "bm3": 300}
    empirical = {"bm1": 95, "bm2": 190, "bm3": 285}
    actual = {"bm1": 98, "bm2": 195, "bm3": 290}

    result = compare_methods(theoretical, empirical, actual)

    # 生成 ECharts 配置
    chart = result.to_echarts()

    print("生成 ECharts 配置成功")
    print(f"  - 图表类型: 柱状图")
    print(f"  - 对比维度: MAE, MAPE, RMSE, Max Error%")

    # 可以保存为 HTML
    # with open("comparison.html", "w") as f:
    #     f.write(chart.to_html())


def demo_calibrate_and_compare():
    """演示一键校准并对比"""
    print("\n" + "=" * 70)
    print("5. 一键校准并对比 (完整工作流)")
    print("=" * 70)

    # 1. 准备数据
    archive = MeasurementArchive()

    benchmarks = {
        "mm_256": Benchmark("mm_256").add_op("mm", "matmul", M=256, N=256, K=256),
        "mm_512": Benchmark("mm_512").add_op("mm", "matmul", M=512, N=512, K=512),
        "mm_1024": Benchmark("mm_1024").add_op("mm", "matmul", M=1024, N=1024, K=1024),
        "gelu_512": Benchmark("gelu_512").add_op("gelu", "gelu", M=512, N=512),
        "ffn_512": (
            Benchmark("ffn_512")
            .add_op("mm1", "matmul", M=512, N=2048, K=512)
            .add_op("gelu", "gelu", M=512, N=2048)
            .add_op("mm2", "matmul", M=512, N=512, K=2048)
        ),
    }

    # 模拟实测数据
    results = {
        "mm_256": 30.0,
        "mm_512": 115.0,
        "mm_1024": 450.0,
        "gelu_512": 28.0,
        "ffn_512": 275.0,
    }

    archive.import_from_benchmarks(results, benchmarks)

    print(f"导入 {len(archive)} 条实测数据")

    # 2. 一键校准并对比
    print("\n执行校准 (Random Search, 快速演示)...")
    result = calibrate_and_compare(
        archive,
        method="random_search",
        max_iterations=20,
    )

    # 3. 查看结果
    print(result.summary())

    # 4. 校准后的预测
    print("\n校准后时延预测:")
    for bm_name, pred in result.calibrated_predictions.items():
        actual = results[bm_name]
        error_pct = abs(pred - actual) / actual * 100
        print(f"  {bm_name}: 预测={pred:.2f}us, 实测={actual:.2f}us, 误差={error_pct:.1f}%")


if __name__ == "__main__":
    demo_simple_compare()
    demo_detailed_analysis()
    demo_scenario_analysis()
    demo_generate_echarts()
    demo_calibrate_and_compare()

    print("\n" + "=" * 70)
    print("Demo 完成!")
    print("=" * 70)
