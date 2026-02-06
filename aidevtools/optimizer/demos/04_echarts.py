"""
Demo: ECharts 可视化

展示如何将分析结果转换为 ECharts 图表
"""

from aidevtools.optimizer import (
    Benchmark,
    FusionEvaluator,
    get_fusion_rules,
)
from aidevtools.optimizer.views import (
    EChartsConverter,
    ChartType,
    to_echarts,
)


def demo_bar_chart():
    """演示柱状图"""
    print("=" * 60)
    print("1. 柱状图 - 策略比较")
    print("=" * 60)

    chart = EChartsConverter.bar_chart(
        x_data=["Baseline", "EfficiencyAware", "FuseSpeedup"],
        series_data={
            "Compute Cycles": [10000, 8500, 7200],
            "Memory Cycles": [5000, 4200, 3800],
        },
        title="Strategy Comparison",
        x_label="Strategy",
        y_label="Cycles",
        stack=True,
    )

    print(f"JSON 配置 (前 500 字符):")
    print(chart.to_json()[:500] + "...")

    # 保存为 HTML
    # chart.to_html() 可以生成完整的 HTML 页面


def demo_pie_chart():
    """演示饼图"""
    print("\n" + "=" * 60)
    print("2. 饼图 - 时间分布")
    print("=" * 60)

    chart = EChartsConverter.pie_chart(
        data={
            "Compute": 45,
            "Memory": 30,
            "Overhead": 15,
            "DMA": 10,
        },
        title="Cycle Distribution",
        radius=["40%", "70%"],  # 环形图
    )

    print(f"生成饼图配置成功")
    print(f"  - 数据项: 4 个")
    print(f"  - 类型: 环形图")


def demo_line_chart():
    """演示折线图"""
    print("\n" + "=" * 60)
    print("3. 折线图 - 性能趋势")
    print("=" * 60)

    chart = EChartsConverter.line_chart(
        x_data=["256", "512", "1024", "2048", "4096"],
        series_data={
            "Fused": [50, 98, 195, 380, 750],
            "Unfused": [80, 155, 310, 620, 1240],
        },
        title="Latency vs Batch Size",
        x_label="Batch Size",
        y_label="Latency (us)",
        smooth=True,
        area=False,
    )

    print(f"生成折线图配置成功")
    print(f"  - X 轴: Batch Size")
    print(f"  - 系列: Fused, Unfused")


def demo_roofline():
    """演示 Roofline 图"""
    print("\n" + "=" * 60)
    print("4. Roofline 图")
    print("=" * 60)

    chart = EChartsConverter.roofline_chart(
        peak_compute=100.0,  # 100 GFLOPS
        peak_bandwidth=50.0,  # 50 GB/s
        points=[
            {"name": "MatMul_512", "ai": 10.0, "perf": 80.0},
            {"name": "GELU_512", "ai": 0.5, "perf": 25.0},
            {"name": "LayerNorm", "ai": 0.2, "perf": 10.0},
        ],
        title="Roofline Model",
    )

    print(f"生成 Roofline 图配置成功")
    print(f"  - 峰值算力: 100 GFLOPS")
    print(f"  - 峰值带宽: 50 GB/s")
    print(f"  - 数据点: 3 个")


def demo_gauge():
    """演示仪表盘"""
    print("\n" + "=" * 60)
    print("5. 仪表盘 - 利用率")
    print("=" * 60)

    chart = EChartsConverter.gauge_chart(
        value=78.5,
        max_value=100,
        title="Compute Utilization",
        unit="%",
    )

    print(f"生成仪表盘配置成功")
    print(f"  - 当前值: 78.5%")


def demo_heatmap():
    """演示热力图"""
    print("\n" + "=" * 60)
    print("6. 热力图 - Tile 效率")
    print("=" * 60)

    chart = EChartsConverter.heatmap_chart(
        x_data=["M=64", "M=128", "M=256", "M=512"],
        y_data=["N=64", "N=128", "N=256", "N=512"],
        values=[
            [0.65, 0.72, 0.78, 0.82],
            [0.70, 0.75, 0.82, 0.88],
            [0.75, 0.80, 0.86, 0.92],
            [0.78, 0.84, 0.90, 0.95],
        ],
        title="Tile Efficiency Matrix",
    )

    print(f"生成热力图配置成功")
    print(f"  - 矩阵: 4x4")


def demo_from_evaluator():
    """演示从评估器结果生成图表"""
    print("\n" + "=" * 60)
    print("7. 从评估器结果生成图表")
    print("=" * 60)

    # 创建评估器
    evaluator = FusionEvaluator()

    # 创建 benchmark
    bm = (
        Benchmark("demo_ffn")
        .add_op("mm1", "matmul", M=512, N=3072, K=768)
        .add_op("gelu", "gelu", M=512, N=3072)
        .add_op("mm2", "matmul", M=512, N=768, K=3072)
    )

    # 评估
    result = evaluator.evaluate(bm)

    # 从结果生成柱状图
    chart = EChartsConverter.from_tiling_result(
        result.tiling_result,
        title="FFN Operator Breakdown",
    )

    print(f"从 TilingResult 生成图表成功")
    print(f"  - 算子数: {len(result.tiling_result.op_results)}")


def demo_save_html():
    """演示保存为 HTML"""
    print("\n" + "=" * 60)
    print("8. 保存为 HTML")
    print("=" * 60)

    chart = EChartsConverter.bar_chart(
        x_data=["A", "B", "C"],
        series_data={"Value": [10, 20, 30]},
        title="Demo Chart",
    )

    html = chart.to_html(width="800px", height="400px")

    print(f"生成 HTML 成功")
    print(f"  - 大小: {len(html)} 字符")
    print(f"  - 包含 ECharts CDN 引用")

    # 保存示例 (注释掉)
    # with open("demo_chart.html", "w") as f:
    #     f.write(html)


if __name__ == "__main__":
    demo_bar_chart()
    demo_pie_chart()
    demo_line_chart()
    demo_roofline()
    demo_gauge()
    demo_heatmap()
    demo_from_evaluator()
    demo_save_html()

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
