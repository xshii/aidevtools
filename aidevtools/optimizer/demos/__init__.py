"""
Optimizer Demos

演示脚本列表:
    01_basic.py       - 基础用法 (PyTorch 劫持、Benchmark 提取、评估)
    02_calibration.py - ML 校准流程 (生成测试用例、导入实测、校准)
    03_fusion_rules.py - 融合规则配置 (全局规则、多算子模式)
    04_echarts.py     - ECharts 可视化 (图表生成)
    05_comparison.py  - 理论 vs 工程化对比
    06_bridge.py      - PyTorch 劫持 → Benchmark 桥接

运行方式:
    python -m aidevtools.optimizer.demos.01_basic
    python -m aidevtools.optimizer.demos.02_calibration
    python -m aidevtools.optimizer.demos.03_fusion_rules
    python -m aidevtools.optimizer.demos.04_echarts
    python -m aidevtools.optimizer.demos.05_comparison
    python -m aidevtools.optimizer.demos.06_bridge

推荐使用方式 (PyTorch 劫持):

    import torch
    import torch.nn.functional as F
    import aidevtools.golden as golden
    from aidevtools.optimizer import extract_benchmark, extract_and_evaluate
    from aidevtools.ops import clear as ops_clear

    # 清空计算图
    ops_clear()

    # 执行 PyTorch 代码 (自动被劫持)
    x = torch.randn(512, 768)
    w1 = torch.randn(3072, 768)
    w2 = torch.randn(768, 3072)

    y = F.linear(x, w1)
    y = F.gelu(y)
    y = F.linear(y, w2)

    # 自动提取 Benchmark
    bm = extract_benchmark("my_ffn")
    print(bm.summary())

    # 或一键提取 + 评估
    result = extract_and_evaluate("my_ffn")
    print(result.summary())

数据生成 (用于 DUT 测试):

    from aidevtools.ops import OpDataGenerator

    gen = OpDataGenerator(seed=42, l2_base=0x100000, alignment=256)
    data = gen.generate("linear", input_shape=(512, 768), out_features=3072)

    # 导出
    gen.export_dut("golden/")
    gen.export_header("golden/layout.h")
"""
