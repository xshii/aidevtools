"""AI Dev Tools

数据生成 (推荐):
    from aidevtools import DataGenerator

    gen = DataGenerator(seed=42)

    # 自动生成 (读取 @register_op 配置)
    data = gen.generate("linear", input_shape=(512, 768), out_features=3072)

    # 生成 + golden
    data, golden = gen.generate_with_golden("linear", input_shape=(512, 768), out_features=3072)

    # 手动生成
    x = gen.randn((512, 768), name="x")
    w = gen.xavier((3072, 768), name="weight")

    # 导出 DUT
    gen.export("./golden/")

PyTorch 劫持:
    import aidevtools.golden
    import torch.nn.functional as F

    y = F.linear(x, w)  # 自动走 golden

比对模块:
    from aidevtools.compare import compare_full, CompareStatus

    result = compare_full(dut, golden_pure, golden_qnt)

优化器:
    from aidevtools.optimizer import extract_benchmark, FusionEvaluator

    bm = extract_benchmark("my_model")
    result = FusionEvaluator().evaluate(bm)
"""
__version__ = "0.1.0"

# 统一数据生成器 (推荐入口)
from aidevtools.datagen import (
    DataGenerator, GeneratedTensor, FourTrackGolden, Model, ModelTensor,
)

# 精度配置
from aidevtools.frontend.types import PrecisionConfig

# 模块级导入
from aidevtools import compare, frontend, ops

# 便捷导出工具函数
from aidevtools.ops import clear, dump, seed

__all__ = [
    # 数据生成
    "DataGenerator",
    "GeneratedTensor",
    "FourTrackGolden",
    # 精度配置
    "PrecisionConfig",
    # Model DSL
    "Model",
    "ModelTensor",
    # 模块
    "ops",
    "compare",
    "frontend",
    # 便捷函数
    "seed",
    "clear",
    "dump",
]
