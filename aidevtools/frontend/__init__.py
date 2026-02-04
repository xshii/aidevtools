"""
前端模块

提供统一的 Python/C 前端 API。

基本使用:
    from aidevtools.frontend import (
        Tensor,
        DataGenerator,
        OpContext,
        compile_to_dut,
    )

    # 数据生成
    gen = DataGenerator(seed=42)
    x = gen.gen_input(shape=(2, 64), dtype="bfp16", dist="normal")
    w = gen.gen_weight(shape=(64, 64), dtype="bfp16", init="xavier")

    # 编译
    result = compile_to_dut(
        source="model.py",
        output="build/model.bin",
        golden_dir="golden/",
    )
"""

from .types import (
    CompileConfig,
    DistType,
    DType,
    OpContext,
    Tensor,
    TensorMeta,
)
from .datagen import DataGenerator
from .compile import (
    Compiler,
    CompileError,
    compile_to_dut,
)

__all__ = [
    # 类型
    "Tensor",
    "TensorMeta",
    "DType",
    "DistType",
    "OpContext",
    "CompileConfig",
    # 数据生成
    "DataGenerator",
    # 编译
    "Compiler",
    "CompileError",
    "compile_to_dut",
]
