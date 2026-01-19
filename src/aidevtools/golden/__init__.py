"""Golden API

提供 CPU Golden 实现，通过 subprocess 调用 C++ 可执行文件。

用法:
    # 方式1: 直接调用
    from aidevtools.golden.cpu_ops import matmul, softmax, layernorm
    c = matmul(a, b, dtype="gfp16")

    # 方式2: 注册到 ops 框架
    from aidevtools.golden.cpu_ops import register_all_cpu_golden
    from aidevtools.ops import set_golden_mode

    register_all_cpu_golden("gfp16")
    set_golden_mode("cpp")

    # 之后调用 ops.nn.matmul 会自动使用 cpu_golden
"""
from aidevtools.golden.cpu_ops import (
    matmul,
    softmax,
    layernorm,
    transpose,
    register_all_cpu_golden,
    is_cpu_golden_available,
)

__all__ = [
    "matmul",
    "softmax",
    "layernorm",
    "transpose",
    "register_all_cpu_golden",
    "is_cpu_golden_available",
]
