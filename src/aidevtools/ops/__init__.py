"""算子 API

两种使用方式:

1. PyTorch 风格 functional API (推荐):
    from aidevtools import F

    y = F.linear(x, weight, bias)
    y = F.relu(y)
    y = F.softmax(y, dim=-1)
    y = F.layer_norm(y, normalized_shape, weight, bias)

2. 简化 API (自动生成测试数据):
    from aidevtools import ops

    ops.seed(42)
    ops.clear()

    y = ops.matmul((2, 8, 64), (64, 32), dtype="bfp8")
    y = ops.layernorm(y, dtype="bfp8")
    y = ops.softmax(y, dtype="bfp8")

    ops.dump("./workspace")

高级用法:
    from aidevtools.ops import set_golden_mode, get_profiles

    set_golden_mode("cpp")  # 使用 C++ golden
    # ... 执行算子 ...
    profiles = get_profiles()  # 获取性能数据
"""
# 基础 API
from aidevtools.ops.base import (
    Op,
    register_golden_cpp,
    has_golden_cpp,
    get_records,
    clear,
    dump,
    set_golden_mode,
    get_golden_mode,
    set_compute_golden,
    get_compute_golden,
    # Profile API (用于 Paper Analysis)
    get_profiles,
    set_profile_enabled,
    get_profile_enabled,
    set_profile_only,
    get_profile_only,
    profile_only,  # 上下文管理器
)

# 工具函数
from aidevtools.ops.auto import seed, get_seed

# 简化 API (shape-based, 自动生成数据)
from aidevtools.ops.auto import (
    matmul,
    linear,
    layernorm,
    softmax,
    relu,
    gelu,
    attention,
    embedding,
    transpose,
)

# PyTorch 风格 functional API
from aidevtools.ops import functional

# 导入 nn 以触发算子注册
from aidevtools.ops import nn

__all__ = [
    # 工具函数
    "seed",
    "get_seed",
    "clear",
    "dump",
    # 配置
    "set_golden_mode",
    "get_golden_mode",
    "set_compute_golden",
    "get_compute_golden",
    "get_records",
    "get_profiles",
    "set_profile_enabled",
    "get_profile_enabled",
    "set_profile_only",
    "get_profile_only",
    "profile_only",
    # 高级
    "Op",
    "register_golden_cpp",
    "has_golden_cpp",
    # 简化 API
    "matmul",
    "linear",
    "layernorm",
    "softmax",
    "relu",
    "gelu",
    "attention",
    "embedding",
    "transpose",
    # functional API
    "functional",
]
