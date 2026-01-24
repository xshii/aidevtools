"""算子 API

PyTorch 风格 functional API:
    from aidevtools import F

    y = F.linear(x, weight, bias)
    y = F.relu(y)
    y = F.softmax(y, dim=-1)
    y = F.layer_norm(y, normalized_shape, weight, bias)
    y = F.scaled_dot_product_attention(q, k, v)

工具函数:
    from aidevtools import ops

    ops.seed(42)       # 设置随机种子
    ops.clear()        # 清空记录
    # ... 执行算子 ...
    ops.dump("./out")  # 导出数据

高级用法:
    from aidevtools.ops import set_golden_mode, get_profiles

    set_golden_mode("cpp")  # 使用 C++ golden
    profiles = get_profiles()  # 获取性能数据
"""
# 基础 API
# PyTorch 风格 functional API
# 导入 nn 以触发算子注册
from aidevtools.ops import functional, nn

# 工具函数
from aidevtools.ops.auto import get_seed, seed
from aidevtools.ops.base import (
    Op,
    clear,
    dump,
    get_compute_golden,
    get_golden_mode,
    get_profile_enabled,
    get_profile_only,
    # Profile API
    get_profiles,
    get_records,
    has_golden_cpp,
    profile_only,
    register_golden_cpp,
    set_compute_golden,
    set_golden_mode,
    set_profile_enabled,
    set_profile_only,
)

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
    # functional API
    "functional",
]
