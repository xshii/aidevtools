"""算子 API

推荐用法 - 通过 PyTorch 劫持:
    import aidevtools.golden  # 导入即启用劫持

    import torch.nn.functional as F
    y = F.linear(x, w)  # 自动走 golden

工具函数:
    from aidevtools import ops

    ops.seed(42)       # 设置随机种子
    ops.clear()        # 清空记录
    ops.dump("./out")  # 导出数据

高级用法:
    from aidevtools.ops import set_golden_mode, get_profiles

    set_golden_mode("cpp")  # 使用 C++ golden
    profiles = get_profiles()  # 获取性能数据
"""
# 工具函数
from aidevtools.ops.auto import get_seed, seed
from aidevtools.ops.quantized_tensor import (
    QuantizedTensor,
    ensure_quantized,
    quantize,
    wrap_output,
)
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

# 内部模块，仅用于触发算子注册，不对外暴露
from aidevtools.ops import _functional as _F  # noqa: F401

__all__ = [
    # 工具函数
    "seed",
    "get_seed",
    "clear",
    "dump",
    # 量化张量
    "QuantizedTensor",
    "quantize",
    "ensure_quantized",
    "wrap_output",
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
]
