"""Golden API 模块

提供 C++ 实现的精确算子，用于比对验证。

使用方式:
    from aidevtools.golden import register_gfloat_golden

    # 注册 gfloat C++ golden 实现
    register_gfloat_golden()

    # 之后调用 gfloat 相关算子会使用 C++ 实现
"""
from aidevtools.golden.gfloat_wrapper import register_gfloat_golden

__all__ = ["register_gfloat_golden"]
