"""BFPP (Block Floating Point Python) Golden 模块

提供 BFPP 格式的 Python Golden 实现。

使用方式:
    from aidevtools.formats.custom.bfpp import golden
    mantissas, exps, meta = golden.fp32_to_bfp(data)

C++ Golden 包装器在 aidevtools.formats.custom.bfp 中。
"""
from aidevtools.formats.custom.bfpp import golden

__all__ = ["golden"]
