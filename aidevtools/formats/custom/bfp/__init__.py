"""BFP 真实格式模块 (待实现)

提供 BFP 格式注册 (stub) 和 C++ Golden 包装器。
"""
from aidevtools.formats.custom.bfp import golden
from aidevtools.formats.custom.bfp.wrapper import is_cpp_available, register_bfp_golden

__all__ = ["golden", "register_bfp_golden", "is_cpp_available"]
