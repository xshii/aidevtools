"""量化类型注册表 (内部模块)

将注册表独立出来避免循环导入：
- quantize.py 使用这里的注册函数
- gfloat/bfp golden.py 使用这里的装饰器注册
"""
from typing import Callable, Dict

# 量化类型注册表
_quantize_registry: Dict[str, Callable] = {}


def register_quantize(name: str):
    """
    注册量化转换函数

    示例:
        @register_quantize("int8_symmetric")
        def to_int8_symmetric(data: np.ndarray, **kwargs) -> np.ndarray:
            scale = np.max(np.abs(data)) / 127
            return np.round(data / scale).astype(np.int8), {"scale": scale}
    """

    def decorator(func: Callable):
        _quantize_registry[name] = func
        return func

    return decorator


def get_quantize(name: str) -> Callable:
    """获取量化函数"""
    if name not in _quantize_registry:
        raise ValueError(f"未知量化类型: {name}, 可用: {list(_quantize_registry.keys())}")
    return _quantize_registry[name]


def list_quantize() -> list:
    """列出所有注册的量化类型"""
    return list(_quantize_registry.keys())
