"""算子工具函数

提供 seed, clear, dump 等工具函数。

算子 API 请使用 PyTorch 风格:
    from aidevtools import F

    y = F.linear(x, weight, bias)
    y = F.relu(y)
    y = F.softmax(y, dim=-1)
"""
import numpy as np

from aidevtools.ops.base import (
    clear as _clear,
)
from aidevtools.ops.base import (
    dump as _dump,
)

# 全局随机种子
_seed: int = 42


def seed(s: int) -> None:
    """设置随机种子"""
    global _seed
    _seed = s
    np.random.seed(s)


def get_seed() -> int:
    """获取当前种子值"""
    return _seed


def clear() -> None:
    """清空记录"""
    _clear()


def dump(output_dir: str = "./workspace", fmt: str = "raw") -> None:
    """导出所有 bin 文件"""
    _dump(output_dir, fmt=fmt)
