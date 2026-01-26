"""算子工具函数

提供 seed, clear, dump 等工具函数。

推荐用法 - 通过 PyTorch 劫持:
    import aidevtools.golden  # 导入即启用劫持

    import torch.nn.functional as F
    y = F.linear(x, w)  # 自动走 golden
"""
import numpy as np
import torch

from aidevtools.ops.base import (
    clear as _clear,
)
from aidevtools.ops.base import (
    dump as _dump,
)

# 全局随机种子
_seed: int = 42


def seed(s: int) -> None:
    """设置随机种子 (numpy + torch)"""
    global _seed  # pylint: disable=global-statement
    _seed = s
    np.random.seed(s)
    torch.manual_seed(s)


def get_seed() -> int:
    """获取当前种子值"""
    return _seed


def clear() -> None:
    """清空记录"""
    _clear()


def dump(output_dir: str = "./workspace", fmt: str = "raw") -> None:
    """导出所有 bin 文件"""
    _dump(output_dir, fmt=fmt)
