"""格式基类"""
from math import ceil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from aidevtools.formats._registry import get, register


class FormatBase:
    """格式基类"""

    name: str = ""

    def load(self, path: str, **kwargs) -> np.ndarray:
        """加载数据文件"""
        raise NotImplementedError

    def save(self, path: str, data: np.ndarray, **kwargs):
        """保存数据到文件"""
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            register(cls.name, cls())


# BFP 格式默认参数: {qtype: (block_size, mantissa_bits)}
_BFP_DEFAULTS = {
    "bfp16": (16, 8),
    "bfp8": (32, 4),
    "bfp4": (64, 2),
}


_KNOWN_QTYPES = set(_BFP_DEFAULTS) | {"float16", "float32"}


def _infer_qtype(path: str) -> Optional[str]:
    """从文件名后缀推断 qtype

    命名约定: {bm}_{name}_{shape}.{qtype}.bin
    示例: encoder_linear_0_weight_64x64.bfp4.bin
    """
    import os
    base = os.path.basename(path)
    if base.endswith(".bin"):
        base = base[:-4]
    if "." in base:
        suffix = base.rsplit(".", 1)[-1]
        if suffix in _KNOWN_QTYPES:
            return suffix
    return None


def _infer_shape(path: str) -> Optional[Tuple[int, ...]]:
    """从文件名推断 shape

    命名约定: {bm}_{name}_{NxMxK}.{qtype}.bin
    示例: encoder_linear_0_weight_64x64.bfp4.bin → (64, 64)
    """
    import os
    import re
    base = os.path.basename(path)
    # 去掉 .qtype.bin 或 .bin
    if base.endswith(".bin"):
        base = base[:-4]
    if "." in base:
        base = base.rsplit(".", 1)[0]
    # 取最后一个 _ 分隔的部分，检查是否为 NxMxK 格式
    if "_" in base:
        last = base.rsplit("_", 1)[-1]
        if re.fullmatch(r"\d+(x\d+)*", last):
            return tuple(int(d) for d in last.split("x"))
    return None


def load(
    path: str,
    fmt: str = "raw",
    qtype: Optional[str] = None,
    shape: Optional[Union[Tuple[int, ...], list]] = None,
    **kwargs,
) -> np.ndarray:
    """加载数据

    Args:
        path: 文件路径
        fmt: 文件格式 (raw, numpy)
        qtype: DUT 量化类型。指定后自动反量化为 fp32。
               若不指定，从文件名后缀推断 (如 input_0.bfp8.bin)。
               支持: bfp4, bfp8, bfp16, float16, float32
        shape: 输出 shape。BFP 类型必须指定 (用于计算 num_blocks)
        **kwargs: 传给底层格式的额外参数

    Returns:
        numpy 数组。指定 qtype 时返回 fp32。

    Example:
        >>> data = load("encoder_input_0_2x16x64.bfp8.bin")  # 全自动
        >>> data = load("input.bin", qtype="bfp8", shape=(2, 16, 64))
    """
    # 自动推断 qtype 和 shape
    if qtype is None:
        qtype = _infer_qtype(path)
    if shape is None:
        shape = _infer_shape(path)
    if qtype is None:
        if shape is not None:
            kwargs["shape"] = shape
        return get(fmt).load(path, **kwargs)

    if qtype == "float32":
        data = get(fmt).load(path, dtype=np.float32, **kwargs)
        if shape is not None:
            data = data.reshape(shape)
        return data

    if qtype == "float16":
        data = get(fmt).load(path, dtype=np.float16, **kwargs)
        data = data.astype(np.float32)
        if shape is not None:
            data = data.reshape(shape)
        return data

    if qtype in _BFP_DEFAULTS:
        if shape is None:
            raise ValueError(f"load(qtype='{qtype}') 需要指定 shape 参数")
        block_size, mantissa_bits = _BFP_DEFAULTS[qtype]
        size = 1
        for s in shape:
            size *= s
        num_blocks = ceil(size / block_size)
        packed = np.fromfile(path, dtype=np.int8)
        from aidevtools.formats.quantize import dequantize
        meta = {
            "block_size": block_size,
            "mantissa_bits": mantissa_bits,
            "num_blocks": num_blocks,
            "original_shape": tuple(shape),
        }
        return dequantize(packed, qtype, meta)

    raise ValueError(f"load: 不支持的 qtype '{qtype}'")


def _infer_name(path: str, bm: str = "") -> str:
    """从文件名提取 tensor 名

    encoder_linear_0_weight_64x64.bfp4.bin → linear_0_weight
    (去掉 bm 前缀、shape 后缀、qtype.bin 后缀)
    """
    import os
    import re
    base = os.path.basename(path)
    # 去掉 .qtype.bin 或 .bin
    if base.endswith(".bin"):
        base = base[:-4]
    if "." in base:
        base = base.rsplit(".", 1)[0]
    # 去掉 bm 前缀
    if bm and base.startswith(bm + "_"):
        base = base[len(bm) + 1:]
    # 去掉 shape 后缀
    if "_" in base:
        head, last = base.rsplit("_", 1)
        if re.fullmatch(r"\d+(x\d+)*", last):
            base = head
    return base


def load_dir(
    directory: Union[str, Path],
    bm: str = "",
) -> Dict[str, np.ndarray]:
    """自动扫描目录，加载所有 DUT bin 文件

    从文件名推断 qtype、shape、tensor 名，全自动反量化为 fp32。

    Args:
        directory: 包含 .bin 文件的目录
        bm: benchmark 前缀过滤 (如 "encoder")，
            空字符串表示加载所有 bin 文件

    Returns:
        Dict[tensor_name, fp32_array]

    Example:
        >>> tensors = load_dir("./golden/", bm="encoder")
        >>> tensors["input_0"].shape   # (2, 16, 64)
        >>> tensors["linear_0_weight"].shape  # (64, 64)
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"load_dir: '{directory}' 不是目录")

    result = {}
    for f in sorted(directory.glob("*.bin")):
        # bm 过滤
        if bm and not f.name.startswith(bm + "_"):
            continue
        # 跳过无法推断 qtype 的文件
        qtype = _infer_qtype(str(f))
        if qtype is None:
            continue
        shape = _infer_shape(str(f))
        name = _infer_name(str(f), bm=bm)
        data = load(str(f), qtype=qtype, shape=shape)
        result[name] = data
    return result


def save(path: str, data: np.ndarray, fmt: str = "raw", **kwargs):
    """保存数据"""
    get(fmt).save(path, data, **kwargs)


# 注意: 内置格式的注册已移至 formats/__init__.py
# 这样可以避免循环导入 (base <-> numpy_fmt/raw)
