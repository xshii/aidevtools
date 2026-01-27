"""QuantizedTensor - 带精度状态跟踪的张量包装类

用于在多算子连续计算中跟踪数据的精度状态，避免：
1. 重复量化（数据已经是目标精度）
2. 忘记量化（权重等数据未量化就进入计算）

典型用法：
    # 在数据源头量化
    x = quantize(input_data, "gfp16")
    w = quantize(weight_data, "gfp16")

    # 连续计算（自动检查精度）
    y = F.matmul(x, w)  # 输出也是 QuantizedTensor
    z = F.gelu(y)       # 继续保持精度状态

    # 最终获取结果
    result = z.numpy()
"""
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np

# 支持的精度类型
GFloatType = Literal["gfp4", "gfp8", "gfp16"]
BFPType = Literal["bfp4", "bfp8", "bfp16"]
QuantizedType = Literal["gfp4", "gfp8", "gfp16", "bfp4", "bfp8", "bfp16"]

# CPU Golden 可执行文件路径
_GOLDEN_DIR = Path(__file__).parent.parent / "golden"
_CPU_GOLDEN_PATH = _GOLDEN_DIR / "cpu_golden"
_CPU_GOLDEN_BFP_PATH = _GOLDEN_DIR / "cpu_golden_bfp"


def _is_gfloat_type(dtype: str) -> bool:
    """判断是否是 GFloat 类型"""
    return dtype in ("gfp4", "gfp8", "gfp16")


def _is_bfp_type(dtype: str) -> bool:
    """判断是否是 BFP 类型"""
    return dtype in ("bfp4", "bfp8", "bfp16")


def _get_executable(dtype: str) -> Path:
    """根据 dtype 获取对应的可执行文件"""
    if _is_gfloat_type(dtype):
        return _CPU_GOLDEN_PATH
    elif _is_bfp_type(dtype):
        return _CPU_GOLDEN_BFP_PATH
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


@dataclass
class QuantizedTensor:
    """带精度状态跟踪的张量

    Attributes:
        data: numpy 数组（fp32 存储，但值可能已量化到目标精度）
        dtype: 当前精度状态
               - None: 原始 fp32，未量化
               - "gfp4"/"gfp8"/"gfp16": 已量化为 GFloat 精度
               - "bfp4"/"bfp8"/"bfp16": 已量化为 BFP 精度
        shape: 张量形状
    """
    data: np.ndarray
    dtype: Optional[QuantizedType] = None

    def __post_init__(self):
        # 确保数据是 fp32
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def is_quantized(self) -> bool:
        """是否已量化"""
        return self.dtype is not None

    def numpy(self) -> np.ndarray:
        """返回底层 numpy 数组"""
        return self.data

    def quantize(self, target_dtype: QuantizedType) -> "QuantizedTensor":
        """量化到目标精度

        如果已经是目标精度，直接返回自身（避免重复量化）。

        Args:
            target_dtype: 目标精度类型

        Returns:
            量化后的 QuantizedTensor
        """
        if self.dtype == target_dtype:
            # 已经是目标精度，不需要重复量化
            return self

        if self.dtype is not None and self.dtype != target_dtype:
            warnings.warn(
                f"Re-quantizing from {self.dtype} to {target_dtype}. "
                f"This may introduce additional precision loss.",
                UserWarning,
                stacklevel=2
            )

        # 调用 C++ quantize 命令
        quantized_data = _run_quantize(self.data, target_dtype)
        return QuantizedTensor(data=quantized_data, dtype=target_dtype)

    def ensure_quantized(self, target_dtype: QuantizedType, warn: bool = True) -> "QuantizedTensor":
        """确保数据已量化到目标精度

        如果未量化，自动量化并发出警告。
        如果已是目标精度，直接返回。

        Args:
            target_dtype: 目标精度类型
            warn: 是否在自动量化时发出警告

        Returns:
            量化后的 QuantizedTensor
        """
        if self.dtype == target_dtype:
            return self

        if warn and self.dtype is None:
            warnings.warn(
                f"Input data is not quantized. Auto-quantizing to {target_dtype}. "
                f"For better control, quantize data at source using quantize().",
                UserWarning,
                stacklevel=2
            )

        return self.quantize(target_dtype)

    def __repr__(self) -> str:
        dtype_str = self.dtype if self.dtype else "fp32 (unquantized)"
        return f"QuantizedTensor(shape={self.shape}, dtype={dtype_str})"

    # 支持 numpy 数组操作
    def reshape(self, *shape) -> "QuantizedTensor":
        """重塑形状"""
        return QuantizedTensor(data=self.data.reshape(*shape), dtype=self.dtype)

    def transpose(self, *axes) -> "QuantizedTensor":
        """转置"""
        return QuantizedTensor(data=self.data.transpose(*axes), dtype=self.dtype)

    def __getitem__(self, key) -> "QuantizedTensor":
        """切片"""
        return QuantizedTensor(data=self.data[key], dtype=self.dtype)


def _run_quantize(data: np.ndarray, dtype: QuantizedType) -> np.ndarray:
    """调用 C++ quantize 命令

    Args:
        data: 输入数据（fp32）
        dtype: 目标精度类型

    Returns:
        量化后的数据（fp32 存储，但值已量化）
    """
    executable = _get_executable(dtype)

    if not executable.exists():
        raise FileNotFoundError(
            f"CPU Golden executable not found: {executable}\n"
            f"Please build it first: cd {_GOLDEN_DIR / 'cpp'} && ./build.sh"
        )

    flat_data = data.astype(np.float32).flatten()
    size = flat_data.size

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.bin"
        output_path = tmpdir / "output.bin"

        # 保存 fp32 输入
        flat_data.tofile(input_path)

        # 调用 C++ quantize
        cmd = [str(executable), "quantize", dtype, str(input_path), str(output_path), str(size)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"quantize failed: {result.stderr}")

        # 读取 fp32 输出
        output = np.fromfile(output_path, dtype=np.float32)

    return output.reshape(data.shape)


def quantize(
    data: Union[np.ndarray, "QuantizedTensor"],
    dtype: QuantizedType,
) -> QuantizedTensor:
    """将数据量化到目标精度

    这是推荐的量化入口函数。在数据源头调用一次，后续算子
    会自动保持精度状态。

    Args:
        data: 输入数据（np.ndarray 或 QuantizedTensor）
        dtype: 目标精度类型

    Returns:
        量化后的 QuantizedTensor

    Examples:
        >>> x = quantize(np.random.randn(4, 8).astype(np.float32), "gfp16")
        >>> print(x)
        QuantizedTensor(shape=(4, 8), dtype=gfp16)
    """
    if isinstance(data, QuantizedTensor):
        return data.quantize(dtype)
    else:
        tensor = QuantizedTensor(data=np.asarray(data))
        return tensor.quantize(dtype)


def ensure_quantized(
    data: Union[np.ndarray, "QuantizedTensor"],
    dtype: QuantizedType,
    warn: bool = True,
) -> QuantizedTensor:
    """确保数据已量化到目标精度

    如果输入未量化，自动量化并发出警告。
    适用于算子内部检查输入数据。

    Args:
        data: 输入数据
        dtype: 目标精度类型
        warn: 是否在自动量化时发出警告

    Returns:
        量化后的 QuantizedTensor
    """
    if isinstance(data, QuantizedTensor):
        return data.ensure_quantized(dtype, warn=warn)
    else:
        if warn:
            warnings.warn(
                f"Input is raw numpy array, not QuantizedTensor. "
                f"Auto-quantizing to {dtype}. "
                f"For better control, use quantize() at data source.",
                UserWarning,
                stacklevel=2
            )
        tensor = QuantizedTensor(data=np.asarray(data))
        return tensor.quantize(dtype)


def wrap_output(data: np.ndarray, dtype: Optional[QuantizedType]) -> QuantizedTensor:
    """包装算子输出

    算子内部使用，将计算结果包装为 QuantizedTensor。

    Args:
        data: 算子输出数据
        dtype: 输出精度（通常与输入相同）

    Returns:
        包装后的 QuantizedTensor
    """
    return QuantizedTensor(data=data, dtype=dtype)
