"""TracedTensor - 带溯源和精度信息的张量包装类

合并原 QuantizedTensor 功能，新增：
- source_op: 来源算子名（用于追踪计算图）
- is_weight: 是否为权重

用于在多算子连续计算中：
1. 跟踪数据的精度状态，避免重复量化
2. 记录数据来源，用于生成测试用例
3. 构建计算图，支持混合比对模式

典型用法：
    # 创建带精度追踪的张量
    x = traced(input_data, "gfp16")
    w = traced(weight_data, "gfp16", is_weight=True)

    # 连续计算（自动追踪来源）
    y = F.matmul(x, w)  # y.source_op = "matmul_0"
    z = F.gelu(y)       # z.source_op = "gelu_0"

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
class TracedTensor:
    """带溯源和精度信息的张量（合并原 QuantizedTensor）

    Attributes:
        data: numpy 数组（fp32 存储，但值可能已量化到目标精度）
        dtype: 当前精度状态
               - None: 原始 fp32，未量化
               - "gfp4"/"gfp8"/"gfp16": 已量化为 GFloat 精度
               - "bfp4"/"bfp8"/"bfp16": 已量化为 BFP 精度
        source_op: 来源算子名（如 "matmul_0"），None 表示输入数据
        is_weight: 是否为权重
    """
    data: np.ndarray
    dtype: Optional[QuantizedType] = None
    source_op: Optional[str] = None
    is_weight: bool = False

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
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> "TracedTensor":
        """转置（.T 属性，支持 weight.T 语法）"""
        return TracedTensor(
            data=self.data.T,
            dtype=self.dtype,
            source_op=self.source_op,
            is_weight=self.is_weight,
        )

    @property
    def is_quantized(self) -> bool:
        """是否已量化"""
        return self.dtype is not None

    def numpy(self) -> np.ndarray:
        """返回底层 numpy 数组"""
        return self.data

    def quantize(self, target_dtype: QuantizedType) -> "TracedTensor":
        """量化到目标精度

        如果已经是目标精度，直接返回自身（避免重复量化）。

        Args:
            target_dtype: 目标精度类型

        Returns:
            量化后的 TracedTensor
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
        return TracedTensor(
            data=quantized_data,
            dtype=target_dtype,
            source_op=self.source_op,
            is_weight=self.is_weight,
        )

    def ensure_quantized(self, target_dtype: QuantizedType, warn: bool = True) -> "TracedTensor":
        """确保数据已量化到目标精度

        如果未量化，自动量化并发出警告。
        如果已是目标精度，直接返回。

        Args:
            target_dtype: 目标精度类型
            warn: 是否在自动量化时发出警告

        Returns:
            量化后的 TracedTensor
        """
        if self.dtype == target_dtype:
            return self

        if warn and self.dtype is None:
            warnings.warn(
                f"Input data is not quantized. Auto-quantizing to {target_dtype}. "
                f"For better control, quantize data at source using traced().",
                UserWarning,
                stacklevel=2
            )

        return self.quantize(target_dtype)

    def with_source(self, source_op: str) -> "TracedTensor":
        """创建带新来源标记的副本

        Args:
            source_op: 来源算子名

        Returns:
            带新来源标记的 TracedTensor
        """
        return TracedTensor(
            data=self.data,
            dtype=self.dtype,
            source_op=source_op,
            is_weight=self.is_weight,
        )

    def __repr__(self) -> str:
        dtype_str = self.dtype if self.dtype else "fp32 (unquantized)"
        source_str = f", source={self.source_op}" if self.source_op else ""
        weight_str = ", weight=True" if self.is_weight else ""
        return f"TracedTensor(shape={self.shape}, dtype={dtype_str}{source_str}{weight_str})"

    # 支持 numpy 数组操作
    def reshape(self, *shape) -> "TracedTensor":
        """重塑形状"""
        return TracedTensor(
            data=self.data.reshape(*shape),
            dtype=self.dtype,
            source_op=self.source_op,
            is_weight=self.is_weight,
        )

    def transpose(self, *axes) -> "TracedTensor":
        """转置"""
        return TracedTensor(
            data=self.data.transpose(*axes),
            dtype=self.dtype,
            source_op=self.source_op,
            is_weight=self.is_weight,
        )

    def __getitem__(self, key) -> "TracedTensor":
        """切片"""
        return TracedTensor(
            data=self.data[key],
            dtype=self.dtype,
            source_op=self.source_op,
            is_weight=self.is_weight,
        )

    def flatten(self) -> "TracedTensor":
        """展平"""
        return TracedTensor(
            data=self.data.flatten(),
            dtype=self.dtype,
            source_op=self.source_op,
            is_weight=self.is_weight,
        )

    def astype(self, dtype) -> "TracedTensor":
        """类型转换（仅影响底层 numpy 数组）"""
        return TracedTensor(
            data=self.data.astype(dtype),
            dtype=self.dtype,
            source_op=self.source_op,
            is_weight=self.is_weight,
        )

    def __array__(self, dtype=None) -> np.ndarray:
        """支持 np.asarray(TracedTensor) 直接转换为 numpy 数组"""
        if dtype is None:
            return self.data
        return self.data.astype(dtype)


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


def traced(
    data: Union[np.ndarray, "TracedTensor"],
    dtype: Optional[QuantizedType] = None,
    is_weight: bool = False,
) -> TracedTensor:
    """创建 TracedTensor

    这是推荐的入口函数。在数据源头调用一次，后续算子
    会自动保持精度状态和追踪信息。

    Args:
        data: 输入数据（np.ndarray 或 TracedTensor）
        dtype: 目标精度类型，None 表示不量化
        is_weight: 是否为权重

    Returns:
        TracedTensor

    Examples:
        >>> x = traced(np.random.randn(4, 8).astype(np.float32), "gfp16")
        >>> print(x)
        TracedTensor(shape=(4, 8), dtype=gfp16)

        >>> w = traced(weight_data, "gfp16", is_weight=True)
        >>> print(w)
        TracedTensor(shape=(768, 768), dtype=gfp16, weight=True)
    """
    if isinstance(data, TracedTensor):
        # 已经是 TracedTensor
        tensor = TracedTensor(
            data=data.data,
            dtype=data.dtype,
            source_op=data.source_op,
            is_weight=is_weight or data.is_weight,
        )
    else:
        tensor = TracedTensor(
            data=np.asarray(data),
            dtype=None,
            source_op=None,
            is_weight=is_weight,
        )

    if dtype is not None:
        tensor = tensor.quantize(dtype)

    return tensor


def ensure_traced(
    data: Union[np.ndarray, TracedTensor],
    dtype: QuantizedType,
    warn: bool = True,
) -> TracedTensor:
    """确保数据是 TracedTensor 且已量化到目标精度

    如果输入未量化，自动量化并发出警告。
    适用于算子内部检查输入数据。

    Args:
        data: 输入数据
        dtype: 目标精度类型
        warn: 是否在自动量化时发出警告

    Returns:
        量化后的 TracedTensor
    """
    if isinstance(data, TracedTensor):
        return data.ensure_quantized(dtype, warn=warn)
    else:
        if warn:
            warnings.warn(
                f"Input is raw numpy array, not TracedTensor. "
                f"Auto-quantizing to {dtype}. "
                f"For better control, use traced() at data source.",
                UserWarning,
                stacklevel=2
            )
        tensor = TracedTensor(data=np.asarray(data))
        return tensor.quantize(dtype)


def wrap_traced_output(
    data: np.ndarray,
    dtype: Optional[QuantizedType],
    source_op: str,
) -> TracedTensor:
    """包装算子输出为 TracedTensor

    算子内部使用，将计算结果包装为 TracedTensor 并记录来源。

    Args:
        data: 算子输出数据
        dtype: 输出精度（通常与输入相同）
        source_op: 来源算子名（如 "matmul_0"）

    Returns:
        包装后的 TracedTensor
    """
    return TracedTensor(
        data=data,
        dtype=dtype,
        source_op=source_op,
        is_weight=False,  # 输出不是权重
    )


def quantize(
    data: Union[np.ndarray, TracedTensor],
    dtype: QuantizedType,
) -> TracedTensor:
    """将数据量化到目标精度（兼容旧 API）

    等同于 traced(data, dtype)。保留此函数以兼容现有代码。

    Args:
        data: 输入数据
        dtype: 目标精度类型

    Returns:
        量化后的 TracedTensor
    """
    return traced(data, dtype)


def ensure_quantized(
    data: Union[np.ndarray, TracedTensor],
    dtype: QuantizedType,
    warn: bool = True,
) -> TracedTensor:
    """确保数据已量化到目标精度（兼容旧 API）

    等同于 ensure_traced。保留此函数以兼容现有代码。
    """
    return ensure_traced(data, dtype, warn=warn)


def wrap_output(data: np.ndarray, dtype: Optional[QuantizedType]) -> TracedTensor:
    """包装算子输出（兼容旧 API）

    注意：此函数不设置 source_op，请使用 wrap_traced_output 以获得完整追踪。
    """
    return TracedTensor(data=data, dtype=dtype)
