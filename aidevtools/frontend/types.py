"""
前端类型定义

提供统一的 Tensor、OpContext 等类型。
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


class DType(Enum):
    """数据类型"""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    BFP16 = "bfp16"
    BFP8 = "bfp8"
    GFP16 = "gfp16"
    GFP8 = "gfp8"
    INT8 = "int8"

    @classmethod
    def from_str(cls, s: str) -> "DType":
        """从字符串创建"""
        mapping = {
            "fp32": cls.FP32,
            "float32": cls.FP32,
            "fp16": cls.FP16,
            "float16": cls.FP16,
            "bf16": cls.BF16,
            "bfloat16": cls.BF16,
            "bfp16": cls.BFP16,
            "bfp8": cls.BFP8,
            "gfp16": cls.GFP16,
            "gfp8": cls.GFP8,
            "int8": cls.INT8,
        }
        return mapping.get(s.lower(), cls.FP32)


class DistType(Enum):
    """数据分布类型"""

    NORMAL = "normal"  # 正态分布
    UNIFORM = "uniform"  # 均匀分布
    ZEROS = "zeros"  # 全零
    ONES = "ones"  # 全一
    XAVIER = "xavier"  # Xavier 初始化
    KAIMING = "kaiming"  # Kaiming 初始化


@dataclass
class TensorMeta:
    """Tensor 元信息"""

    shape: Tuple[int, ...]
    dtype: DType = DType.FP32
    name: str = ""
    qtype: Optional[str] = None  # 量化类型
    scale: Optional[float] = None  # 量化 scale
    zero_point: Optional[int] = None  # 量化 zero point


@dataclass
class Tensor:
    """
    统一 Tensor 类型

    封装 fp32 数据和可选的量化数据。
    """

    data: np.ndarray  # fp32 数据
    quant_data: Optional[bytes] = None  # 量化后的字节数据
    meta: TensorMeta = field(default_factory=lambda: TensorMeta(shape=()))

    def __post_init__(self):
        if not self.meta.shape:
            self.meta = TensorMeta(shape=self.data.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        """获取 Tensor 形状"""
        return self.data.shape

    @property
    def dtype(self) -> DType:
        """获取数据类型"""
        return self.meta.dtype

    @property
    def name(self) -> str:
        """获取 Tensor 名称"""
        return self.meta.name

    @classmethod
    def from_numpy(
        cls, data: np.ndarray, name: str = "", dtype: DType = DType.FP32
    ) -> "Tensor":
        """从 numpy 数组创建"""
        return cls(
            data=data.astype(np.float32),
            meta=TensorMeta(shape=data.shape, dtype=dtype, name=name),
        )

    @classmethod
    def empty(
        cls, shape: Tuple[int, ...], dtype: DType = DType.FP32, name: str = ""
    ) -> "Tensor":
        """创建空 Tensor"""
        data = np.empty(shape, dtype=np.float32)
        return cls(
            data=data,
            meta=TensorMeta(shape=shape, dtype=dtype, name=name),
        )

    @classmethod
    def zeros(
        cls, shape: Tuple[int, ...], dtype: DType = DType.FP32, name: str = ""
    ) -> "Tensor":
        """创建全零 Tensor"""
        data = np.zeros(shape, dtype=np.float32)
        return cls(
            data=data,
            meta=TensorMeta(shape=shape, dtype=dtype, name=name),
        )

    def numpy(self) -> np.ndarray:
        """转换为 numpy 数组"""
        return self.data

    def save(self, path: Union[str, Path]):
        """保存到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 fp32 数据
        np.save(str(path.with_suffix(".npy")), self.data)

        # 保存量化数据 (如有)
        if self.quant_data is not None:
            path.with_suffix(".bin").write_bytes(self.quant_data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Tensor":
        """从文件加载"""
        path = Path(path)
        data = np.load(str(path.with_suffix(".npy")))

        quant_data = None
        bin_path = path.with_suffix(".bin")
        if bin_path.exists():
            quant_data = bin_path.read_bytes()

        return cls(data=data, quant_data=quant_data)


@dataclass
class OpContext:
    """
    算子上下文

    控制算子执行时的量化、比对等行为。
    """

    dtype: DType = DType.FP32  # 默认数据类型
    enable_gc: bool = True  # 是否启用 GC 比对
    gc_level: int = 2  # GC 级别 (1=step, 2=segment, 3=full)
    name: str = ""  # 上下文名称


@dataclass
class CompileConfig:
    """编译配置"""

    output_dir: str = "./build"
    golden_dir: str = "./golden"
    target: str = "dut"  # "dut" | "sim"
    optimize: int = 2  # 优化级别 0-3
    verbose: bool = False
    # 工具链版本
    py2c_version: Optional[str] = None
    c2dut_version: Optional[str] = None
