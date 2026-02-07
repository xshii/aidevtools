"""
前端类型定义

提供统一的 Tensor、OpContext 等类型。
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


class DType(Enum):
    """数据类型

    本地格式: FP32, FP16, BF16, INT16, INT8
    硬件定制格式: BFP16, BFP8, GFP16, GFP8
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT16 = "int16"
    INT8 = "int8"
    # 硬件定制格式
    BFP16 = "bfp16"
    BFP8 = "bfp8"
    GFP16 = "gfp16"
    GFP8 = "gfp8"

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
            "int16": cls.INT16,
            "int8": cls.INT8,
            "bfp16": cls.BFP16,
            "bfp8": cls.BFP8,
            "gfp16": cls.GFP16,
            "gfp8": cls.GFP8,
        }
        return mapping.get(s.lower(), cls.FP32)

    @property
    def is_local(self) -> bool:
        """是否为本地格式 (fp32/fp16/bf16/int16/int8)"""
        return self in (DType.FP32, DType.FP16, DType.BF16, DType.INT16, DType.INT8)

    @property
    def is_hw_quant(self) -> bool:
        """是否为硬件定制量化格式 (bfp/gfp)"""
        return self in (DType.BFP16, DType.BFP8, DType.GFP16, DType.GFP8)


class DistType(Enum):
    """数据分布类型"""

    NORMAL = "normal"  # 正态分布
    UNIFORM = "uniform"  # 均匀分布
    ZEROS = "zeros"  # 全零
    ONES = "ones"  # 全一
    XAVIER = "xavier"  # Xavier 初始化
    KAIMING = "kaiming"  # Kaiming 初始化


@dataclass
class PrecisionConfig:
    """算子精度配置

    两层精度控制:
    1. 角色级 (默认): input_dtype / weight_dtype / compute_dtype / output_dtype
    2. 参数级 (覆盖): param_dtypes = {"x": "bfp8", "weight": "int8", ...}

    查询优先级: param_dtypes[参数名] > 角色级默认值

    用法:
        # 角色级: 所有 input 用 fp16, 所有 weight 用 bfp8
        pc = PrecisionConfig(input_dtype="fp16", weight_dtype="bfp8")

        # 参数级: 每个参数独立精度
        pc = PrecisionConfig(
            compute_dtype="fp16",
            param_dtypes={"x": "bfp8", "weight": "int8", "bias": "fp32"},
        )

        # 混用: 角色级做默认, 参数级覆盖个别
        pc = PrecisionConfig(
            input_dtype="fp16",
            weight_dtype="bfp8",
            param_dtypes={"bias": "fp32"},  # bias 不跟 weight_dtype
        )

        # 查询
        pc.get_dtype("x")               # → "bfp8" (param_dtypes 命中)
        pc.get_dtype("gamma")            # → "fp16" (fallback input_dtype)
        pc.get_dtype("weight", is_weight=True)  # → "int8" (param_dtypes 命中)
    """

    compute_dtype: str = "fp32"    # 计算精度
    input_dtype: str = "fp32"      # 输入精度
    output_dtype: str = "fp32"     # 输出精度
    weight_dtype: str = "fp32"     # 权重精度 (默认跟 input_dtype)

    # 逐参数精度覆盖 (优先级最高)
    param_dtypes: Dict[str, str] = field(default_factory=dict)

    # 量化感知随机数开关
    qa_aware: bool = False         # 是否启用量化感知随机数
    qa_center: float = 1.0         # 量化感知随机数中心值
    qa_amplitude: float = 0.5      # 量化感知随机数波动幅度
    qa_max_ratio: float = 10.0     # 输出最大值/最小值比值上限

    def __post_init__(self):
        """默认 weight_dtype 跟 input_dtype"""
        if self.weight_dtype == "fp32" and self.input_dtype != "fp32":
            self.weight_dtype = self.input_dtype

    def get_dtype(self, param_name: str, is_weight: bool = False) -> str:
        """查询参数精度

        优先级: param_dtypes[param_name] > weight_dtype/input_dtype

        Args:
            param_name: 参数名 (如 "x", "weight", "bias", "gamma")
            is_weight: 是否为权重参数 (fallback 到 weight_dtype)

        Returns:
            精度字符串 (如 "bfp8", "fp16", "int8")
        """
        if param_name in self.param_dtypes:
            return self.param_dtypes[param_name]
        return self.weight_dtype if is_weight else self.input_dtype

    @classmethod
    def from_dict(cls, d: dict) -> "PrecisionConfig":
        """从字典创建"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @property
    def is_mixed(self) -> bool:
        """是否为混合精度"""
        dtypes = {self.compute_dtype, self.input_dtype,
                  self.output_dtype, self.weight_dtype}
        dtypes.update(self.param_dtypes.values())
        return len(dtypes) > 1


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
