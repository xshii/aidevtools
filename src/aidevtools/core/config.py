"""全局配置模块"""
import threading
from dataclasses import dataclass, field
from typing import Optional

from aidevtools.core.constants import (
    DEFAULT_ATOL,
    DEFAULT_GFLOAT_TYPE,
    DEFAULT_MIN_COSINE,
    DEFAULT_MIN_QSNR,
    DEFAULT_RTOL,
    GFLOAT_TYPES,
)


@dataclass
class ExactConfig:
    """精确比对配置"""
    max_abs: float = 0.0      # 允许的最大绝对误差 (0=bit级精确)
    max_count: int = 0        # 允许超阈值的元素个数 (0=全部精确)


@dataclass
class FuzzyConfig:
    """模糊比对配置"""
    atol: float = DEFAULT_ATOL           # 绝对误差阈值
    rtol: float = DEFAULT_RTOL           # 相对误差阈值
    min_qsnr: float = DEFAULT_MIN_QSNR   # 最小 QSNR (dB)
    min_cosine: float = DEFAULT_MIN_COSINE  # 最小余弦相似度


@dataclass
class CpuGoldenConfig:
    """CPU Golden 配置"""
    dtype: str = DEFAULT_GFLOAT_TYPE       # gfp4 | gfp8 | gfp16
    dtype_matmul_a: Optional[str] = None   # matmul A 矩阵类型 (混合精度)
    dtype_matmul_b: Optional[str] = None   # matmul B 矩阵类型 (混合精度)
    dtype_matmul_out: Optional[str] = None # matmul 输出类型 (混合精度)


@dataclass
class GlobalConfig:
    """全局配置"""
    golden_mode: str = "python"    # python | cpp
    precision: str = "quant"       # pure | quant
    seed: int = 42
    compute_golden: bool = True    # 是否计算 golden

    cpu_golden: CpuGoldenConfig = field(default_factory=CpuGoldenConfig)
    exact: ExactConfig = field(default_factory=ExactConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)

    def validate(self):
        """验证配置"""
        if self.golden_mode not in ("python", "cpp"):
            raise ValueError(f"golden_mode must be 'python' or 'cpp', got '{self.golden_mode}'")
        if self.precision not in ("pure", "quant"):
            raise ValueError(f"precision must be 'pure' or 'quant', got '{self.precision}'")
        if self.cpu_golden.dtype not in GFLOAT_TYPES:
            raise ValueError(f"cpu_golden.dtype must be one of {GFLOAT_TYPES}")


# 全局配置实例 (线程安全)
_config_lock = threading.Lock()
_global_config: Optional[GlobalConfig] = None


def get_config() -> GlobalConfig:
    """获取全局配置"""
    global _global_config
    with _config_lock:
        if _global_config is None:
            _global_config = GlobalConfig()
        return _global_config


def set_config(
    golden_mode: str = None,
    precision: str = None,
    seed: int = None,
    compute_golden: bool = None,
    cpu_golden_dtype: str = None,
    cpu_golden_dtype_matmul_a: str = None,
    cpu_golden_dtype_matmul_b: str = None,
    cpu_golden_dtype_matmul_out: str = None,
    exact: ExactConfig = None,
    fuzzy: FuzzyConfig = None,
) -> GlobalConfig:
    """设置全局配置"""
    global _global_config
    with _config_lock:
        if _global_config is None:
            _global_config = GlobalConfig()

        # 顶层配置映射
        top_level = {"golden_mode": golden_mode, "precision": precision, "seed": seed, "compute_golden": compute_golden, "exact": exact, "fuzzy": fuzzy}
        for key, value in top_level.items():
            if value is not None:
                setattr(_global_config, key, value)

        # CPU Golden 子配置映射
        cpu_golden_attrs = {"dtype": cpu_golden_dtype, "dtype_matmul_a": cpu_golden_dtype_matmul_a, "dtype_matmul_b": cpu_golden_dtype_matmul_b, "dtype_matmul_out": cpu_golden_dtype_matmul_out}
        for key, value in cpu_golden_attrs.items():
            if value is not None:
                setattr(_global_config.cpu_golden, key, value)

        _global_config.validate()
        return _global_config


def reset_config():
    """重置为默认配置"""
    global _global_config
    with _config_lock:
        _global_config = GlobalConfig()
