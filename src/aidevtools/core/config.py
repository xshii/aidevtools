"""全局配置模块"""
from dataclasses import dataclass, field
from typing import Optional
import threading


@dataclass
class ExactConfig:
    """精确比对配置"""
    max_abs: float = 0.0      # 允许的最大绝对误差 (0=bit级精确)
    max_count: int = 0        # 允许超阈值的元素个数 (0=全部精确)


@dataclass
class FuzzyConfig:
    """模糊比对配置"""
    atol: float = 1e-5        # 绝对误差阈值
    rtol: float = 1e-3        # 相对误差阈值
    min_qsnr: float = 30.0    # 最小 QSNR (dB)
    min_cosine: float = 0.999 # 最小余弦相似度


@dataclass
class CpuGoldenConfig:
    """CPU Golden 配置"""
    dtype: str = "gfp16"           # gfp4 | gfp8 | gfp16
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
        valid_dtypes = ("gfp4", "gfp8", "gfp16")
        if self.cpu_golden.dtype not in valid_dtypes:
            raise ValueError(f"cpu_golden.dtype must be one of {valid_dtypes}")


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
    """
    设置全局配置

    Args:
        golden_mode: "python" 或 "cpp"
        precision: "pure" 或 "quant"
        seed: 随机种子
        compute_golden: 是否计算 golden
        cpu_golden_dtype: CPU golden 默认 dtype (gfp4/gfp8/gfp16)
        cpu_golden_dtype_matmul_a: matmul A 矩阵类型 (混合精度)
        cpu_golden_dtype_matmul_b: matmul B 矩阵类型 (混合精度)
        cpu_golden_dtype_matmul_out: matmul 输出类型 (混合精度)
        exact: 精确比对配置
        fuzzy: 模糊比对配置

    Returns:
        更新后的配置
    """
    global _global_config
    with _config_lock:
        if _global_config is None:
            _global_config = GlobalConfig()

        if golden_mode is not None:
            _global_config.golden_mode = golden_mode
        if precision is not None:
            _global_config.precision = precision
        if seed is not None:
            _global_config.seed = seed
        if compute_golden is not None:
            _global_config.compute_golden = compute_golden

        # CPU Golden 配置
        if cpu_golden_dtype is not None:
            _global_config.cpu_golden.dtype = cpu_golden_dtype
        if cpu_golden_dtype_matmul_a is not None:
            _global_config.cpu_golden.dtype_matmul_a = cpu_golden_dtype_matmul_a
        if cpu_golden_dtype_matmul_b is not None:
            _global_config.cpu_golden.dtype_matmul_b = cpu_golden_dtype_matmul_b
        if cpu_golden_dtype_matmul_out is not None:
            _global_config.cpu_golden.dtype_matmul_out = cpu_golden_dtype_matmul_out

        if exact is not None:
            _global_config.exact = exact
        if fuzzy is not None:
            _global_config.fuzzy = fuzzy

        _global_config.validate()
        return _global_config


def reset_config():
    """重置为默认配置"""
    global _global_config
    with _config_lock:
        _global_config = GlobalConfig()
