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
class GlobalConfig:
    """全局配置"""
    golden_mode: str = "python"    # python | cpp
    precision: str = "quant"       # pure | quant
    seed: int = 42

    exact: ExactConfig = field(default_factory=ExactConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)

    def validate(self):
        """验证配置"""
        if self.golden_mode not in ("python", "cpp"):
            raise ValueError(f"golden_mode must be 'python' or 'cpp', got '{self.golden_mode}'")
        if self.precision not in ("pure", "quant"):
            raise ValueError(f"precision must be 'pure' or 'quant', got '{self.precision}'")


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
    exact: ExactConfig = None,
    fuzzy: FuzzyConfig = None,
) -> GlobalConfig:
    """设置全局配置"""
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
