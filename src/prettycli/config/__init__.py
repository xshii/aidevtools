"""配置加载模块"""
from pathlib import Path
from typing import Optional
import yaml

__all__ = ["load_config", "DEFAULT_CONFIG_PATH", "USER_CONFIG_PATHS"]

# 默认配置文件路径
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"

# 用户配置文件查找路径（按优先级）
USER_CONFIG_PATHS = [
    Path(".prettycli.yaml"),           # 当前目录
    Path(".prettycli.yml"),
    Path.home() / ".prettycli.yaml",   # 用户目录
    Path.home() / ".prettycli.yml",
    Path.home() / ".config/prettycli/config.yaml",  # XDG 风格
]


def load_config(config_path: Optional[Path] = None) -> dict:
    """加载配置

    优先级（从高到低）：
    1. 指定的 config_path
    2. 当前目录 .prettycli.yaml
    3. 用户目录 ~/.prettycli.yaml
    4. XDG 目录 ~/.config/prettycli/config.yaml
    5. 默认配置

    Args:
        config_path: 可选的配置文件路径

    Returns:
        合并后的配置字典
    """
    # 加载默认配置
    config = {}
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}

    # 查找用户配置
    user_config_path = config_path
    if not user_config_path:
        for path in USER_CONFIG_PATHS:
            if path.exists():
                user_config_path = path
                break

    # 合并用户配置
    if user_config_path and user_config_path.exists():
        with open(user_config_path) as f:
            user_config = yaml.safe_load(f) or {}
            config.update(user_config)

    return config
