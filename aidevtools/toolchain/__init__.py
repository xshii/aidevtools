"""
工具链管理模块

管理外部编译器（py2c, c2dut 等）的下载、安装和版本管理。

基本使用:
    from aidevtools.toolchain import ensure_toolchain, get_compiler

    # 确保所有工具已安装
    ensure_toolchain()

    # 获取编译器路径
    py2c = get_compiler("py2c")
    c2dut = get_compiler("c2dut")

配置文件 (toolchain.yaml):
    registry:
      url: "https://registry.example.com/toolchain"
      # proxy: "http://proxy:8080"  # 可选
      # auth:
      #   type: bearer
      #   token: "${TOOLCHAIN_TOKEN}"

    versions:
      py2c: "1.0.0"
      c2dut: "1.0.0"

    files:
      "py2c-1.0.0-linux-x64.tar.gz": "sha256:..."
"""

from .config import ToolchainConfig, create_default_config
from .download import Downloader, get_platform_suffix, verify_hash
from .manager import ToolchainManager, get_manager, reset_manager

__all__ = [
    # 核心类
    "ToolchainManager",
    "ToolchainConfig",
    "Downloader",
    # 便捷函数
    "ensure_toolchain",
    "get_compiler",
    "list_installed",
    "clean_cache",
    # 工具函数
    "get_platform_suffix",
    "verify_hash",
    "create_default_config",
    # 内部
    "get_manager",
    "reset_manager",
]


def ensure_toolchain(config_path: str = None) -> dict:
    """
    确保所有配置的工具链已安装

    Args:
        config_path: 配置文件路径（可选）

    Returns:
        工具名到路径的映射

    Example:
        paths = ensure_toolchain()
        print(paths)  # {"py2c": "/path/to/py2c", "c2dut": "/path/to/c2dut"}
    """
    if config_path:
        config = ToolchainConfig.load(config_path)
        manager = ToolchainManager(config)
    else:
        manager = get_manager()

    return manager.ensure_all()


def get_compiler(name: str, version: str = None) -> str:
    """
    获取编译器路径，如果未安装则自动下载安装

    Args:
        name: 编译器名称，如 "py2c", "c2dut"
        version: 版本号（可选，默认使用配置文件中的版本）

    Returns:
        编译器可执行文件的完整路径

    Example:
        py2c = get_compiler("py2c")
        c2dut = get_compiler("c2dut", "2.0.0")
    """
    return get_manager().get_path(name, version)


def list_installed() -> dict:
    """
    列出已安装的工具及其版本

    Returns:
        工具名到版本列表的映射

    Example:
        installed = list_installed()
        print(installed)  # {"py2c": ["1.0.0", "1.1.0"], "c2dut": ["2.0.0"]}
    """
    return get_manager().list_installed()


def clean_cache(keep: int = 2) -> list:
    """
    清理旧版本缓存

    Args:
        keep: 每个工具保留的版本数

    Returns:
        被删除的目录列表
    """
    return get_manager().clean(keep)
