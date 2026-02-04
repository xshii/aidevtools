"""
工具链配置管理
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

import yaml


@dataclass
class RegistryConfig:
    """注册表配置"""

    url: str = ""
    proxy: Optional[str] = None
    auth: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    verify: bool = True
    timeout: float = 60.0
    retries: int = 3


@dataclass
class ToolchainConfig:
    """工具链配置"""

    registry: RegistryConfig = field(default_factory=RegistryConfig)
    versions: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)  # filename -> hash
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache/aidevtools/toolchain")

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "ToolchainConfig":
        """
        加载配置文件

        查找顺序:
        1. 指定的 config_path
        2. 当前目录的 toolchain.yaml
        3. 项目根目录的 toolchain.yaml (向上查找)
        4. 使用默认配置
        """
        if config_path:
            path = Path(config_path)
            if path.exists():
                return cls._from_file(path)
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # 向上查找到项目根目录
        current = Path.cwd()
        for _ in range(10):  # 最多向上 10 级
            for name in ["toolchain.yaml", "toolchain.yml"]:
                candidate = current / name
                if candidate.exists():
                    return cls._from_file(candidate)
            parent = current.parent
            if parent == current:
                break
            current = parent

        # 未找到配置文件，返回默认配置
        return cls()

    @classmethod
    def _from_file(cls, path: Path) -> "ToolchainConfig":
        """从文件加载配置"""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ToolchainConfig":
        """从字典创建配置"""
        config = cls()

        # 解析 registry
        if reg := data.get("registry"):
            config.registry = RegistryConfig(
                url=reg.get("url", ""),
                proxy=_expand_env(reg.get("proxy")),
                auth=_expand_auth(reg.get("auth")),
                headers=reg.get("headers"),
                verify=reg.get("verify", True),
                timeout=reg.get("timeout", 60.0),
                retries=reg.get("retries", 3),
            )

        # 解析 versions
        config.versions = data.get("versions", {})

        # 解析 files
        config.files = data.get("files", {})

        # 解析 cache_dir
        if cache_dir := data.get("cache_dir"):
            config.cache_dir = Path(os.path.expanduser(os.path.expandvars(cache_dir)))

        return config

    def get_file_hash(self, filename: str) -> Optional[str]:
        """获取文件的期望哈希值"""
        return self.files.get(filename)

    def get_version(self, tool: str) -> Optional[str]:
        """获取工具的版本"""
        return self.versions.get(tool)


def _expand_env(value: Optional[str]) -> Optional[str]:
    """展开环境变量"""
    if value is None:
        return None
    return os.path.expandvars(value)


def _expand_auth(auth: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """展开认证配置中的环境变量"""
    if auth is None:
        return None

    result = {}
    for key, value in auth.items():
        if isinstance(value, str):
            result[key] = os.path.expandvars(value)
        else:
            result[key] = value
    return result


# 默认配置模板
DEFAULT_CONFIG_TEMPLATE = """\
# 工具链配置文件
# 放置在项目根目录，文件名: toolchain.yaml

registry:
  # 下载地址
  url: "https://registry.example.com/toolchain"

  # 代理 (可选，支持环境变量)
  # proxy: "${HTTP_PROXY}"

  # 认证 (可选)
  # auth:
  #   type: bearer  # basic | bearer
  #   token: "${TOOLCHAIN_TOKEN}"
  #   # 或 Basic Auth:
  #   # type: basic
  #   # username: "user"
  #   # password: "${TOOLCHAIN_PASSWORD}"

  # SSL 验证 (可选)
  # verify: true  # false 或 CA 证书路径

  # 超时和重试
  timeout: 60
  retries: 3

# 工具版本
versions:
  py2c: "1.0.0"
  c2dut: "1.0.0"

# 文件哈希 (用于校验)
# 格式: "文件名": "sha256:哈希值"
files:
  # Linux x64
  # "py2c-1.0.0-linux-x64.tar.gz": "sha256:..."
  # "c2dut-1.0.0-linux-x64.tar.gz": "sha256:..."
  # macOS arm64
  # "py2c-1.0.0-darwin-arm64.tar.gz": "sha256:..."
  # "c2dut-1.0.0-darwin-arm64.tar.gz": "sha256:..."
"""


def create_default_config(path: Path = None) -> Path:
    """创建默认配置文件"""
    if path is None:
        path = Path.cwd() / "toolchain.yaml"

    path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    return path
