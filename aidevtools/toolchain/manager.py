"""
工具链管理器
"""

import stat
from pathlib import Path
from typing import Dict, List, Optional

from .config import ToolchainConfig
from .download import Downloader, get_platform_suffix


class ToolchainManager:
    """
    工具链管理器

    管理外部编译器二进制的下载、安装和版本切换。

    使用示例:
        manager = ToolchainManager()
        manager.ensure_all()  # 确保所有工具已安装

        py2c_path = manager.get_path("py2c")
        c2dut_path = manager.get_path("c2dut")
    """

    def __init__(self, config: Optional[ToolchainConfig] = None):
        """
        Args:
            config: 工具链配置，不传则自动加载 toolchain.yaml
        """
        self.config = config or ToolchainConfig.load()
        self.cache_dir = self.config.cache_dir
        self._downloader: Optional[Downloader] = None

    @property
    def downloader(self) -> Downloader:
        """延迟初始化下载器"""
        if self._downloader is None:
            reg = self.config.registry
            self._downloader = Downloader(
                base_url=reg.url,
                proxy=reg.proxy,
                auth=reg.auth,
                headers=reg.headers,
                verify=reg.verify,
                timeout=reg.timeout,
                retries=reg.retries,
            )
        return self._downloader

    def ensure_all(self) -> Dict[str, str]:
        """
        确保所有配置的工具都已安装

        Returns:
            工具名到路径的映射
        """
        paths = {}
        for tool, version in self.config.versions.items():
            paths[tool] = self.ensure(tool, version)
        return paths

    def ensure(self, tool: str, version: Optional[str] = None) -> str:
        """
        确保指定工具已安装

        Args:
            tool: 工具名，如 "py2c", "c2dut"
            version: 版本号，不传则使用配置文件中的版本

        Returns:
            可执行文件路径
        """
        if version is None:
            version = self.config.get_version(tool)
            if version is None:
                raise ValueError(
                    f"Version not specified for {tool} and not found in config"
                )

        # 检查是否已安装
        bin_path = self._get_bin_path(tool, version)
        if bin_path.exists():
            return str(bin_path)

        # 下载安装
        self._install(tool, version)

        if not bin_path.exists():
            raise RuntimeError(
                f"Installation completed but binary not found: {bin_path}"
            )

        return str(bin_path)

    def get_path(self, tool: str, version: Optional[str] = None) -> str:
        """
        获取工具路径，如果未安装则自动安装

        Args:
            tool: 工具名
            version: 版本号（可选）

        Returns:
            可执行文件路径
        """
        return self.ensure(tool, version)

    def is_installed(self, tool: str, version: str) -> bool:
        """检查工具是否已安装"""
        return self._get_bin_path(tool, version).exists()

    def list_installed(self) -> Dict[str, List[str]]:
        """
        列出所有已安装的工具及其版本

        Returns:
            工具名到版本列表的映射
        """
        result: Dict[str, List[str]] = {}
        installed_dir = self.cache_dir / "installed"

        if not installed_dir.exists():
            return result

        for item in installed_dir.iterdir():
            if item.is_dir():
                # 解析目录名: tool-version-platform
                name = item.name
                parts = name.rsplit("-", 2)  # 从右边分割
                if len(parts) >= 2:
                    # 尝试提取工具名和版本
                    # 格式: py2c-1.0.0-darwin-arm64
                    tool = parts[0]
                    version = parts[1] if len(parts) > 1 else "unknown"
                    if tool not in result:
                        result[tool] = []
                    if version not in result[tool]:
                        result[tool].append(version)

        return result

    def clean(self, keep: int = 2) -> List[Path]:
        """
        清理旧版本，每个工具只保留最新的 N 个版本

        Args:
            keep: 保留的版本数

        Returns:
            被删除的目录列表
        """
        import shutil

        deleted = []
        installed = self.list_installed()

        for tool, versions in installed.items():
            if len(versions) <= keep:
                continue

            # 按版本号排序（简单字符串排序，假设语义版本）
            sorted_versions = sorted(versions, reverse=True)
            to_delete = sorted_versions[keep:]

            for version in to_delete:
                # 查找并删除对应目录
                for item in (self.cache_dir / "installed").iterdir():
                    if item.name.startswith(f"{tool}-{version}"):
                        shutil.rmtree(item)
                        deleted.append(item)

        return deleted

    def _install(self, tool: str, version: str) -> Path:
        """安装工具"""
        platform_suffix = get_platform_suffix()
        filename = f"{tool}-{version}-{platform_suffix}.tar.gz"

        # 获取期望的哈希值
        expected_hash = self.config.get_file_hash(filename)

        # 下载并解压
        extract_dir = self.downloader.download_and_extract(
            filename,
            self.cache_dir,
            expected_hash,
        )

        # 设置可执行权限
        bin_path = self._find_binary(extract_dir, tool)
        if bin_path:
            bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        return extract_dir

    def _get_bin_path(self, tool: str, version: str) -> Path:
        """获取二进制文件路径"""
        platform_suffix = get_platform_suffix()
        install_dir = self.cache_dir / "installed" / f"{tool}-{version}-{platform_suffix}"

        # 尝试多种可能的路径
        candidates = [
            install_dir / "bin" / tool,
            install_dir / tool,
            install_dir / f"{tool}.exe",  # Windows
            install_dir / "bin" / f"{tool}.exe",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # 默认返回标准路径
        return install_dir / "bin" / tool

    def _find_binary(self, extract_dir: Path, tool: str) -> Optional[Path]:
        """在解压目录中查找二进制文件"""
        candidates = [
            extract_dir / "bin" / tool,
            extract_dir / tool,
            extract_dir / f"{tool}.exe",
            extract_dir / "bin" / f"{tool}.exe",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # 递归查找
        for path in extract_dir.rglob(tool):
            if path.is_file():
                return path

        return None


# 全局管理器实例
_manager: Optional[ToolchainManager] = None


def get_manager() -> ToolchainManager:
    """获取全局工具链管理器实例"""
    global _manager
    if _manager is None:
        _manager = ToolchainManager()
    return _manager


def reset_manager():
    """重置全局管理器（用于测试）"""
    global _manager
    _manager = None
