"""
工具链管理模块测试
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aidevtools.toolchain import (
    ToolchainConfig,
    ToolchainManager,
    Downloader,
    get_platform_suffix,
    verify_hash,
    create_default_config,
)


class TestGetPlatformSuffix:
    """平台后缀测试"""

    def test_returns_string(self):
        suffix = get_platform_suffix()
        assert isinstance(suffix, str)
        assert "-" in suffix  # 格式: os-arch

    @patch("platform.system")
    @patch("platform.machine")
    def test_darwin_arm64(self, mock_machine, mock_system):
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        assert get_platform_suffix() == "darwin-arm64"

    @patch("platform.system")
    @patch("platform.machine")
    def test_linux_x64(self, mock_machine, mock_system):
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        assert get_platform_suffix() == "linux-x64"


class TestVerifyHash:
    """哈希校验测试"""

    def test_sha256_correct(self, tmp_path):
        # 创建测试文件
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        # hello 的 sha256
        expected = "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert verify_hash(test_file, expected) is True

    def test_sha256_incorrect(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        expected = "sha256:wronghash"
        assert verify_hash(test_file, expected) is False

    def test_without_prefix(self, tmp_path):
        """不带算法前缀，默认 sha256"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        # 不带 sha256: 前缀
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert verify_hash(test_file, expected) is True


class TestToolchainConfig:
    """配置加载测试"""

    def test_default_config(self):
        config = ToolchainConfig()
        assert config.registry.url == ""
        assert config.versions == {}
        assert config.files == {}

    def test_load_from_dict(self):
        data = {
            "registry": {
                "url": "https://example.com",
                "timeout": 30,
            },
            "versions": {
                "py2c": "1.0.0",
                "c2dut": "2.0.0",
            },
            "files": {
                "py2c-1.0.0-linux-x64.tar.gz": "sha256:abc123",
            },
        }
        config = ToolchainConfig._from_dict(data)

        assert config.registry.url == "https://example.com"
        assert config.registry.timeout == 30
        assert config.versions["py2c"] == "1.0.0"
        assert config.get_version("py2c") == "1.0.0"
        assert config.get_file_hash("py2c-1.0.0-linux-x64.tar.gz") == "sha256:abc123"

    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "toolchain.yaml"
        config_file.write_text("""
registry:
  url: "https://test.com"
versions:
  py2c: "1.2.0"
""")
        config = ToolchainConfig.load(str(config_file))
        assert config.registry.url == "https://test.com"
        assert config.versions["py2c"] == "1.2.0"

    def test_env_expansion(self, tmp_path):
        """测试环境变量展开"""
        os.environ["TEST_TOKEN"] = "secret123"
        try:
            config_file = tmp_path / "toolchain.yaml"
            config_file.write_text("""
registry:
  url: "https://test.com"
  auth:
    type: bearer
    token: "${TEST_TOKEN}"
""")
            config = ToolchainConfig.load(str(config_file))
            assert config.registry.auth["token"] == "secret123"
        finally:
            del os.environ["TEST_TOKEN"]


class TestCreateDefaultConfig:
    """默认配置文件生成测试"""

    def test_creates_file(self, tmp_path):
        config_path = tmp_path / "toolchain.yaml"
        result = create_default_config(config_path)

        assert result == config_path
        assert config_path.exists()
        content = config_path.read_text()
        assert "registry:" in content
        assert "versions:" in content


class TestDownloader:
    """下载器测试"""

    def test_init_without_httpx(self):
        """httpx 未安装时的行为"""
        # 这个测试在 httpx 已安装的环境中会跳过
        pass

    def test_init_with_options(self):
        """测试初始化参数"""
        pytest.importorskip("httpx")

        downloader = Downloader(
            base_url="https://example.com",
            proxy="http://proxy:8080",
            auth={"type": "bearer", "token": "mytoken"},
            timeout=30.0,
        )
        assert downloader.base_url == "https://example.com"
        assert downloader.timeout == 30.0


class TestToolchainManager:
    """工具链管理器测试"""

    def test_init_with_config(self, tmp_path):
        config = ToolchainConfig()
        config.cache_dir = tmp_path / "cache"

        manager = ToolchainManager(config)
        assert manager.cache_dir == tmp_path / "cache"

    def test_is_installed_false(self, tmp_path):
        config = ToolchainConfig()
        config.cache_dir = tmp_path / "cache"
        manager = ToolchainManager(config)

        assert manager.is_installed("py2c", "1.0.0") is False

    def test_list_installed_empty(self, tmp_path):
        config = ToolchainConfig()
        config.cache_dir = tmp_path / "cache"
        manager = ToolchainManager(config)

        assert manager.list_installed() == {}

    @patch("aidevtools.toolchain.manager.get_platform_suffix")
    def test_get_bin_path(self, mock_platform, tmp_path):
        mock_platform.return_value = "linux-x64"

        config = ToolchainConfig()
        config.cache_dir = tmp_path / "cache"
        manager = ToolchainManager(config)

        bin_path = manager._get_bin_path("py2c", "1.0.0")
        expected = tmp_path / "cache/installed/py2c-1.0.0-linux-x64/bin/py2c"
        assert bin_path == expected
