"""
下载器 - 基于 httpx 实现
"""

import hashlib
import platform
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import httpx
except ImportError:
    httpx = None


def _check_httpx():
    if httpx is None:
        raise ImportError(
            "httpx is required for toolchain management. "
            "Install with: pip install httpx"
        )


def get_platform_suffix() -> str:
    """获取当前平台后缀"""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        arch = "arm64" if machine == "arm64" else "x64"
        return f"darwin-{arch}"
    elif system == "linux":
        arch = "arm64" if machine == "aarch64" else "x64"
        return f"linux-{arch}"
    elif system == "windows":
        return "windows-x64"
    else:
        raise RuntimeError(f"Unsupported platform: {system}-{machine}")


def verify_hash(path: Path, expected: str) -> bool:
    """
    校验文件哈希

    Args:
        path: 文件路径
        expected: 格式为 "算法:哈希值"，如 "sha256:abc123..."

    Returns:
        是否匹配
    """
    if ":" not in expected:
        # 默认 sha256
        algo, hash_val = "sha256", expected
    else:
        algo, hash_val = expected.split(":", 1)

    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest().lower() == hash_val.lower()


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    """
    解压归档文件

    支持 .tar.gz, .tgz, .tar, .zip

    Returns:
        解压后的目录
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()

    if name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    elif name.endswith(".tar"):
        with tarfile.open(archive_path, "r:") as tf:
            tf.extractall(extract_dir)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    return extract_dir


class Downloader:
    """
    HTTP 下载器

    支持:
    - 代理
    - Basic Auth / Bearer Token
    - 自定义 headers
    - SSL 证书配置
    - 超时和重试
    """

    def __init__(
        self,
        base_url: str = "",
        proxy: Optional[str] = None,
        auth: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        verify: bool = True,
        timeout: float = 60.0,
        retries: int = 3,
    ):
        """
        Args:
            base_url: 基础 URL
            proxy: 代理地址，如 "http://proxy:8080"
            auth: 认证配置
                - {"type": "basic", "username": "...", "password": "..."}
                - {"type": "bearer", "token": "..."}
            headers: 自定义请求头
            verify: SSL 验证，True/False 或 CA 证书路径
            timeout: 超时秒数
            retries: 重试次数
        """
        _check_httpx()

        self.base_url = base_url.rstrip("/") if base_url else ""
        self.timeout = timeout
        self.retries = retries

        # 构建 httpx.Client 参数
        self._client_kwargs: Dict[str, Any] = {}

        if proxy:
            self._client_kwargs["proxy"] = proxy

        if auth:
            auth_type = auth.get("type", "none")
            if auth_type == "basic":
                self._client_kwargs["auth"] = (
                    auth.get("username", ""),
                    auth.get("password", ""),
                )
            elif auth_type == "bearer":
                headers = headers or {}
                headers["Authorization"] = f"Bearer {auth.get('token', '')}"

        if headers:
            self._client_kwargs["headers"] = headers

        if verify is False:
            self._client_kwargs["verify"] = False
        elif isinstance(verify, str):
            self._client_kwargs["verify"] = verify

        self._client_kwargs["timeout"] = timeout

    def download(
        self,
        url_or_filename: str,
        output_path: Path,
        expected_hash: Optional[str] = None,
    ) -> Path:
        """
        下载文件

        Args:
            url_or_filename: 完整 URL 或文件名（会拼接 base_url）
            output_path: 输出文件路径
            expected_hash: 期望的哈希值（可选）

        Returns:
            下载后的文件路径

        Raises:
            httpx.HTTPError: 下载失败
            ValueError: 哈希校验失败
        """
        # 构建完整 URL
        if url_or_filename.startswith(("http://", "https://")):
            url = url_or_filename
        else:
            if not self.base_url:
                raise ValueError("base_url not set and url_or_filename is not a full URL")
            url = f"{self.base_url}/{url_or_filename}"

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 下载（带重试）
        for attempt in range(self.retries):
            try:
                with httpx.Client(**self._client_kwargs) as client:
                    with client.stream("GET", url, follow_redirects=True) as resp:
                        resp.raise_for_status()
                        with open(output_path, "wb") as f:
                            for chunk in resp.iter_bytes(chunk_size=8192):
                                f.write(chunk)
                break
            except httpx.HTTPError:
                if attempt < self.retries - 1:
                    continue
                raise

        # 校验哈希
        if expected_hash:
            if not verify_hash(output_path, expected_hash):
                output_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Hash verification failed for {output_path}. "
                    f"Expected: {expected_hash}"
                )

        return output_path

    def download_and_extract(
        self,
        url_or_filename: str,
        cache_dir: Path,
        expected_hash: Optional[str] = None,
    ) -> Path:
        """
        下载并解压

        Args:
            url_or_filename: 完整 URL 或文件名
            cache_dir: 缓存目录
            expected_hash: 期望的哈希值

        Returns:
            解压后的目录路径
        """
        # 提取文件名
        filename = url_or_filename.split("/")[-1]
        archive_path = cache_dir / "downloads" / filename

        # 解压目录名（去掉扩展名）
        extract_name = filename
        for ext in [".tar.gz", ".tgz", ".tar", ".zip"]:
            if extract_name.endswith(ext):
                extract_name = extract_name[:-len(ext)]
                break
        extract_dir = cache_dir / "installed" / extract_name

        # 如果已解压，直接返回
        if extract_dir.exists():
            return extract_dir

        # 下载
        if not archive_path.exists():
            self.download(url_or_filename, archive_path, expected_hash)
        elif expected_hash and not verify_hash(archive_path, expected_hash):
            # 缓存的文件哈希不匹配，重新下载
            archive_path.unlink()
            self.download(url_or_filename, archive_path, expected_hash)

        # 解压
        extract_archive(archive_path, extract_dir)

        return extract_dir
