"""C++ 扩展加载器

提供统一的 C++ 扩展加载和诊断功能。
"""
import sys
from pathlib import Path
from typing import Any, Optional

from aidevtools.core.log import logger


class CppExtensionLoader:
    """C++ 扩展加载器

    用法:
        loader = CppExtensionLoader(
            name="BFP",
            module_path=Path(__file__).parent,
            import_path="aidevtools.formats.custom.bfp",
            module_name="bfp_golden",
        )

        # 检查是否可用
        if loader.is_available():
            result = loader.module.some_function()

        # 或者强制检查（不可用时抛异常）
        loader.check()
        result = loader.module.some_function()
    """

    def __init__(
        self,
        name: str,
        module_path: Path,
        import_path: str,
        module_name: str,
    ):
        """
        Args:
            name: 扩展名称 (用于日志和错误信息)
            module_path: 模块目录路径 (用于诊断)
            import_path: 导入路径 (如 "aidevtools.formats.custom.bfp")
            module_name: 模块名称 (如 "bfp_golden")
        """
        self.name = name
        self.module_path = module_path
        self.cpp_dir = module_path / "cpp"
        self.import_path = import_path
        self.module_name = module_name

        self._module: Optional[Any] = None
        self._import_error: Optional[Exception] = None
        self._import_detail: str = ""

        self._try_load()

    def _try_load(self):
        """尝试加载 C++ 扩展"""
        try:
            import importlib
            full_path = f"{self.import_path}.{self.module_name}"
            self._module = importlib.import_module(full_path)
            logger.debug(f"{self.name} C++ Golden API 加载成功")
        except ImportError as e:
            self._import_error = e
            self._collect_diagnostics()
            logger.warning(f"{self.name} C++ Golden API 加载失败: {e}")

    def _collect_diagnostics(self):
        """收集诊断信息"""
        so_files = list(self.module_path.glob("*.so")) + list(self.module_path.glob("*.pyd"))
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        expected_suffix = f"cpython-{sys.version_info.major}{sys.version_info.minor}"

        if not so_files:
            self._import_detail = "未找到编译产物 (.so/.pyd 文件)"
        else:
            so_names = [f.name for f in so_files]
            if not any(expected_suffix in name for name in so_names):
                self._import_detail = (
                    f"Python 版本不匹配\n"
                    f"  当前 Python: {python_version} (需要 {expected_suffix})\n"
                    f"  已有文件: {so_names}"
                )
            else:
                self._import_detail = f"文件存在但加载失败: {so_names}"

    @property
    def module(self) -> Any:
        """获取 C++ 模块 (不可用时返回 None)"""
        return self._module

    def is_available(self) -> bool:
        """检查 C++ 扩展是否可用"""
        return self._module is not None

    def check(self):
        """检查 C++ 扩展是否可用，不可用时抛出详细异常"""
        if self._module is None:
            error_msg = (
                f"{self.name} C++ Golden API 加载失败\n"
                f"{'=' * 50}\n"
                f"原因: {self._import_detail}\n"
                f"原始错误: {self._import_error}\n"
                f"{'=' * 50}\n"
                f"目录: {self.module_path}\n"
                f"{'=' * 50}\n"
                f"解决方法:\n"
                f"  cd {self.cpp_dir}\n"
                f"  bash build.sh\n"
            )
            raise ImportError(error_msg)
